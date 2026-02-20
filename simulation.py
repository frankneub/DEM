import math
import time
import random
from collections import deque
import numpy as np
from numba import jit
from particle import Particle


@jit(nopython=True, cache=True)
def resolve_particle_collisions(positions, velocities, masses, radii, 
                                 restitution, friction, collision_pairs):
    """JIT-compiled particle collision resolution."""
    for pair_idx in range(len(collision_pairs)):
        i, j = int(collision_pairs[pair_idx, 0]), int(collision_pairs[pair_idx, 1])
        
        d = positions[j] - positions[i]
        dist_sq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
        
        if dist_sq < 1e-24:
            continue
        
        dist = math.sqrt(dist_sq)
        overlap = radii[i] + radii[j] - dist
        
        if overlap <= 0:
            continue
        
        # Normal vector
        nvec = d / dist
        
        # Relative velocity
        rv = velocities[j] - velocities[i]
        vn = rv[0]*nvec[0] + rv[1]*nvec[1] + rv[2]*nvec[2]
        
        # Compute impulse
        e = restitution
        j_impulse = -(1.0 + e) * vn
        denom = (1.0 / masses[i] + 1.0 / masses[j])
        
        if denom == 0:
            continue
        
        j_impulse /= denom
        
        # Apply impulse
        impulse = j_impulse * nvec
        velocities[i] -= impulse / masses[i]
        velocities[j] += impulse / masses[j]
        
        # Positional correction
        corr = nvec * (overlap * 0.5 + 1e-5)
        positions[i] -= corr
        positions[j] += corr
        
        # Tangential friction
        tangent = rv - vn * nvec
        tn_sq = tangent[0]*tangent[0] + tangent[1]*tangent[1] + tangent[2]*tangent[2]
        
        if tn_sq > 1e-18:
            tn = math.sqrt(tn_sq)
            tdir = tangent / tn
            ft = -friction * j_impulse
            velocities[i] -= ft * tdir / masses[i]
            velocities[j] += ft * tdir / masses[j]
    
    return positions, velocities


class DEMSimulation:
    """Handles all DEM (Discrete Element Method) simulation logic."""
    
    def __init__(self):
        # Physics parameters
        self.particles = []
        self.gravity = np.array([0.0, -9.81, 0.0])
        self.restitution = 0.3
        self.friction = 0.3
        self.floor_friction = 0.3
        self.rolling_resistance = 0.05
        self.floor_y = 0.0
        self.time_step = 1.0 / 120.0

        # Injector / spawning
        self.mass_per_hour = 1000.0
        self.spawn_region = ((-0.2, 0, -0.2), (0.2, 0, 0.2))
        self.spawn_height = 2.0
        self.spawn_queue = deque()
        self.last_spawn_time = time.time()
        self.spawned_count = 0
        self.ppsec = 0
        self.time = 0
        # Particle size distribution defaults
        self.min_radius = 0.02
        self.max_radius = 0.08
        self.radius_mean = 0.05
        self.radius_std = 0.01
        self.radius_distribution = 'Uniform'  # 'Uniform' or 'Normal'

    def step(self, dt):
        """Execute one simulation step."""
        self._spawn_particles(dt)
        self._integrate(dt)

    def _spawn_particles(self, dt):
        """Spawn particles based on mass_per_hour rate."""
        # Simple model: choose spheres with random radius and density
        mass_per_sec = self.mass_per_hour / 3600.0
        mass_to_emit = mass_per_sec * dt
        self.time += dt
        self.ppsec = self.spawned_count / max(self.time, 1e-6)
        # Keep a leftover buffer in queue
        self.spawn_queue.append(mass_to_emit)
        total = sum(self.spawn_queue)
        
        # Spawn while enough mass for one typical particle
        while total > 0:
            # Choose a radius according to configured distribution
            if getattr(self, 'radius_distribution', 'Uniform') == 'Normal':
                r = random.gauss(self.radius_mean, self.radius_std)
            else:
                r = random.uniform(self.min_radius, self.max_radius)
            # clamp and enforce small positive radius
            r = max(0.001, min(r, getattr(self, 'max_radius', 0.08)))
            density = 2500.0
            m = (4.0 / 3.0) * math.pi * r**3 * density
            
            if total >= m * 0.1:
                # Spawn one particle
                x = random.uniform(self.spawn_region[0][0], self.spawn_region[1][0])
                z = random.uniform(self.spawn_region[0][2], self.spawn_region[1][2])
                pos = np.array([x, self.spawn_height, z])
                vel = np.array([random.uniform(-0.2, 0.2), 0.0, random.uniform(-0.2, 0.2)])
                p = Particle(pos, vel, r, m, rolling_resistance=self.rolling_resistance)
                
                # Approximate initial angular velocity for rolling
                v = np.array(vel, dtype=np.float64)
                up = np.array([0.0, 1.0, 0.0])
                v_horiz = np.array([v[0], 0.0, v[2]])
                speed = np.linalg.norm(v_horiz)
                
                if speed > 1e-9:
                    axis = np.cross(up, v_horiz)
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 1e-12:
                        axis = axis / axis_norm
                        omega_mag = speed / max(r, 1e-6)
                        p.angular_vel = axis * omega_mag
                else:
                    p.angular_vel = np.array([0.0, 0.0, 0.0])
                
                self.particles.append(p)
                self.spawned_count += 1
                total -= m
            else:
                break
        
        # Reassign leftover
        self.spawn_queue.clear()
        if total > 0:
            self.spawn_queue.append(total)

    def _integrate(self, dt):
        """Integrate physics: velocities, collisions, positions."""
        n = len(self.particles)
        if n == 0:
            return
        
        # Simple Python operations (fast enough without JIT)
        # Integrate velocities
        for p in self.particles:
            p.vel += self.gravity * dt
        
        # Collision resolution: particle-particle
        # Build collision pairs using spatial hash
        max_r = max((p.radius for p in self.particles), default=0.05)
        cell_size = max(0.001, max_r * 2.0)

        grid = {}
        
        def cell_coords(pos):
            return (int(math.floor(pos[0] / cell_size)),
                    int(math.floor(pos[1] / cell_size)),
                    int(math.floor(pos[2] / cell_size)))

        # Insert particles into grid
        for idx, p in enumerate(self.particles):
            key = cell_coords(p.pos)
            grid.setdefault(key, []).append(idx)

        # Neighbor offsets
        neighbor_offsets = [(dx, dy, dz) for dx in (-1, 0, 1)
                                           for dy in (-1, 0, 1)
                                           for dz in (-1, 0, 1)]

        # Collect collision pairs
        collision_pairs = []
        for key, indices in grid.items():
            for i in indices:
                for off in neighbor_offsets:
                    nk = (key[0] + off[0], key[1] + off[1], key[2] + off[2])
                    if nk not in grid:
                        continue
                    for j in grid[nk]:
                        if j <= i:
                            continue
                        collision_pairs.append((i, j))
        
        # Convert to NumPy array for JIT function
        if len(collision_pairs) > 0:
            collision_pairs_array = np.array(collision_pairs, dtype=np.int64)
            positions = np.array([p.pos for p in self.particles], dtype=np.float64)
            velocities = np.array([p.vel for p in self.particles], dtype=np.float64)
            masses = np.array([p.mass for p in self.particles], dtype=np.float64)
            radii = np.array([p.radius for p in self.particles], dtype=np.float64)
            
            # JIT-compiled collision resolution
            positions, velocities = resolve_particle_collisions(
                positions, velocities, masses, radii,
                self.restitution, self.friction, collision_pairs_array
            )
            
            # Update particles with resolved collisions
            for i, p in enumerate(self.particles):
                p.pos = positions[i]
                p.vel = velocities[i]

        # Integrate positions
        for p in self.particles:
            p.pos += p.vel * dt
        
        # Ground collision
        for p in self.particles:
            if p.pos[1] - p.radius < self.floor_y:
                p.pos[1] = self.floor_y + p.radius
                if p.vel[1] < 0:
                    p.vel[1] = -p.vel[1] * self.restitution
                p.vel[0] *= (1.0 - self.floor_friction)
                p.vel[2] *= (1.0 - self.floor_friction)

        # Prune old particles far away
        self.particles = [p for p in self.particles if p.pos[1] > -5 and np.linalg.norm(p.pos) < 200]
