import numpy as np

class Particle:
    def __init__(self, pos, vel, radius, mass, rolling_resistance=None):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)
        self.radius = float(radius)
        self.mass = float(mass)
        self.color = (1.0, 0.0, 0.0)
        # per-particle rolling resistance (damping applied to velocity)
        if rolling_resistance is None:
            self.rolling_resistance = 0.05
        else:
            self.rolling_resistance = float(rolling_resistance)
        # angular velocity (rad/s) for rotational dynamics
        self.angular_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
