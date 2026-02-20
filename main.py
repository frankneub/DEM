import sys
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

from simulation import DEMSimulation


def velocity_to_color(v, vmin, vmax):
    # Ensure v is within bounds and normalize
    t = np.clip((v - vmin) / (vmax - vmin + 1e-9), 0, 1)

    # Define color anchors as a NumPy array for faster math
    cmap = np.array([
        (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 1, 1), (0, 0, 1), (0.5, 0, 0.9)
    ])

    n = len(cmap) - 1
    scaled_t = t * n
    idx = int(scaled_t)
    frac = scaled_t - idx

    # Handle the edge case where t = 1.0
    if idx >= n:
        return tuple(cmap[-1])

    c = cmap[idx] * (1 - frac) + cmap[idx + 1] * frac
    return tuple(c)


class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(1000, 700)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.running = False

        # Camera / trackball
        self.distance = 20.0
        self.azimuth = 45.0
        self.elevation = 20.0
        self.last_pos = None

        # DEM simulation
        self.simulation = DEMSimulation()

        # Visual mapping
        self.vmin = 0.0
        self.vmax = 10.0
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.fps_timer = QtCore.QTimer(self)
        self.fps_timer.timeout.connect(self._update_fps)
        self.fps_timer.start(1000)  # Update FPS every 1 second

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.12, 0.12, 0.12, 1.0)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (4.0, 8.0, 4.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(1.0, h), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # reposition any overlay legend widget inside the GL view
        margin = 12
        if hasattr(self, 'legend_widget') and self.legend_widget is not None:
            lw = self.legend_widget
            # make legend roughly 60% of the GL height (about 3x previous min)
            desired_h = int(self.height() * 0.6)
            desired_h = max(lw.minimumHeight(), min(desired_h, self.height() - 2 * margin))
            lw.setFixedHeight(desired_h)
            # set legend width to a small fraction of GL width (clamped)
            desired_w = int(self.width() * 0.025)
            desired_w = max(lw.minimumWidth(), min(desired_w, self.width() - 2 * margin))
            lw.setFixedWidth(desired_w)
            lw.move(self.width() - lw.width() - margin, margin)
            lw.raise_()
        
        # reposition labels
        self.reposition_labels()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # camera transform: orbit
        glTranslatef(0, 0, -self.distance)
        glRotatef(self.elevation, 1, 0, 0)
        glRotatef(self.azimuth, 0, 1, 0)

        # ground plane
        self.draw_ground()

        # draw particles
        for p in self.simulation.particles:
            glPushMatrix()
            glTranslatef(*p.pos)
            glColor3f(*p.color)
            quad = gluNewQuadric()
            gluSphere(quad, p.radius, 16, 12)
            gluDeleteQuadric(quad)
            glPopMatrix()

    def draw_ground(self):
        glDisable(GL_LIGHTING)
        glColor3f(0.25, 0.25, 0.25)
        s = 20.0
        glBegin(GL_QUADS)
        glVertex3f(-s, self.simulation.floor_y, -s)
        glVertex3f(s, self.simulation.floor_y, -s)
        glVertex3f(s, self.simulation.floor_y, s)
        glVertex3f(-s, self.simulation.floor_y, s)
        glEnd()
        glEnable(GL_LIGHTING)

    def start(self):
        if not self.running:
            self.timer.start(int(self.simulation.time_step * 1000))
            self.running = True

    def stop(self):
        if self.running:
            self.timer.stop()
            self.running = False

    def step(self):
        dt = self.simulation.time_step
        self.simulation.step(dt)
        # Update color mapping based on particle velocities
        speeds = [np.linalg.norm(p.vel) for p in self.simulation.particles]
        if speeds:
            self.vmax = max(self.vmax, max(speeds))
            self.vmin = min(self.vmin, min(speeds))
        for p in self.simulation.particles:
            c = velocity_to_color(np.linalg.norm(p.vel), self.vmin, max(self.vmax, 1e-6))
            p.color = c
        self.frame_count += 1
        self.update()

    def _spawn_particles(self, dt):
        # simple model: choose spheres with random radius and density
        mass_per_sec = self.mass_per_hour / 3600.0
        mass_to_emit = mass_per_sec * dt
        # keep a leftover buffer in queue
        self.spawn_queue.append(mass_to_emit)
        total = sum(self.spawn_queue)
        # spawn while enough mass for one typical particle
        while total > 0:
            # choose a radius (0.05-0.2 m)
            r = random.uniform(0.02, 0.08)
            density = 2500.0
            m = (4.0 / 3.0) * math.pi * r**3 * density
            if total >= m * 0.1:
                # spawn one particle (we only deduct its mass fraction)
                x = random.uniform(self.spawn_region[0][0], self.spawn_region[1][0])
                z = random.uniform(self.spawn_region[0][2], self.spawn_region[1][2])
                pos = np.array([x, self.spawn_height, z])
                vel = np.array([random.uniform(-0.2, 0.2), 0.0, random.uniform(-0.2, 0.2)])
                p = Particle(pos, vel, r, m, rolling_resistance=self.rolling_resistance)
                # approximate initial angular velocity for rolling: omega = cross(up, v)/r
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
        # reassign leftover
        self.spawn_queue.clear()
        if total > 0:
            self.spawn_queue.append(total)

    def _integrate(self, dt):
        n = len(self.particles)
        if n == 0:
            return
        # integrate velocities
        for p in self.particles:
            p.vel += self.gravity * dt
            # apply per-particle rolling resistance as rotational damping
            # Rolling resistance affects angular velocity, not linear velocity.
            p.angular_vel *= (1.0 - p.rolling_resistance * dt)
        # collision resolution: particle-particle
        # Use a simple spatial-hash (uniform grid) to avoid O(n^2) checks.
        # Choose cell size based on largest particle diameter so neighbors are covered.
        max_r = max((p.radius for p in self.particles), default=0.05)
        cell_size = max(0.001, max_r * 2.0)

        grid = {}
        def cell_coords(pos):
            return (int(math.floor(pos[0] / cell_size)),
                    int(math.floor(pos[1] / cell_size)),
                    int(math.floor(pos[2] / cell_size)))

        # insert particles into grid
        for idx, p in enumerate(self.particles):
            key = cell_coords(p.pos)
            grid.setdefault(key, []).append(idx)

        # neighbor offsets (including current cell)
        neighbor_offsets = [(dx, dy, dz) for dx in (-1, 0, 1)
                                           for dy in (-1, 0, 1)
                                           for dz in (-1, 0, 1)]

        # for each particle, only check neighbors in same/adjacent cells
        for key, indices in grid.items():
            for i in indices:
                pi = self.particles[i]
                for off in neighbor_offsets:
                    nk = (key[0] + off[0], key[1] + off[1], key[2] + off[2])
                    if nk not in grid:
                        continue
                    for j in grid[nk]:
                        # ensure each pair is handled once
                        if j <= i:
                            continue
                        pj = self.particles[j]
                        d = pj.pos - pi.pos
                        dist = np.linalg.norm(d)
                        if dist <= 1e-12:
                            continue
                        overlap = pi.radius + pj.radius - dist
                        if overlap > 0:
                            # normal
                            nvec = d / dist
                            # relative velocity
                            rv = pj.vel - pi.vel
                            vn = np.dot(rv, nvec)
                            # compute impulse scalar
                            e = min(self.restitution, self.restitution)
                            j_impulse = -(1.0 + e) * vn
                            denom = (1.0 / pi.mass + 1.0 / pj.mass)
                            if denom == 0:
                                continue
                            j_impulse /= denom
                            # apply impulse
                            impulse = j_impulse * nvec
                            pi.vel -= impulse / pi.mass
                            pj.vel += impulse / pj.mass
                            # positional correction to remove sinking
                            corr = nvec * (overlap * 0.5 + 1e-5)
                            pi.pos -= corr
                            pj.pos += corr
                            # tangential friction: simple Coulomb-ish
                            tangent = rv - vn * nvec
                            tn = np.linalg.norm(tangent)
                            if tn > 1e-9:
                                tdir = tangent / tn
                                ft = -self.friction * j_impulse
                                pi.vel -= ft * tdir / pi.mass
                                pj.vel += ft * tdir / pj.mass

        # integrate positions and collisions with ground
        for p in self.particles:
            p.pos += p.vel * dt
            # ground collision
            if p.pos[1] - p.radius < self.floor_y:
                # push above ground
                p.pos[1] = self.floor_y + p.radius
                if p.vel[1] < 0:
                    p.vel[1] = -p.vel[1] * self.restitution
                # approximate friction with ground (tangential damping)
                p.vel[0] *= (1.0 - self.floor_friction)
                p.vel[2] *= (1.0 - self.floor_friction)

        # update colors based on speed
        speeds = [np.linalg.norm(p.vel) for p in self.particles]
        if speeds:
            self.vmax = max(self.vmax, max(speeds))
            self.vmin = min(self.vmin, min(speeds))
        for p in self.particles:
            c = velocity_to_color(np.linalg.norm(p.vel), self.vmin, max(self.vmax, 1e-6))
            p.color = c

        # prune old particles far away
        self.particles = [p for p in self.particles if p.pos[1] > -5 and np.linalg.norm(p.pos) < 200]

    # Mouse controls
    def mousePressEvent(self, event):
        self.last_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            self.last_pos = event.position()
            return
        dp = event.position() - self.last_pos
        buttons = event.buttons()
        if buttons & QtCore.Qt.LeftButton:
            self.azimuth += dp.x() * 0.5
            self.elevation += dp.y() * 0.5
            self.elevation = max(-89.9, min(89.9, self.elevation))
        elif buttons & QtCore.Qt.RightButton:
            self.distance += dp.y() * 0.05
            self.distance = max(1.0, min(500.0, self.distance))
        self.last_pos = event.position()
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        self.distance *= (0.95 ** delta)
        self.update()

    def reposition_labels(self):
        """Reposition all overlay labels vertically at bottom-left."""
        if hasattr(self, 'time_label') and hasattr(self, 'spawn_label') and hasattr(self, 'ppsec_label') and hasattr(self, 'fps_label'):
            margin = 12
            y_pos = self.height() - margin
            for label in [self.fps_label, self.ppsec_label, self.spawn_label, self.time_label]:
                label.adjustSize()
                y_pos -= label.height()
                label.move(margin, y_pos)
                y_pos -= 4  # spacing between labels

    def _update_fps(self):
        """Update FPS counter (called every 1 second)."""
        self.fps = self.frame_count
        self.frame_count = 0

class LegendWidget(QtWidgets.QWidget):
    def __init__(self, glwidget, parent=None):
        super().__init__(parent)
        self.gl = glwidget
        # allow narrower legend
        self.setMinimumWidth(40)
        self.setMinimumHeight(140)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        # draw a dark background so the legend is visible
        painter.fillRect(self.rect(), QtGui.QColor(40, 40, 40))
        rect = self.rect().adjusted(10, 10, -10, -10)
        # draw vertical gradient from vmax (top) to vmin (bottom)
        img = QtGui.QImage(rect.width(), rect.height(), QtGui.QImage.Format_RGB32)
        for y in range(rect.height()):
            t = 1.0 - y / max(1.0, rect.height() - 1)
            v = self.gl.vmin + t * (max(self.gl.vmax, 1e-6) - self.gl.vmin)
            c = velocity_to_color(v, self.gl.vmin, max(self.gl.vmax, 1e-6))
            r, g, b = [int(255 * x) for x in c]
            for x in range(rect.width()):
                img.setPixel(x, y, QtGui.qRgb(r, g, b))
        painter.drawImage(rect, img)
        # draw labels
        painter.setPen(QtGui.QColor(220, 220, 220))
        painter.drawText(rect.left(), rect.top() - 2, f"{self.gl.vmax:.2f} m/s")
        painter.drawText(rect.left(), rect.bottom() + 14, f"{self.gl.vmin:.2f} m/s")
        # (spawn count is now shown via an overlay QLabel at bottom-left)
        painter.end()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('3D DEM Demo')
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        self.gl = GLWidget(self)
        layout.addWidget(self.gl, 1)

        # create a right-side widget to reliably host controls and legend
        right_widget = QtWidgets.QWidget()
        right_widget.setMinimumWidth(140)
        right = QtWidgets.QVBoxLayout(right_widget)
        layout.addWidget(right_widget)

        # create legend as an overlay inside the GL view (so it's drawn on top)
        self.legend = LegendWidget(self.gl, parent=self.gl)
        # expose to GL widget so it can reposition on resize
        self.gl.legend_widget = self.legend
        self.legend.show()

        # time label (overlay at bottom-left of GL view)
        self.time_label = QtWidgets.QLabel(self.gl)
        self.time_label.setStyleSheet('color: rgb(220,220,220); background-color: rgba(60,60,60,200); padding:4px; border-radius:4px;')
        self.time_label.setText(f"Time: {self.gl.simulation.time}")
        # expose so GLWidget can reposition on resize
        self.gl.time_label = self.time_label
        self.time_label.show()

        # spawn count label (overlay at bottom-left of GL view)
        self.spawn_label = QtWidgets.QLabel(self.gl)
        self.spawn_label.setStyleSheet('color: rgb(220,220,220); background-color: rgba(60,60,60,200); padding:4px; border-radius:4px;')
        self.spawn_label.setText(f"Spawned: {self.gl.simulation.spawned_count}")
        # expose so GLWidget can reposition on resize
        self.gl.spawn_label = self.spawn_label
        self.spawn_label.show()

        # spawn count label (overlay at bottom-left of GL view)
        self.ppsec_label = QtWidgets.QLabel(self.gl)
        self.ppsec_label.setStyleSheet('color: rgb(220,220,220); background-color: rgba(60,60,60,200); padding:4px; border-radius:4px;')
        self.ppsec_label.setText(f"Particles/sec: {self.gl.simulation.ppsec}")
        # expose so GLWidget can reposition on resize
        self.gl.ppsec_label = self.ppsec_label
        self.ppsec_label.show()

        # FPS label (overlay at bottom-left of GL view)
        self.fps_label = QtWidgets.QLabel(self.gl)
        self.fps_label.setStyleSheet('color: rgb(220,220,220); background-color: rgba(60,60,60,200); padding:4px; border-radius:4px;')
        self.fps_label.setText(f"FPS: {self.gl.fps:.0f}")
        # expose so GLWidget can reposition on resize
        self.gl.fps_label = self.fps_label
        self.fps_label.show()
        # ensure legend repaints each GL frame so vmax/vmin updates are visible
        try:
            self.gl.timer.timeout.connect(self.legend.update)
            self.gl.timer.timeout.connect(self._update_time_label)
            self.gl.timer.timeout.connect(self._update_spawn_label)
            self.gl.timer.timeout.connect(self._update_ppsec_label)
            self.gl.timer.timeout.connect(self._update_fps_label)
            
        except Exception:
            pass

        frm = QtWidgets.QFormLayout()
        self.mass_spin = QtWidgets.QDoubleSpinBox()
        self.mass_spin.setRange(0, 1e7)
        self.mass_spin.setValue(1000.0)
        self.mass_spin.setSuffix(' kg/h')
        self.mass_spin.valueChanged.connect(self.on_mass_changed)
        frm.addRow('Mass / hour:', self.mass_spin)

        self.restitution_spin = QtWidgets.QDoubleSpinBox()
        self.restitution_spin.setRange(0, 1)
        self.restitution_spin.setSingleStep(0.05)
        self.restitution_spin.setValue(self.gl.simulation.restitution)
        self.restitution_spin.valueChanged.connect(self.on_restitution)
        frm.addRow('Restitution:', self.restitution_spin)

        self.friction_spin = QtWidgets.QDoubleSpinBox()
        self.friction_spin.setRange(0, 1)
        self.friction_spin.setSingleStep(0.05)
        self.friction_spin.setValue(self.gl.simulation.friction)
        self.friction_spin.valueChanged.connect(self.on_friction)
        frm.addRow('Friction:', self.friction_spin)

        self.floor_friction_spin = QtWidgets.QDoubleSpinBox()
        self.floor_friction_spin.setRange(0, 1)
        self.floor_friction_spin.setSingleStep(0.05)
        self.floor_friction_spin.setValue(self.gl.simulation.floor_friction)
        self.floor_friction_spin.valueChanged.connect(self.on_floor_friction)
        frm.addRow('Floor friction:', self.floor_friction_spin)

        self.rolling_resistance_spin = QtWidgets.QDoubleSpinBox()
        self.rolling_resistance_spin.setRange(0, 1)
        self.rolling_resistance_spin.setSingleStep(0.01)
        self.rolling_resistance_spin.setValue(self.gl.simulation.rolling_resistance)
        self.rolling_resistance_spin.valueChanged.connect(self.on_rolling_resistance)
        frm.addRow('Rolling resistance:', self.rolling_resistance_spin)

        # Particle size distribution controls
        self.dist_combo = QtWidgets.QComboBox()
        self.dist_combo.addItems(['Uniform', 'Normal'])
        self.dist_combo.setCurrentText('Uniform')
        self.dist_combo.currentTextChanged.connect(self.on_radius_distribution)
        frm.addRow('Radius distribution:', self.dist_combo)

        self.min_radius_spin = QtWidgets.QDoubleSpinBox()
        self.min_radius_spin.setRange(0.001, 1.0)
        self.min_radius_spin.setSingleStep(0.005)
        self.min_radius_spin.setValue(self.gl.simulation.min_radius)
        self.min_radius_spin.setSuffix(' m')
        self.min_radius_spin.valueChanged.connect(self.on_min_radius)
        frm.addRow('Min radius:', self.min_radius_spin)

        self.max_radius_spin = QtWidgets.QDoubleSpinBox()
        self.max_radius_spin.setRange(0.001, 1.0)
        self.max_radius_spin.setSingleStep(0.005)
        self.max_radius_spin.setValue(self.gl.simulation.max_radius)
        self.max_radius_spin.setSuffix(' m')
        self.max_radius_spin.valueChanged.connect(self.on_max_radius)
        frm.addRow('Max radius:', self.max_radius_spin)

        self.mean_radius_spin = QtWidgets.QDoubleSpinBox()
        self.mean_radius_spin.setRange(0.001, 1.0)
        self.mean_radius_spin.setSingleStep(0.005)
        self.mean_radius_spin.setValue(self.gl.simulation.radius_mean)
        self.mean_radius_spin.setSuffix(' m')
        self.mean_radius_spin.valueChanged.connect(self.on_mean_radius)
        frm.addRow('Mean radius (normal):', self.mean_radius_spin)

        self.std_radius_spin = QtWidgets.QDoubleSpinBox()
        self.std_radius_spin.setRange(0.0, 1.0)
        self.std_radius_spin.setSingleStep(0.005)
        self.std_radius_spin.setValue(self.gl.simulation.radius_std)
        self.std_radius_spin.setSuffix(' m')
        self.std_radius_spin.valueChanged.connect(self.on_std_radius)
        frm.addRow('Std dev (normal):', self.std_radius_spin)

        right.addLayout(frm)

        btns = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton('Start')
        self.start_btn.clicked.connect(self.on_start)
        btns.addWidget(self.start_btn)
        self.stop_btn = QtWidgets.QPushButton('Stop')
        self.stop_btn.clicked.connect(self.on_stop)
        btns.addWidget(self.stop_btn)
        right.addLayout(btns)

        spacer = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        right.addItem(spacer)

        self.status = QtWidgets.QLabel('Ready')
        right.addWidget(self.status)

        # start paused
        self.gl.stop()

    def on_mass_changed(self, v):
        self.gl.simulation.mass_per_hour = v

    def on_restitution(self, v):
        self.gl.simulation.restitution = v

    def on_friction(self, v):
        self.gl.simulation.friction = v

    def on_floor_friction(self, v):
        self.gl.simulation.floor_friction = v

    def on_rolling_resistance(self, v):
        # update default for newly spawned particles
        self.gl.simulation.rolling_resistance = v
        # also propagate to existing particles
        for p in self.gl.simulation.particles:
            p.rolling_resistance = v

    def on_start(self):
        self.gl.start()
        self.status.setText('Running')

    def on_radius_distribution(self, text):
        try:
            self.gl.simulation.radius_distribution = text
        except Exception:
            pass

    def on_min_radius(self, v):
        try:
            self.gl.simulation.min_radius = v
        except Exception:
            pass

    def on_max_radius(self, v):
        try:
            self.gl.simulation.max_radius = v
        except Exception:
            pass

    def on_mean_radius(self, v):
        try:
            self.gl.simulation.radius_mean = v
        except Exception:
            pass

    def on_std_radius(self, v):
        try:
            self.gl.simulation.radius_std = v
        except Exception:
            pass

    def _update_time_label(self):
        try:
            self.time_label.setText(f"Time: {self.gl.simulation.time:.2f}s")
            self.time_label.adjustSize()
            self.gl.reposition_labels()
        except Exception:
            pass

    def _update_spawn_label(self):
        try:
            self.spawn_label.setText(f"Spawned: {self.gl.simulation.spawned_count}")
            self.spawn_label.adjustSize()
            self.gl.reposition_labels()
        except Exception:
            pass

    def _update_ppsec_label(self):
        try:
            self.ppsec_label.setText(f"Particles/sec: {self.gl.simulation.ppsec:.2f}")
            self.ppsec_label.adjustSize()
            self.gl.reposition_labels()
        except Exception:
            pass

    def _update_fps_label(self):
        try:
            self.fps_label.setText(f"FPS: {self.gl.fps:.0f}")
            self.fps_label.adjustSize()
            self.gl.reposition_labels()
        except Exception:
            pass

    def on_stop(self):
        self.gl.stop()
        self.status.setText('Stopped')


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
