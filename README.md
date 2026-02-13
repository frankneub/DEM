# 3D DEM Demo (PySide6 + OpenGL)

Simple 3D discrete element method demo using PySide6 and PyOpenGL. Particles are spherical, injected from a local region 2 m above the ground. Particle color maps to speed (red=slow, violet=fast). Mouse: left-drag to orbit, right-drag to zoom, scroll to zoom.

Requirements
- Python 3.8+
- See `requirements.txt`

Run
```powershell
pip install -r requirements.txt
python "main.py"
```

Notes
- Physics is a simplified DEM (pairwise impulse resolution, simple friction and rolling damping).
- Tune `Mass / hour`, `Friction`, and `Restitution` in the UI.
