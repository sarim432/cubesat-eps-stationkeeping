# orbit_keplerian.py
# Week 1 Task: Propagate a 2-body Keplerian orbit and verify Kepler's third law
# CubeSat EPS + Station-Keeping Research — Sarim, June 2026

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Constants ──────────────────────────────────────────────────────────────────
MU = 398600.4418        # Earth gravitational parameter [km³/s²]
R_EARTH = 6371.0        # Earth radius [km]

# ── Orbit parameters ───────────────────────────────────────────────────────────
ALTITUDE = 500.0                        # [km]
a = R_EARTH + ALTITUDE                  # semi-major axis [km]

# ── Initial state vector ───────────────────────────────────────────────────────
# Satellite starts directly above the equator (on the x-axis)
# Velocity is perpendicular to position (circular orbit, y-direction)
# For a circular orbit: v = sqrt(µ/a)
v_circ = np.sqrt(MU / a)               # circular orbit velocity [km/s]

# State vector: [x, y, z, vx, vy, vz] all in km and km/s
state0 = [a, 0.0, 0.0,                 # position: on x-axis at altitude
          0.0, v_circ, 0.0]            # velocity: perpendicular, in y-direction

# ── Equations of motion ────────────────────────────────────────────────────────
def two_body_eom(t, state):
    """
    Two-body equations of motion.
    state = [x, y, z, vx, vy, vz]
    returns d(state)/dt = [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state

    r = np.sqrt(x**2 + y**2 + z**2)    # distance from Earth center [km]
    factor = -MU / r**3                 # gravitational acceleration factor

    ax = factor * x
    ay = factor * y
    az = factor * z

    return [vx, vy, vz, ax, ay, az]

# ── Kepler's third law prediction ─────────────────────────────────────────────
T_kepler = 2 * np.pi * np.sqrt(a**3 / MU)   # [seconds]
print(f"Kepler's third law predicts T = {T_kepler:.1f} s  ({T_kepler/60:.2f} min)")

# ── Propagate for exactly 3 orbital periods ───────────────────────────────────
t_span = (0, 3 * T_kepler)             # simulate 3 full orbits
t_eval = np.linspace(0, 3 * T_kepler, 3000)   # 3000 output points

solution = solve_ivp(
    fun=two_body_eom,
    t_span=t_span,
    y0=state0,
    method='RK45',
    t_eval=t_eval,
    rtol=1e-10,                         # tight tolerance for accuracy
    atol=1e-12
)

# Extract position arrays
x = solution.y[0]
y = solution.y[1]
z = solution.y[2]

# ── Verify orbital period from simulation ─────────────────────────────────────
# The satellite returns to start when it crosses x-axis again (y changes sign)
# Find first zero-crossing of y after t=0
y_sign_changes = np.where(np.diff(np.sign(y)))[0]
if len(y_sign_changes) >= 2:
    # First crossing going up = half period, second = full period
    idx1 = y_sign_changes[0]
    idx2 = y_sign_changes[1]
    T_simulated = 2 * (t_eval[idx2] - t_eval[idx1])
    print(f"Simulation gives T       = {T_simulated:.1f} s  ({T_simulated/60:.2f} min)")
    error_pct = abs(T_simulated - T_kepler) / T_kepler * 100
    print(f"Error vs Kepler's law    = {error_pct:.4f}%")
else:
    print("Could not detect period from simulation — check solver settings")

# ── Compute orbital altitude at each step ─────────────────────────────────────
r_arr = np.sqrt(x**2 + y**2 + z**2)
altitude_arr = r_arr - R_EARTH
print(f"\nMin altitude in sim = {altitude_arr.min():.2f} km")
print(f"Max altitude in sim = {altitude_arr.max():.2f} km")
print(f"Altitude variation  = {altitude_arr.max() - altitude_arr.min():.4f} km  (should be ~0 for circular orbit)")

# ── Plot 1: 3D orbit ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(111, projection='3d')

# Draw Earth as a sphere
u_e = np.linspace(0, 2*np.pi, 50)
v_e = np.linspace(0, np.pi, 50)
xe = R_EARTH * np.outer(np.cos(u_e), np.sin(v_e))
ye = R_EARTH * np.outer(np.sin(u_e), np.sin(v_e))
ze = R_EARTH * np.outer(np.ones(np.size(u_e)), np.cos(v_e))
ax3d.plot_surface(xe, ye, ze, color='cornflowerblue', alpha=0.4)

# Draw orbit
ax3d.plot(x, y, z, 'r-', linewidth=1.5, label='CubeSat orbit')
ax3d.scatter([state0[0]], [state0[1]], [state0[2]], color='green', s=50, label='Start', zorder=5)

ax3d.set_xlabel('X [km]')
ax3d.set_ylabel('Y [km]')
ax3d.set_zlabel('Z [km]')
ax3d.set_title(f'Keplerian Orbit — 500 km Circular — 3 Orbits')
ax3d.legend()
plt.tight_layout()
plt.savefig('orbit_3d.png', dpi=150)
plt.show()
print("\nFigure saved: orbit_3d.png")
