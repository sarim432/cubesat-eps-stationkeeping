# orbit_with_drag.py
# Week 2: Add atmospheric drag — simulate CubeSat orbital decay
# CubeSat EPS + Station-Keeping Research — Sarim, June 2026

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ── Constants ──────────────────────────────────────────────────────────────────
MU       = 398600.4418   # Earth gravitational parameter [km³/s²]
R_EARTH  = 6371.0        # Earth mean radius [km]

# ── 3U CubeSat parameters ─────────────────────────────────────────────────────
MASS = 4.0               # [kg]
AREA = 0.01              # frontal area [m²]  (10 cm × 10 cm face)
CD   = 2.2               # drag coefficient — standard CubeSat assumption

print(f"Ballistic coefficient = {MASS/(CD*AREA):.1f} kg/m²")
# Expected: ~181.8 kg/m²

# ── Atmosphere model ───────────────────────────────────────────────────────────
# US Standard Atmosphere 1976 tabulated densities
# In the full paper we replace this with NRLMSISE-00
# For now this gives the right order-of-magnitude at each altitude
ALT_KM = np.array([150, 200, 250, 300, 350,
                    400, 450, 500, 550, 600, 700, 800])
RHO_KGM3 = np.array([2.07e-9,  2.54e-10, 6.24e-11, 1.92e-11,
                      5.97e-12, 2.08e-12, 7.26e-13, 2.59e-13,
                      9.49e-14, 3.56e-14, 5.32e-15, 1.57e-15])

# Log-linear interpolation (density is exponential in altitude)
_log_rho = interp1d(ALT_KM, np.log(RHO_KGM3),
                    kind='linear', fill_value='extrapolate')

def air_density(alt_km):
    """Atmospheric density [kg/m³] at altitude alt_km [km]."""
    alt_km = np.clip(alt_km, 150.0, 900.0)
    return float(np.exp(_log_rho(alt_km)))

# ── Equations of motion: gravity + drag ──────────────────────────────────────
def eom_drag(t, state):
    x, y, z, vx, vy, vz = state

    r     = np.sqrt(x**2 + y**2 + z**2)         # distance from Earth [km]
    alt   = r - R_EARTH                           # altitude [km]

    # --- Gravity ---
    g_fac = -MU / r**3
    ax    = g_fac * x
    ay    = g_fac * y
    az    = g_fac * z

    # --- Drag ---
    v_kms  = np.sqrt(vx**2 + vy**2 + vz**2)      # speed [km/s]
    v_ms   = v_kms * 1000.0                        # convert to [m/s]
    rho    = air_density(alt)                      # [kg/m³]

    # Drag deceleration [m/s²] = 0.5 * rho * v² * Cd * A / m
    a_drag_ms2  = 0.5 * rho * v_ms**2 * CD * AREA / MASS
    a_drag_kms2 = a_drag_ms2 / 1000.0             # convert back to [km/s²]

    # Direction: opposite to velocity
    ax += -a_drag_kms2 * (vx / v_kms)
    ay += -a_drag_kms2 * (vy / v_kms)
    az += -a_drag_kms2 * (vz / v_kms)

    return [vx, vy, vz, ax, ay, az]

# ── Re-entry detection: stop simulation when altitude < 150 km ───────────────
def reentry(t, state):
    r = np.sqrt(state[0]**2 + state[1]**2 + state[2]**2)
    return r - R_EARTH - 150.0    # returns 0 when alt = 150 km

reentry.terminal  = True          # stop integration
reentry.direction = -1            # only trigger when descending

# ── Simulate for a given starting altitude ────────────────────────────────────
def run(start_alt_km, sim_days=365):
    a0     = R_EARTH + start_alt_km
    v0     = np.sqrt(MU / a0)                     # circular orbit velocity
    state0 = [a0, 0.0, 0.0, 0.0, v0, 0.0]

    T_orbit = 2 * np.pi * np.sqrt(a0**3 / MU)    # [s]
    T_total = sim_days * 86400                     # [s]

    # Sample once per orbit to keep memory low
    n_pts  = int(T_total / T_orbit) + 1
    t_eval = np.linspace(0, T_total, n_pts)

    sol = solve_ivp(
        fun      = eom_drag,
        t_span   = (0, T_total),
        y0       = state0,
        method   = 'RK45',
        t_eval   = t_eval,
        events   = reentry,
        rtol     = 1e-9,
        atol     = 1e-11,
    )

    t_days = sol.t / 86400
    r_arr  = np.sqrt(sol.y[0]**2 + sol.y[1]**2 + sol.y[2]**2)
    alt    = r_arr - R_EARTH

    # Did it re-enter?
    if sol.t_events[0].size > 0:
        reentry_day = sol.t_events[0][0] / 86400
        print(f"  {start_alt_km} km → RE-ENTERED at day {reentry_day:.1f}")
    else:
        print(f"  {start_alt_km} km → survived 1 year  |  "
              f"altitude after 365 days: {alt[-1]:.1f} km  |  "
              f"total decay: {start_alt_km - alt[-1]:.1f} km")

    return t_days, alt

# ── Run simulations ───────────────────────────────────────────────────────────
print("\nRunning simulations (expect ~60–90 seconds)...")

altitudes = [400, 500, 600]
results   = {}
for h in altitudes:
    t, a = run(h)
    results[h] = (t, a)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
colors  = ['tab:red', 'tab:blue', 'tab:green']

for h, color in zip(altitudes, colors):
    t_days, alt = results[h]
    ax.plot(t_days, alt, color=color, linewidth=2.0,
            label=f'Initial altitude: {h} km')

ax.set_xlabel('Time [days]',  fontsize=13)
ax.set_ylabel('Altitude [km]', fontsize=13)
ax.set_title('Orbital Decay Due to Atmospheric Drag\n'
             '3U CubeSat  |  Mass 4 kg  |  Area 0.01 m²  |  Cd = 2.2',
             fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 365])
plt.tight_layout()
plt.savefig('orbital_decay.png', dpi=150)
plt.show()
print("\nFigure saved: orbital_decay.png")
print("Push both files to GitHub when done.")
