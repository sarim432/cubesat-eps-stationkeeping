# propulsion_budget.py
# Week 4: Electric propulsion Δv budget and propellant sizing
# CubeSat EPS + Station-Keeping Research — Sarim, June 2026

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
G0       = 9.81          # standard gravity [m/s²]
MU       = 398600.4418   # [km³/s²]
R_EARTH  = 6371.0        # [km]
SEC_YEAR = 365.25 * 86400  # seconds per year

# ── CubeSat parameters ────────────────────────────────────────────────────────
MASS_DRY    = 4.0        # dry mass [kg]
PROP_BUDGET = 0.10       # propellant budget [kg] — 100g allocation for 6U
MASS_WET    = MASS_DRY + PROP_BUDGET

# ── Accion TILE thruster parameters ──────────────────────────────────────────
THRUST   = 1.0e-4        # [N] = 100 µN
ISP      = 1500.0        # specific impulse [s]
P_THRUST = 7.5           # total power (thruster + PPU) [W]

# ── Atmosphere model (same density table as orbit_with_drag.py) ───────────────
ALT_KM   = np.array([200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
RHO_KGM3 = np.array([2.54e-10, 6.24e-11, 1.92e-11, 5.97e-12,
                      2.08e-12, 7.26e-13, 2.59e-13, 9.49e-14,
                      3.56e-14, 1.36e-14, 5.32e-15])
from scipy.interpolate import interp1d
_log_rho = interp1d(ALT_KM, np.log(RHO_KGM3), kind='linear',
                    fill_value='extrapolate')

def air_density(alt_km):
    return float(np.exp(_log_rho(np.clip(alt_km, 200.0, 700.0))))

# ── Drag parameters (3U CubeSat) ─────────────────────────────────────────────
CD   = 2.2
AREA = 0.01              # frontal area [m²]

# ── Core functions ────────────────────────────────────────────────────────────
def orbital_velocity(alt_km):
    """Circular orbital velocity [m/s]."""
    a = (R_EARTH + alt_km) * 1000.0           # convert to [m]
    return np.sqrt(MU * 1e9 / a)              # MU in m³/s² = 3.986e14

def drag_decel(alt_km):
    """Drag deceleration magnitude [m/s²] at given altitude."""
    rho   = air_density(alt_km)
    v_ms  = orbital_velocity(alt_km)
    return 0.5 * rho * v_ms**2 * CD * AREA / MASS_WET

def delta_v_per_year(alt_km):
    """Annual Δv needed to compensate drag [m/s/year]."""
    return drag_decel(alt_km) * SEC_YEAR

def firing_time_per_year(dv_year):
    """Total thruster on-time needed per year [hours]."""
    a_thrust = THRUST / MASS_WET              # thrust deceleration [m/s²]
    t_sec    = dv_year / a_thrust             # firing time [seconds]
    return t_sec / 3600.0                     # convert to hours

def propellant_per_year(dv_year):
    """Propellant mass consumed per year [grams]."""
    mdot = THRUST / (ISP * G0)               # mass flow rate [kg/s]
    t_sec = (dv_year / (THRUST / MASS_WET))  # firing time [s]
    return mdot * t_sec * 1000.0             # convert to grams

def station_keeping_lifetime(alt_km, prop_kg):
    """How many years can the thruster maintain the orbit? [years]"""
    dv_yr   = delta_v_per_year(alt_km)
    prop_yr = propellant_per_year(dv_yr) / 1000.0   # kg/year
    return prop_kg / prop_yr

def firing_fraction(alt_km):
    """Fraction of each orbit the thruster must fire."""
    t_fire_hr_yr = firing_time_per_year(delta_v_per_year(alt_km))
    return (t_fire_hr_yr * 3600) / SEC_YEAR

# ── Print results across altitudes ────────────────────────────────────────────
print("=" * 75)
print(f"{'Altitude':>10} {'Drag Δv':>12} {'Fire time':>12} "
      f"{'Prop/yr':>12} {'Lifetime':>12} {'Fire%':>8}")
print(f"{'[km]':>10} {'[m/s/yr]':>12} {'[hrs/yr]':>12} "
      f"{'[g/yr]':>12} {'[years]':>12} {'':>8}")
print("-" * 75)

altitudes = [400, 450, 500, 550, 600]
for h in altitudes:
    dv   = delta_v_per_year(h)
    t_hr = firing_time_per_year(dv)
    prop = propellant_per_year(dv)
    life = station_keeping_lifetime(h, PROP_BUDGET)
    frac = firing_fraction(h) * 100

    print(f"{h:>10}  {dv:>11.2f}  {t_hr:>11.1f}  "
          f"{prop:>11.2f}  {life:>11.1f}  {frac:>7.2f}%")

print("=" * 75)
print(f"\nThruster: Accion TILE  |  F={THRUST*1e6:.0f} µN  |"
      f"  Isp={ISP:.0f} s  |  Power={P_THRUST:.1f} W")
print(f"Propellant budget: {PROP_BUDGET*1000:.0f} g")

# ── Power check: can the 0.09 m² panel actually sustain firing? ───────────────
print("\n" + "=" * 55)
print("Power feasibility check at 500 km (0.09 m² panels)")
print("=" * 55)
PANEL_AREA  = 0.09
P_solar     = PANEL_AREA * 1361 * 0.30 * 0.90   # [W]
P_baseline  = 15.0                                # [W]
P_margin    = P_solar - P_baseline                # [W] available for thruster
print(f"  Solar power (sunlit):    {P_solar:.1f} W")
print(f"  Baseline load:           {P_baseline:.1f} W")
print(f"  Available for thruster:  {P_margin:.1f} W")
print(f"  Thruster needs:          {P_THRUST:.1f} W")
print(f"  Margin after thrusting:  {P_margin - P_THRUST:.1f} W")
print(f"  Can sustain firing:      {'YES ✓' if P_margin >= P_THRUST else 'NO ✗'}")

# ── Edelbaum orbit raising (bonus calculation) ────────────────────────────────
print("\n" + "=" * 55)
print("Edelbaum: Δv for orbit raising (bonus)")
print("=" * 55)
for h1, h2 in [(400, 500), (500, 600), (400, 600)]:
    v1 = orbital_velocity(h1)
    v2 = orbital_velocity(h2)
    # No inclination change → Edelbaum = |v2 - v1|
    dv_edelbaum = abs(v2 - v1)
    # Hohmann for comparison
    a1 = (R_EARTH + h1) * 1000
    a2 = (R_EARTH + h2) * 1000
    dv_hohmann = (np.sqrt(2*MU*1e9*a2/(a1*(a1+a2))) - np.sqrt(MU*1e9/a1)) + \
                 (np.sqrt(MU*1e9/a2) - np.sqrt(2*MU*1e9*a1/(a2*(a1+a2))))
    print(f"  {h1}→{h2} km:  Edelbaum={dv_edelbaum:.1f} m/s  "
          f"Hohmann={dv_hohmann:.1f} m/s")
