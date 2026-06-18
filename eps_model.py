# eps_model.py
# Week 3: CubeSat Electric Power System — solar array + battery model
# CubeSat EPS + Station-Keeping Research — Sarim, June 2026

import numpy as np
import matplotlib.pyplot as plt

# ── Physical constants ────────────────────────────────────────────────────────
G_SOLAR  = 1361.0       # solar flux at Earth [W/m²]
R_EARTH  = 6371.0       # Earth radius [km]
MU       = 398600.4418  # [km³/s²]

# ── CubeSat configuration (6U) ────────────────────────────────────────────────
# We'll sweep panel area later; start with 0.06 m²
PANEL_AREA       = 0.06     # solar panel area [m²]
PANEL_EFF        = 0.30     # triple-junction GaAs efficiency
INCIDENCE_FACTOR = 0.90     # deployable 1-axis tracking panels (not perfect tracking)

BATT_CAPACITY_WH = 40.0     # battery capacity [Wh]
BATT_DOD_MAX     = 0.30     # max depth-of-discharge (preserve battery life)
BATT_EFF_CHARGE  = 0.95     # charge efficiency
BATT_EFF_DISCH   = 0.95     # discharge efficiency
SOC_MIN          = 1.0 - BATT_DOD_MAX   # = 0.70 (never go below 70% SoC)
SOC_INIT         = 0.95     # start at 95% charged

# ── Spacecraft power budget [W] ───────────────────────────────────────────────
P_OBC      = 3.0    # on-board computer + housekeeping
P_COMMS    = 5.0    # radio (average over transmit + receive + idle cycles)
P_ADCS     = 2.0    # attitude determination and control system
P_PAYLOAD  = 5.0    # science payload (Earth observation camera etc.)
P_THRUSTER = 7.5    # electrospray module: 5W thruster + 2.5W power processing unit

P_BASELINE = P_OBC + P_COMMS + P_ADCS + P_PAYLOAD
P_TOTAL    = P_BASELINE + P_THRUSTER

print("=" * 50)
print("CubeSat 6U Power Budget Summary")
print("=" * 50)
print(f"  Solar panel area:      {PANEL_AREA:.3f} m²")
print(f"  Panel efficiency:      {PANEL_EFF*100:.0f}%")
print(f"  Battery capacity:      {BATT_CAPACITY_WH:.0f} Wh")
print(f"  Baseline load:         {P_BASELINE:.1f} W")
print(f"  Thruster demand:       {P_THRUSTER:.1f} W")
print(f"  Total when thrusting:  {P_TOTAL:.1f} W")
print("=" * 50)

# ── Orbital parameters ────────────────────────────────────────────────────────
ALTITUDE = 500.0
a        = R_EARTH + ALTITUDE                        # semi-major axis [km]
T_ORBIT  = 2 * np.pi * np.sqrt(a**3 / MU)           # orbital period [s]

# ── Eclipse fraction (analytical formula) ─────────────────────────────────────
f_eclipse = (1 / np.pi) * np.arcsin(R_EARTH / a)    # fraction of orbit in shadow
f_sunlit  = 1.0 - f_eclipse
T_eclipse = f_eclipse * T_ORBIT                      # eclipse duration [s]
T_sunlit  = f_sunlit  * T_ORBIT                      # sunlit duration [s]

print(f"\nOrbital parameters at {ALTITUDE:.0f} km:")
print(f"  Orbital period:        {T_ORBIT/60:.1f} min")
print(f"  Eclipse fraction:      {f_eclipse*100:.1f}%")
print(f"  Eclipse duration:      {T_eclipse/60:.1f} min per orbit")
print(f"  Sunlit duration:       {T_sunlit/60:.1f} min per orbit")

# ── Solar power generation ────────────────────────────────────────────────────
P_solar_sunlit = PANEL_AREA * G_SOLAR * PANEL_EFF * INCIDENCE_FACTOR
P_solar_avg    = P_solar_sunlit * f_sunlit   # orbit-averaged (zero during eclipse)

print(f"\nSolar power:")
print(f"  Sunlit peak power:     {P_solar_sunlit:.1f} W")
print(f"  Orbit-averaged power:  {P_solar_avg:.1f} W")
print(f"  Margin vs baseline:    {P_solar_avg - P_BASELINE:.1f} W (orbit-averaged)")

# ── Simulate battery SoC over one orbit (60s timesteps) ──────────────────────
dt        = 60.0                               # timestep [s]
t_vec     = np.arange(0, T_ORBIT, dt)         # time array [s]
n_steps   = len(t_vec)

SoC       = np.zeros(n_steps)                 # state of charge [fraction 0–1]
P_gen     = np.zeros(n_steps)                 # power generated [W]
P_load    = np.zeros(n_steps)                 # power consumed [W]
P_net     = np.zeros(n_steps)                 # net power [W]
thrusting = np.zeros(n_steps, dtype=bool)     # is thruster firing?

SoC[0] = SOC_INIT

for i in range(n_steps):
    t = t_vec[i]

    # Eclipse detection:
    # Satellite starts in sunlit region, enters eclipse halfway through orbit
    # Simple model: eclipse occurs in middle portion of orbit
    eclipse_start = T_ORBIT * (0.5 - f_eclipse/2)
    eclipse_end   = T_ORBIT * (0.5 + f_eclipse/2)
    in_eclipse    = (t >= eclipse_start) and (t <= eclipse_end)

    # Power generation
    P_gen[i] = 0.0 if in_eclipse else P_solar_sunlit

    # Thruster firing logic:
    # Fire thruster ONLY if (a) sunlit, (b) SoC > 80%, (c) power budget allows
    can_thrust = (not in_eclipse) and \
                 (SoC[i] > 0.80) and \
                 (P_gen[i] >= P_TOTAL)

    thrusting[i] = can_thrust
    P_load[i]    = P_TOTAL if can_thrust else P_BASELINE
    P_net[i]     = P_gen[i] - P_load[i]

    # Update battery SoC
    if i < n_steps - 1:
        energy_wh = P_net[i] * dt / 3600.0   # convert W·s → Wh

        if energy_wh >= 0:
            # Charging
            delta_SoC = energy_wh * BATT_EFF_CHARGE / BATT_CAPACITY_WH
            SoC[i+1]  = min(1.0, SoC[i] + delta_SoC)
        else:
            # Discharging
            delta_SoC = abs(energy_wh) / (BATT_EFF_DISCH * BATT_CAPACITY_WH)
            SoC[i+1]  = max(SOC_MIN, SoC[i] - delta_SoC)

# ── Results summary ───────────────────────────────────────────────────────────
thrust_fraction = thrusting.sum() / n_steps
thrust_minutes  = thrusting.sum() * dt / 60.0
SoC_min         = SoC.min()
SoC_max         = SoC.max()

print(f"\nOne-orbit simulation results:")
print(f"  Thruster firing fraction:  {thrust_fraction*100:.1f}% of orbit")
print(f"  Thruster on-time:          {thrust_minutes:.1f} min per orbit")
print(f"  Battery SoC range:         {SoC_min*100:.1f}% – {SoC_max*100:.1f}%")
print(f"  SoC at orbit end:          {SoC[-1]*100:.1f}%")
print(f"  Energy balance:            {'SUSTAINED ✓' if SoC[-1] >= SOC_INIT - 0.01 else 'DRAINING ✗'}")

# ── Compare three panel sizes ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Panel area trade — sunlit power vs baseline load")
print("=" * 55)
for area in [0.03, 0.06, 0.09, 0.12]:
    P_sun  = area * G_SOLAR * PANEL_EFF * INCIDENCE_FACTOR
    P_avg  = P_sun * f_sunlit
    margin = P_sun - P_TOTAL
    status = "CAN thrust" if margin > 0 else "CANNOT thrust (power-limited)"
    print(f"  {area:.2f} m²: sunlit={P_sun:.1f}W  avg={P_avg:.1f}W  "
          f"margin vs full load={margin:+.1f}W  → {status}")

# ── Plot ──────────────────────────────────────────────────────────────────────
t_min = t_vec / 60.0   # convert to minutes

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel 1: Power generated vs consumed
axes[0].fill_between(t_min, P_gen, alpha=0.4, color='gold', label='Solar power generated')
axes[0].plot(t_min, P_load, 'r-', linewidth=2, label='Power consumed')
axes[0].axhline(P_BASELINE, color='gray', linestyle='--', alpha=0.7,
                label=f'Baseline load ({P_BASELINE:.0f} W)')
axes[0].axhline(P_TOTAL, color='red', linestyle='--', alpha=0.7,
                label=f'Total with thruster ({P_TOTAL:.0f} W)')
axes[0].set_ylabel('Power [W]', fontsize=11)
axes[0].set_title('CubeSat 6U — One-Orbit Power Balance\n'
                  f'500 km SSO  |  {PANEL_AREA:.2f} m² panels  |  '
                  f'{BATT_CAPACITY_WH:.0f} Wh battery', fontsize=12)
axes[0].legend(fontsize=9, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Panel 2: Battery SoC
axes[1].plot(t_min, SoC * 100, 'b-', linewidth=2)
axes[1].axhline(SOC_MIN * 100, color='red', linestyle='--',
                label=f'Min SoC ({SOC_MIN*100:.0f}% — DoD limit)')
axes[1].fill_between(t_min, SoC * 100, SOC_MIN * 100,
                      where=(SoC * 100 > SOC_MIN * 100),
                      alpha=0.2, color='blue', label='Usable charge')
axes[1].set_ylabel('Battery SoC [%]', fontsize=11)
axes[1].set_ylim([60, 105])
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Panel 3: Thruster firing
axes[2].fill_between(t_min, thrusting.astype(float) * 7.5,
                      alpha=0.7, color='green', label='Thruster firing (7.5 W)')
axes[2].set_ylabel('Thruster [W]', fontsize=11)
axes[2].set_xlabel('Time in orbit [minutes]', fontsize=11)
axes[2].set_ylim([0, 12])
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

# Shade eclipse region on all panels
for ax in axes:
    ax.axvspan(eclipse_start/60, eclipse_end/60,
               alpha=0.12, color='navy', label='Eclipse')

plt.tight_layout()
plt.savefig('eps_one_orbit.png', dpi=150)
plt.show()
print("\nFigure saved: eps_one_orbit.png")
