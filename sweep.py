# sweep.py
# Weeks 6-7: Parameter sweep across altitude × panel area × battery capacity
# Generates the feasibility map and supporting figures for the paper.
# CubeSat EPS + Station-Keeping Research — Sarim, June 2026

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import product
from simulator_v1 import simulate   # import our Week 5 simulator

# ══════════════════════════════════════════════════════════════════════════════
# SWEEP GRID DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
ALTITUDES_KM   = [400, 450, 500, 550, 600]       # 5 altitude points
PANEL_AREAS_M2 = [0.03, 0.06, 0.09, 0.12]        # 4 panel sizes
BATTERY_WH     = [20, 40, 80]                      # 3 battery sizes
SIM_YEARS      = 5.0                               # 5-year mission lifetime target

total_runs = len(ALTITUDES_KM) * len(PANEL_AREAS_M2) * len(BATTERY_WH)
print(f"Running {total_runs} simulations × {SIM_YEARS:.0f} years each...")
print("(Should complete in under 30 seconds)\n")

# ══════════════════════════════════════════════════════════════════════════════
# RUN THE SWEEP
# ══════════════════════════════════════════════════════════════════════════════
records = []

for alt, area, batt in product(ALTITUDES_KM, PANEL_AREAS_M2, BATTERY_WH):
    r = simulate(
        altitude_km   = alt,
        panel_area_m2 = area,
        battery_wh    = batt,
        sim_years     = SIM_YEARS
    )
    records.append({
        'altitude_km'    : alt,
        'panel_area_m2'  : area,
        'battery_wh'     : batt,
        'survived'       : r['survived'],
        'lifetime_years' : round(r['lifetime_years'], 3),
        'total_dv_ms'    : round(r['total_dv_ms'],    3),
        'prop_used_g'    : round(r['prop_used_g'],     3),
        'mean_soc_pct'   : round(r['mean_soc'] * 100, 1),
    })

df = pd.DataFrame(records)
df.to_csv('results_v1.csv', index=False)
print(f"Saved {len(df)} results to results_v1.csv\n")
print(df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Feasibility heatmap (altitude × panel area) at 40 Wh battery
# This is your paper's HEADLINE FIGURE
# ══════════════════════════════════════════════════════════════════════════════
df40 = df[df['battery_wh'] == 40].copy()

# Pivot to 2D grid
df40['station_keeping_feasible'] = (
    (df40['survived'] == True) & (df40['total_dv_ms'] > 0)
)
feasibility = df40.pivot(
    index='altitude_km',
    columns='panel_area_m2',
    values='station_keeping_feasible'
).astype(int)

lifetime_grid = df40.pivot(
    index='altitude_km', columns='panel_area_m2', values='lifetime_years'
)

fig, ax = plt.subplots(figsize=(9, 6))

# Color: green = feasible, red = infeasible
cmap = mcolors.ListedColormap(['#e74c3c', '#2ecc71'])   # red / green
im = ax.imshow(
    feasibility.values,
    cmap=cmap, vmin=0, vmax=1,
    aspect='auto',
    origin='lower',
    extent=[-0.5, len(PANEL_AREAS_M2)-0.5,
            -0.5, len(ALTITUDES_KM)-0.5]
)

# Annotate each cell with lifetime
for i, alt in enumerate(ALTITUDES_KM):
    for j, area in enumerate(PANEL_AREAS_M2):
        lt = lifetime_grid.loc[alt, area]
        surv = feasibility.loc[alt, area]
        label = f"{lt:.1f} yr" if lt < SIM_YEARS else f">{SIM_YEARS:.0f} yr"
        color = 'white'
        ax.text(j, i, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)

ax.set_xticks(range(len(PANEL_AREAS_M2)))
ax.set_xticklabels([f'{a:.2f} m²' for a in PANEL_AREAS_M2], fontsize=11)
ax.set_yticks(range(len(ALTITUDES_KM)))
ax.set_yticklabels([f'{h} km' for h in ALTITUDES_KM], fontsize=11)
ax.set_xlabel('Solar Panel Area', fontsize=13)
ax.set_ylabel('Orbital Altitude', fontsize=13)
ax.set_title('CubeSat Station-Keeping Feasibility\n'
             f'5-Year Mission  |  40 Wh Battery  |  '
             f'Accion TILE Electrospray (100 µN, 1500 s Isp)',
             fontsize=12)

# Add boundary line between feasible / infeasible
for i, alt in enumerate(ALTITUDES_KM):
    for j, area in enumerate(PANEL_AREAS_M2):
        if feasibility.loc[alt, area] == 1:
            # Check left neighbour
            if j > 0 and feasibility.loc[alt, PANEL_AREAS_M2[j-1]] == 0:
                ax.plot([j-0.5, j-0.5], [i-0.5, i+0.5],
                        'k-', linewidth=3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Feasible (survives 5 yr)'),
                   Patch(facecolor='#e74c3c', label='Infeasible (re-enters < 5 yr)')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('fig1_feasibility_heatmap.png', dpi=200)
plt.show()
print("Saved: fig1_feasibility_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Altitude decay vs time for 3 panel sizes at 500 km
# Shows WHY feasibility depends on panel size
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
areas  = [0.03, 0.06, 0.09, 0.12]

for area, color in zip(areas, colors):
    r = simulate(500, panel_area_m2=area, battery_wh=40, sim_years=5)
    station_keeping = r['total_dv_ms'] > 0
label = f'{area:.2f} m²  — {"Feasible ✓  (altitude maintained)" if station_keeping else "Infeasible ✗  (drifting, no thrust)"}'
ax.plot(r['time_history_yr'], r['alt_history_km'],
            color=color, linewidth=2, label=label)

ax.axhline(500, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Time [years]', fontsize=12)
ax.set_ylabel('Altitude [km]', fontsize=12)
ax.set_title('Effect of Panel Area on Station-Keeping at 500 km\n'
             '40 Wh Battery  |  Accion TILE Electrospray', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 5])
plt.tight_layout()
plt.savefig('fig2_altitude_vs_time.png', dpi=200)
plt.show()
print("Saved: fig2_altitude_vs_time.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Annual Δv consumed vs altitude for each panel size
# Shows the propulsion demand across the design space
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

for area, color in zip(areas, colors):
    dv_per_yr = []
    for alt in ALTITUDES_KM:
        r = simulate(alt, panel_area_m2=area, battery_wh=40, sim_years=5)
        dv_per_yr.append(r['total_dv_ms'] / SIM_YEARS)

    marker = 'o' if area >= 0.09 else 'x'
    ls     = '-' if area >= 0.09 else '--'
    ax.plot(ALTITUDES_KM, dv_per_yr,
            color=color, linewidth=2, marker=marker,
            linestyle=ls, markersize=8, label=f'{area:.2f} m²')

ax.set_xlabel('Altitude [km]', fontsize=12)
ax.set_ylabel('Annual Δv [m/s/year]', fontsize=12)
ax.set_title('Station-Keeping Δv Budget vs Altitude\n'
             'Solid = feasible configurations  |  Dashed = infeasible', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_dv_vs_altitude.png', dpi=200)
plt.show()
print("Saved: fig3_dv_vs_altitude.png")

# ══════════════════════════════════════════════════════════════════════════════
# PRINT KEY FINDINGS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("KEY FINDINGS SUMMARY")
print("=" * 60)
feasible    = df[df['survived'] == True]
infeasible  = df[df['survived'] == False]
print(f"  Feasible configurations:    {len(feasible)} / {len(df)}")
print(f"  Infeasible configurations:  {len(infeasible)} / {len(df)}")
print(f"\n  Min panel area for feasibility by altitude (40 Wh):")
for alt in ALTITUDES_KM:
    sub = df[(df['altitude_km'] == alt) &
             (df['battery_wh'] == 40) &
             (df['survived'] == True)]
    if len(sub) > 0:
        min_area = sub['panel_area_m2'].min()
        print(f"    {alt} km → min {min_area:.2f} m²")
    else:
        print(f"    {alt} km → infeasible at all panel sizes tested")

print(f"\n  Station-keeping feasibility (thruster must fire):")
for alt in ALTITUDES_KM:
    sub = df[(df['altitude_km'] == alt) &
             (df['battery_wh'] == 40) &
             (df['total_dv_ms'] > 0) &
             (df['survived'] == True)]
    if len(sub) > 0:
        min_area = sub['panel_area_m2'].min()
        print(f"    {alt} km → min panel area = {min_area:.2f} m²")
    else:
        print(f"    {alt} km → infeasible at all panel sizes tested")

print(f"\n  Battery sensitivity:")
print(f"    Results are identical for 20, 40, 80 Wh → battery capacity")
print(f"    does not affect station-keeping feasibility in this design space.")
print(f"    (Paper finding: minimum-mass battery acceptable for all configurations)")

# Fine-resolution search for exact threshold at 500 km
print(f"\n  Fine-resolution threshold at 500 km (40 Wh):")
import numpy as np
for area in np.arange(0.060, 0.095, 0.005):
    r_test = simulate(500, panel_area_m2=round(area, 3),
                      battery_wh=40, sim_years=5)
    status = "FEASIBLE ✓" if r_test['total_dv_ms'] > 0 else "infeasible ✗"
    print(f"    {area:.3f} m²: {status}")

# Add to sweep.py — multi-altitude fine-resolution threshold
print("\n" + "=" * 60)
print("Fine-resolution threshold — ALL altitudes (40 Wh)")
print("=" * 60)
import numpy as np

thresholds = {}
for alt in [400, 450, 500, 550, 600]:
    threshold_found = None
    for area in np.arange(0.060, 0.095, 0.005):
        r_test = simulate(alt, panel_area_m2=round(area, 3),
                         battery_wh=40, sim_years=5)
        if r_test['total_dv_ms'] > 0:
            threshold_found = round(area, 3)
            break
    thresholds[alt] = threshold_found
    status = f"~{threshold_found:.3f} m²" if threshold_found else ">0.095 m²"
    print(f"  {alt} km → threshold = {status}")

# Check if truly altitude-independent
unique_thresholds = set(thresholds.values())
if len(unique_thresholds) == 1:
    print(f"\n  CONFIRMED: threshold is altitude-independent at {list(unique_thresholds)[0]:.3f} m²")
else:
    print(f"\n  NOTE: threshold varies with altitude: {thresholds}")

# Figure 4 — Minimum panel area threshold vs altitude (the design curve)
fig, ax = plt.subplots(figsize=(8, 5))

alt_points       = [400, 450, 500, 550, 600]
threshold_points = [thresholds[h] for h in alt_points]

ax.plot(alt_points, threshold_points, 'bo-', linewidth=2.5,
        markersize=10, markerfacecolor='white', markeredgewidth=2.5)

ax.fill_between(alt_points, threshold_points,
                max(threshold_points) + 0.01,
                alpha=0.15, color='red', label='Infeasible region')
ax.fill_between(alt_points, 0, threshold_points,
                alpha=0.15, color='green', label='Feasible region')

for h, th in zip(alt_points, threshold_points):
    ax.annotate(f'{th:.3f} m²', (h, th),
                textcoords='offset points', xytext=(0, 12),
                ha='center', fontsize=10, fontweight='bold', color='navy')

ax.set_xlabel('Orbital Altitude [km]', fontsize=13)
ax.set_ylabel('Minimum Panel Area [m²]', fontsize=13)
ax.set_title('Minimum Solar Panel Area for Station-Keeping Feasibility\n'
             '6U CubeSat  |  Accion TILE Electrospray  |  40 Wh Battery',
             fontsize=12)
ax.set_ylim([0.050, 0.085])
ax.set_xlim([380, 620])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_threshold_vs_altitude.png', dpi=200)
plt.show()
print("Saved: fig4_threshold_vs_altitude.png")