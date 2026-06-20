# simulator_v1.py
# Week 5: Coupled CubeSat EPS + electric propulsion simulator (orbit-averaged)
# One timestep = one orbit. Suitable for multi-year parametric sweeps.
# CubeSat EPS + Station-Keeping Research — Sarim, June 2026

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS AND DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
MU       = 3.986004418e14   # Earth gravitational parameter [m³/s²]
R_EARTH  = 6_371_000.0      # Earth radius [m]
G0       = 9.81             # standard gravity [m/s²]
G_SOLAR  = 1361.0           # solar constant [W/m²]

# Atmosphere model: US Standard Atmosphere 1976 (conservative/quiet sun)
# Will be replaced with NRLMSISE-00 for final paper version
_ALT_M   = np.array([150, 200, 250, 300, 350,
                      400, 450, 500, 550, 600, 700, 800]) * 1e3  # [m]
_RHO     = np.array([2.07e-9, 2.54e-10, 6.24e-11, 1.92e-11,
                      5.97e-12, 2.08e-12, 7.26e-13, 2.59e-13,
                      9.49e-14, 3.56e-14, 5.32e-15, 1.57e-15])  # [kg/m³]
_log_rho = interp1d(_ALT_M, np.log(_RHO), kind='linear',
                    fill_value='extrapolate')

def _rho(alt_m):
    return float(np.exp(_log_rho(np.clip(alt_m, 150e3, 800e3))))

# ══════════════════════════════════════════════════════════════════════════════
# ORBITAL MECHANICS (all inputs in meters)
# ══════════════════════════════════════════════════════════════════════════════
def orbital_period(alt_m):
    """Orbital period [s] for circular orbit at altitude alt_m [m]."""
    a = R_EARTH + alt_m
    return 2 * np.pi * np.sqrt(a**3 / MU)

def orbital_velocity(alt_m):
    """Circular orbital speed [m/s]."""
    return np.sqrt(MU / (R_EARTH + alt_m))

def eclipse_fraction(alt_m):
    """Fraction of orbit spent in Earth's shadow."""
    return (1.0 / np.pi) * np.arcsin(R_EARTH / (R_EARTH + alt_m))

def drag_decel(alt_m, mass_kg, cd=2.2, area_m2=0.01):
    """Orbit-averaged drag deceleration [m/s²]."""
    rho   = _rho(alt_m)
    v     = orbital_velocity(alt_m)
    return 0.5 * rho * v**2 * cd * area_m2 / mass_kg

def altitude_loss_per_orbit(alt_m, a_drag_ms2):
    """
    Altitude lost per orbit due to drag [m].
    Derived from orbit energy equation: Δa/orbit = -4π × a_drag × a³ / µ
    """
    a = R_EARTH + alt_m
    return 4 * np.pi * a_drag_ms2 * a**3 / MU   # positive = loss

def altitude_gain_per_orbit(alt_m, thrust_n, mass_kg, t_thrust_s):
    """
    Altitude gained per orbit from low-thrust firing for t_thrust_s seconds.
    Uses impulse approximation: Δv = F/m × t, then Δa from vis-viva.
    """
    if t_thrust_s <= 0:
        return 0.0
    dv    = (thrust_n / mass_kg) * t_thrust_s   # Δv from thruster [m/s]
    v     = orbital_velocity(alt_m)
    a     = R_EARTH + alt_m
    da    = 2 * a * dv / v                       # altitude gain [m] from vis-viva
    return da

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATOR FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def simulate(
        altitude_km,        # starting altitude [km]
        panel_area_m2,      # solar panel area [m²]
        battery_wh,         # battery capacity [Wh]
        # ── fixed parameters (can override) ──
        mass_kg        = 4.1,    # wet mass [kg]
        prop_kg        = 0.10,   # propellant budget [kg]
        sim_years      = 5.0,    # simulation duration [years]
        thrust_n       = 1e-4,   # thruster force [N]
        isp_s          = 1500.0, # specific impulse [s]
        panel_eff      = 0.30,   # solar cell efficiency
        incidence_f    = 0.90,   # incidence angle factor
        p_baseline_w   = 15.0,   # baseline spacecraft load [W]
        p_thruster_w   = 7.5,    # thruster + PPU power [W]
        soc_min        = 0.70,   # minimum battery SoC (DoD limit)
        soc_thrust_thr = 0.80,   # SoC threshold to enable thrusting
        batt_eff       = 0.95,   # one-way battery efficiency
        cd             = 2.2,    # drag coefficient
        area_m2        = 0.01,   # frontal area [m²]
        verbose        = False
    ):
    """
    Orbit-averaged CubeSat EPS + station-keeping simulation.

    Returns dict with:
        survived       : bool — did the CubeSat survive sim_years?
        lifetime_years : float — time until re-entry or end of simulation
        total_dv_ms    : float — total Δv provided by thruster [m/s]
        prop_used_g    : float — propellant consumed [g]
        mean_soc       : float — mean battery SoC over mission
        alt_history_km : np.array — altitude vs time [km]
        time_history_yr: np.array — time array [years]
        dv_history_ms  : np.array — cumulative Δv [m/s]
    """

    # ── Initialize state ───────────────────────────────────────────────────────
    alt_m           = altitude_km * 1e3
    soc             = 0.95          # start 95% charged
    prop_kg_left    = prop_kg
    total_dv        = 0.0
    cumulative_dv   = 0.0
    alt_list        = [alt_m / 1e3]
    time_list       = [0.0]
    dv_list         = [0.0]
    soc_list        = [soc]

    max_orbits = int(sim_years * 365.25 * 24 * 3600 /
                     orbital_period(alt_m)) + 1
    t_elapsed  = 0.0   # [s]

    for _ in range(max_orbits):

        # ── 1. Orbital properties this orbit ─────────────────────────────────
        T_orb   = orbital_period(alt_m)            # [s]
        f_ecl   = eclipse_fraction(alt_m)
        T_sun   = (1 - f_ecl) * T_orb             # sunlit time [s]
        T_ecl   = f_ecl * T_orb                   # eclipse time [s]

        # ── 2. Energy accounting this orbit ──────────────────────────────────
        P_sol   = panel_area_m2 * G_SOLAR * panel_eff * incidence_f  # sunlit [W]
        E_gen   = P_sol * T_sun / 3600.0           # energy generated [Wh]
        E_base  = p_baseline_w * T_orb / 3600.0   # baseline consumption [Wh]
        E_surpl = E_gen - E_base                   # surplus for thruster [Wh]

        # ── 3. Thruster decision and on-time ─────────────────────────────────
        # Conditions: enough surplus energy + SoC above threshold + prop left
        # + solar power CAN support total load when sunlit
        can_thrust = (
            E_surpl > 0 and
            soc >= soc_thrust_thr and
            prop_kg_left > 1e-9 and
            P_sol >= (p_baseline_w + p_thruster_w)
        )

        t_thrust = 0.0
        if can_thrust:
            # ── How much thrust time is actually needed to cancel drag? ──────
            a_drag_val      = drag_decel(alt_m, mass_kg, cd, area_m2)
            dv_drag_orbit   = a_drag_val * T_orb          # Δv drag will remove [m/s]
            a_thrust_val    = thrust_n / mass_kg           # thrust accel [m/s²]
            t_thrust_needed = dv_drag_orbit / a_thrust_val # seconds to cancel drag

            # ── Apply all constraints (take the minimum) ──────────────────────
            t_thrust_energy = (E_surpl * 3600.0) / p_thruster_w  # energy limit [s]
            t_thrust_time   = T_sun * 0.80                         # time limit [s]
            t_thrust        = min(t_thrust_needed,   # ← NEW: only cancel drag
                                  t_thrust_energy,
                                  t_thrust_time)
            
            # Limit by remaining propellant
            mdot            = thrust_n / (isp_s * G0)              # [kg/s]
            t_thrust_prop   = prop_kg_left / mdot
            t_thrust        = min(t_thrust, t_thrust_prop)

        # ── 4. Update propellant and Δv ───────────────────────────────────────
        if t_thrust > 0:
            mdot        = thrust_n / (isp_s * G0)
            prop_burned = mdot * t_thrust
            prop_kg_left = max(0.0, prop_kg_left - prop_burned)
            dv_this_orbit = (thrust_n / mass_kg) * t_thrust
            total_dv    += dv_this_orbit
            cumulative_dv += dv_this_orbit

        # ── 5. Altitude change ────────────────────────────────────────────────
        a_drag    = drag_decel(alt_m, mass_kg, cd, area_m2)
        dh_loss   = altitude_loss_per_orbit(alt_m, a_drag)          # [m] lost
        dh_gain   = altitude_gain_per_orbit(alt_m, thrust_n,
                                            mass_kg, t_thrust)      # [m] gained
        alt_m     = max(150e3, alt_m - dh_loss + dh_gain)

        # ── 6. Battery SoC update ─────────────────────────────────────────────
        E_thrust_used = (p_thruster_w * t_thrust / 3600.0) if t_thrust > 0 else 0.0
        E_net         = E_surpl - E_thrust_used

        if E_net >= 0:
            delta_soc = (E_net * batt_eff) / battery_wh
            soc       = min(1.0, soc + delta_soc)
        else:
            delta_soc = abs(E_net) / (batt_eff * battery_wh)
            soc       = max(soc_min, soc - delta_soc)

        t_elapsed += T_orb

        # ── 7. Record state every ~1 day (avoid huge arrays) ─────────────────
        if len(time_list) == 0 or (t_elapsed - time_list[-1]*3.156e7) >= 86400:
            alt_list.append(alt_m / 1e3)
            time_list.append(t_elapsed / (365.25 * 86400))
            dv_list.append(total_dv)
            soc_list.append(soc)

        # ── 8. Termination checks ─────────────────────────────────────────────
        if alt_m <= 200e3:
            # Re-entered
            return dict(
                survived=False,
                lifetime_years=t_elapsed / (365.25 * 86400),
                total_dv_ms=total_dv,
                prop_used_g=(prop_kg - prop_kg_left) * 1000,
                mean_soc=float(np.mean(soc_list)),
                alt_history_km=np.array(alt_list),
                time_history_yr=np.array(time_list),
                dv_history_ms=np.array(dv_list)
            )

        if t_elapsed >= sim_years * 365.25 * 86400:
            break

    return dict(
        survived=True,
        lifetime_years=sim_years,
        total_dv_ms=total_dv,
        prop_used_g=(prop_kg - prop_kg_left) * 1000,
        mean_soc=float(np.mean(soc_list)),
        alt_history_km=np.array(alt_list),
        time_history_yr=np.array(time_list),
        dv_history_ms=np.array(dv_list)
    )


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION: compare to Week 2 (no thruster)
# ══════════════════════════════════════════════════════════════════════════════
def validate():
    print("=" * 60)
    print("VALIDATION: No-thruster decay vs Week 2 simulation")
    print("=" * 60)

    # Force thruster OFF by setting panel area too small to enable thrusting
    for h0 in [400, 500, 600]:
        r = simulate(h0, panel_area_m2=0.0,   # no panels = no thrusting
                     battery_wh=40, sim_years=1)
        week2_ref = {400: 23.7, 500: 2.3, 600: 0.2}   # from your Week 2 results
        decay = h0 - r['alt_history_km'][-1]
        ref   = week2_ref[h0]
        err   = abs(decay - ref) / ref * 100
        status = "✓" if err < 15 else "✗ CHECK"
        print(f"  {h0} km: decay={decay:.2f} km/yr  "
              f"(Week 2: {ref:.1f} km/yr)  error={err:.1f}%  {status}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# ANCHOR CASE: 6U at 500 km, 0.09 m², 40 Wh
# ══════════════════════════════════════════════════════════════════════════════
def anchor_case():
    print("=" * 60)
    print("ANCHOR CASE: 6U CubeSat, 500 km, 0.09 m², 40 Wh")
    print("=" * 60)

    r = simulate(500, panel_area_m2=0.09, battery_wh=40, sim_years=5)

    print(f"  Survived 5 years:      {r['survived']}")
    print(f"  Final altitude:        {r['alt_history_km'][-1]:.1f} km")
    print(f"  Total Δv provided:     {r['total_dv_ms']:.2f} m/s")
    print(f"  Propellant consumed:   {r['prop_used_g']:.2f} g")
    print(f"  Mean battery SoC:      {r['mean_soc']*100:.1f}%")

    # Quick sanity check
    alt_drop = 500 - r['alt_history_km'][-1]
    expected_drop_no_thrust = 2.3 * 5   # ~11.5 km over 5 years without thruster
    print(f"  Altitude drop:         {alt_drop:.2f} km  "
          f"(without thruster would be ~{expected_drop_no_thrust:.1f} km)")
    print()

    # Plot altitude over 5 years
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(r['time_history_yr'], r['alt_history_km'],
                 'b-', linewidth=2)
    axes[0].axhline(500, color='gray', linestyle='--',
                    alpha=0.5, label='Initial altitude')
    axes[0].set_ylabel('Altitude [km]', fontsize=12)
    axes[0].set_title('Anchor Case: 6U CubeSat, 500 km, 0.09 m² panels, 40 Wh battery',
                       fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([490, 505])

    axes[1].plot(r['time_history_yr'], r['dv_history_ms'],
                 'g-', linewidth=2)
    axes[1].set_ylabel('Cumulative Δv [m/s]', fontsize=12)
    axes[1].set_xlabel('Time [years]', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('anchor_case_5yr.png', dpi=150)
    plt.show()
    print("  Figure saved: anchor_case_5yr.png")


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    validate()
    anchor_case()