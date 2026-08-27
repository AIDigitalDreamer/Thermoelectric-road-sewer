"""
Thermal resistance model for the road-to-sewer thermoelectric system.
Kavish Shah

Every resistance below is "per square metre of module", so they can just be
added together like resistors in series.

Run this file to reproduce the numbers in Sections 7 to 10 of the paper.
"""

import math

# ---- values from Table 1 of the paper -------------------------------------
k_soil     = 1.2      # W/m/K   moist soil                      [ref 11]
k_pipe     = 0.19     # W/m/K   PVC
t_pipe     = 0.010    # m       pipe wall thickness
h_water    = 500.0    # W/m2/K  sewage to pipe wall (assumed)
R_contact  = 315e-6   # m2K/W   one thermal interface            [ref 9]
k_leg      = 1.5      # W/m/K   bismuth telluride
t_leg      = 0.0016   # m       height of the legs
fill       = 0.40     # fraction of the module area that is actually legs
ZT         = 1.0      # figure of merit                          [ref 5]
N_pairs    = 127      # pairs in a standard 40x40 mm module
S_pair     = 400e-6   # V/K     per pair
k_asphalt  = 1.73     # W/m/K   asphalt concrete
cp_water   = 4180.0   # J/kg/K  specific heat of water

# ---- the case I looked at --------------------------------------------------
r_pipe   = 0.15       # m    outer radius of a 300 mm pipe
depth    = 2.0        # m    buried depth
dT_avail = 10.0       # K    sewage warmer than the far-away ground
T_hot    = 293.15     # K    20 C - the top of the measured sewer range [ref 2]


def soil_resistance(r, z, k, spread=1.0):
    """Standard formula for heat leaving a pipe buried at depth z.
    `spread` is how many times bigger the cold-side plate is than the module."""
    return r * math.log(2 * z / r) / k / spread


def efficiency(dT_module, T_h, ZT):
    """How much of the heat a module turns into electricity."""
    carnot = dT_module / T_h
    T_c = T_h - dT_module
    return carnot * (math.sqrt(1 + ZT) - 1) / (math.sqrt(1 + ZT) + T_c / T_h)


def solve(spread=1.0, ZT=ZT, dT=dT_avail):
    R_water  = 1 / h_water
    R_pipe   = t_pipe / k_pipe
    R_module = t_leg / (k_leg * fill)
    R_soil   = soil_resistance(r_pipe, depth, k_soil, spread)
    R_total  = R_water + R_pipe + 2 * R_contact + R_module + R_soil

    phi       = R_module / R_total          # the capture fraction
    dT_module = phi * dT
    heat_flux = dT / R_total                # W/m2
    power     = efficiency(dT_module, T_hot, ZT) * heat_flux   # W/m2
    voltage   = N_pairs * S_pair * dT_module

    return dict(R_water=R_water, R_pipe=R_pipe, R_contact=2 * R_contact,
                R_module=R_module, R_soil=R_soil, R_total=R_total,
                phi=phi, dT_module=dT_module, heat_flux=heat_flux,
                power=power, mW_per_cm2=power / 10, voltage=voltage)


# ---------------------------------------------------------------------------
base = solve()

print("PAPER TABLE 2 (sewer-pipe row) and Section 8 - where the difference goes")
print("-" * 52)
for key, name in [("R_water", "water to pipe wall"), ("R_pipe", "pipe wall"),
                  ("R_contact", "two contact surfaces"), ("R_module", "the module itself"),
                  ("R_soil", "soil out to the far field")]:
    print(f"  {name:<26s} {base[key]:8.4f}   {100*base[key]/base['R_total']:5.1f} %")
print(f"  {'total':<26s} {base['R_total']:8.4f}   100.0 %")

print()
print(f"  capture fraction        = {100*base['phi']:.2f} %")
print(f"  temperature across module = {base['dT_module']:.3f} C")
print(f"  power density           = {base['mW_per_cm2']:.2e} mW/cm2")
print(f"  voltage per module      = {1000*base['voltage']:.1f} mV   (need 20 mV to start)")
print(f"  modules needed in series to reach 20 mV = {math.ceil(0.020/base['voltage'])}")

area_per_km = 2 * math.pi * r_pipe * 1000
print(f"  1 km of pipe fully wrapped = {area_per_km:.0f} m2  ->  {base['power']*area_per_km:.2f} W")

print()
print("A cold-side plate on the sewer-pipe design - background to Section 7.4")
print("-" * 62)
print(f"  {'plate':>6s} {'capture':>9s} {'volts':>8s} {'gain':>7s} {'W per km':>10s}")
for s in [1, 5, 20, 50]:
    r = solve(spread=s)
    print(f"  {s:5d}x {100*r['phi']:8.1f}% {1000*r['voltage']:7.0f}mV "
          f"{r['power']/base['power']:6.0f}x {r['power']*area_per_km:9.1f}")

print()
print("PAPER FIGURE 3 - better material vs better cold side")
print("-" * 46)
for z in [1.0, 1.5, 2.0, 3.0]:
    print(f"  ZT = {z:<4g}          {solve(ZT=z)['power']/base['power']:5.1f}x")
for s in [5, 20, 100]:
    print(f"  plate {s:>3d}x bigger   {solve(spread=s)['power']/base['power']:5.1f}x")

print()
print("PAPER SECTION 8 - the impossible-number check")
print("-" * 46)
eff_full = efficiency(10.0, T_hot, ZT)
heat_needed = 100 / eff_full          # to get 100 W/m2 out
soil_thickness = 10 * k_soil / heat_needed
print(f"  best possible efficiency at 10 C  = {100*eff_full:.2f} %")
print(f"  heat needed for 10 mW/cm2         = {heat_needed:,.0f} W/m2")
print(f"  soil would have to be only        = {1000*soil_thickness:.2f} mm thick")


# ---------------------------------------------------------------------------
# The road case (RTEP).  Different geometry: a flat panel, not a buried pipe,
# so the soil term is a simple slab instead of the cylinder formula.
print()
print("PAPER TABLE 2 (road-panel row)")
print("-" * 46)
d_module   = 0.05      # m   panel 5 cm below the surface
t_subbase  = 0.30      # m   down to the layer that stays at a steady temperature
dT_road    = 20.0      # K   asphalt surface to sub-base, daytime
T_hot_road = 323.15    # K   50 C, a hot asphalt surface

R_asphalt = d_module / k_asphalt
R_sub     = t_subbase / k_soil
R_mod     = t_leg / (k_leg * fill)
R_tot_rd  = R_asphalt + R_sub + 2 * R_contact + R_mod
phi_road  = R_mod / R_tot_rd
dT_mod_rd = phi_road * dT_road
P_road    = efficiency(dT_mod_rd, T_hot_road, ZT) * (dT_road / R_tot_rd)

print(f"  asphalt above the panel   {R_asphalt:8.4f} m2K/W")
print(f"  sub-base below the panel  {R_sub:8.4f} m2K/W")
print(f"  the module                {R_mod:8.4f} m2K/W")
print(f"  capture fraction          {100*phi_road:8.2f} %")
print(f"  temperature across module {dT_mod_rd:8.3f} C")
print(f"  power density             {P_road/10:8.2e} mW/cm2")


# ---------------------------------------------------------------------------
# THE COUPLED DESIGN (Section 9) - road TEG with its cold side carried down
# to the sewer by a heat pipe, instead of dumped into the sub-base soil.
print()
print("PAPER SECTION 9.1 - the coupled road-to-sewer design")
print("-" * 58)
T_road    = 55.0        # C   hot asphalt in the afternoon
T_sewage  = 20.0        # C   sewage
dT_couple = T_road - T_sewage
R_riser   = 0.020       # m2K/W   a heat pipe, taken conservatively
T_hot_c   = 328.15      # K   55 C

R_cold  = R_riser + 1 / h_water
R_tot_c = R_asphalt + 2 * R_contact + R_mod + R_cold
phi_c   = R_mod / R_tot_c
dT_mod  = phi_c * dT_couple
q_c     = dT_couple / R_tot_c                       # W/m2 of module
P_c     = efficiency(dT_mod, T_hot_c, ZT) * q_c     # W/m2 of module
V_c     = N_pairs * S_pair * dT_mod

print(f"  asphalt above the module   {R_asphalt:7.4f} m2K/W")
print(f"  the module                 {R_mod:7.4f} m2K/W")
print(f"  heat pipe down to sewer    {R_riser:7.4f} m2K/W")
print(f"  sewage water film          {1/h_water:7.4f} m2K/W")
print(f"  capture fraction           {100*phi_c:7.1f} %   (was 0.9 % using sub-base soil)")
print(f"  temperature across module  {dT_mod:7.2f} C")
print(f"  heat carried to the sewer  {q_c:7.0f} W per m2 of module")
print(f"  electrical output          {P_c:7.3f} W/m2  = {P_c/10:.4f} mW/cm2")
print(f"  voltage per module         {1000*V_c:7.0f} mV  (needs 20 mV to start)")
print(f"  improvement over soil cooling: {P_c/P_road:.0f} times")

# ---------------------------------------------------------------------------
# How much road can one sewer take before it warms up too much?
print()
print("PAPER TABLE 4 - how much module area one sewer can absorb")
print("-" * 58)
print(f"  {'flow':>8s} {'rise':>6s} {'heat':>10s} {'road area':>11s} {'electrical':>11s}")
for flow_ls in (30, 100, 300):
    for rise in (0.5, 1.0):
        mass  = flow_ls / 1000 * 1000                # kg/s
        P_ht  = mass * cp_water * rise               # W of heat the sewer can take
        area  = P_ht / q_c                           # m2 of module
        elec  = area * P_c
        print(f"  {flow_ls:5d} L/s {rise:5.1f} C {P_ht/1000:8.0f} kW {area:9.0f} m2 "
              f"{elec:9.0f} W")

# ---------------------------------------------------------------------------
print()
print("PAPER SECTION 9.3 - what one coupled section delivers")
print("-" * 58)
flow, rise, duty = 100.0, 1.0, 0.25    # L/s, C, daytime fraction
area_ok = (flow / 1000 * 1000) * cp_water * rise / q_c
peak    = area_ok * P_c
daily   = peak * duty * 24 / 1000      # kWh/day
print(f"  sewer at {flow:.0f} L/s, warming limited to {rise:.1f} C")
print(f"  road area that can be coupled  {area_ok:6.0f} m2")
print(f"    (on a 7 m wide road that is {area_ok/7:.0f} m of length)")
print(f"  peak electrical output         {peak:6.0f} W")
print(f"  average across the day         {peak*duty:6.0f} W")
print(f"  energy per day                 {daily:6.1f} kWh")
print(f"  -> electric car   at 0.15 kWh/km   {daily/0.15:5.0f} km of range per day")
print(f"  -> electric scooter at 0.03 kWh/km {daily/0.03:5.0f} km of range per day")
print(f"  -> 30 W LED streetlights, 12 h/night  {daily/(0.030*12):5.0f} lamps")


# ---------------------------------------------------------------------------
# Section 10.4 - can the sewer be cooled instead?
print()
print("PAPER SECTION 10.4 - can the sewer be cooled instead?")
print("-" * 58)

# (a) spend the electricity on a Peltier cooler
COP = 0.5
print("  (a) spend the electricity on a Peltier cooler")
print(f"      heat pushed in to generate      {q_c:8.1f} W/m2")
print(f"      electricity produced            {P_c:8.3f} W/m2")
print(f"      heat a cooler could pump back   {P_c*COP:8.3f} W/m2")
print(f"      shortfall                       {q_c/(P_c*COP):8.0f} to 1  -> not possible")

# (b) run the system backwards at night
print()
print("  (b) run it backwards at night (road colder than sewage)")
hours_day, hours_night = 8.0, 12.0
into = q_c * hours_day
for dT_n, label in ((3.0, "mild, cloudy"), (5.0, "clear"), (8.0, "clear winter")):
    q_n   = dT_n / R_tot_c
    outof = q_n * hours_night
    print(f"      {label:<14s} night difference {dT_n:.0f} C -> {outof:6.0f} Wh/m2 removed"
          f"  = {100*outof/into:3.0f} % of the {into:.0f} Wh/m2 put in by day")

# (c) ventilate instead
print()
print("  (c) forget cooling - ventilate the chamber instead")
daily_kWh   = 2.3          # from Section 9.3
V_manhole   = 1.2 * 1.2 * 3.0
print(f"      energy available   {daily_kWh} kWh/day = {daily_kWh*1000/24:.0f} W continuous")
for fan_W, flow, label in ((5, 60, "trickle fan"), (15, 180, "small duct fan"),
                           (60, 600, "purge blower")):
    ach = flow / V_manhole
    share = 100 * fan_W / (daily_kWh * 1000 / 24)
    print(f"      {label:<15s} {fan_W:3d} W, {flow:3d} m3/h -> {ach:4.0f} air changes/hour, "
          f"uses {share:3.0f} % of the output, runs continuously")
