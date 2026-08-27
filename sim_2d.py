"""
A 2-D check on the simple model.

teg_model.py treats heat as flowing straight down through one layer after
another. Real heat also spreads sideways, and in a real road the modules are
separate tiles feeding narrow heat pipes rather than a continuous sheet. This
program solves the 2-D heat equation for one repeating slice of road and
compares the answer with the simple model.

Kavish Shah

Needs numpy and scipy.  pip install numpy scipy
"""

import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------- fixed quantities ----------------
H       = 2.00       # road surface down to the sewer, m
D_MOD   = 0.05       # module depth below the surface, m
T_MOD   = 0.004      # module thickness, m
K_ASPH  = 1.73       # W/m/K
K_SOIL  = 1.20
R_MOD_1D = 0.0016 / (1.5 * 0.40)            # the 1-D model's module resistance
K_MOD   = T_MOD / R_MOD_1D                  # so the two models match
R_RISER = 0.020      # m2K/W, the riser resistance assumed in the paper
K_PIPE  = (2.00 - 0.05) / R_RISER   # conductivity that gives exactly that
T_ROAD  = 55.0       # deg C
T_SEWER = 20.0


def solve(pitch, mod_w, pipe_w, ny=2001, nx=61):
    """Solve one repeating slice. Widths are half-widths, by symmetry."""
    W = pitch / 2
    dx, dy = W / (nx - 1), H / (ny - 1)
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    K = np.full((nx, ny), K_SOIL)
    K[Y < D_MOD] = K_ASPH
    K[(X <= pipe_w) & (Y > D_MOD)] = K_PIPE
    K[(np.abs(Y - D_MOD) <= T_MOD / 2) & (X <= mod_w)] = K_MOD

    harm = lambda a, b: 2 * a * b / (a + b)
    aE = np.zeros_like(K); aW = np.zeros_like(K)
    aN = np.zeros_like(K); aS = np.zeros_like(K)
    aE[:-1, :] = harm(K[:-1, :], K[1:, :]) / dx**2
    aW[1:, :]  = harm(K[1:, :],  K[:-1, :]) / dx**2
    aN[:, 1:]  = harm(K[:, 1:],  K[:, :-1]) / dy**2
    aS[:, :-1] = harm(K[:, :-1], K[:, 1:])  / dy**2

    N = nx * ny
    n = np.arange(N).reshape(nx, ny)
    interior = np.ones((nx, ny), dtype=bool)
    interior[:, 0] = interior[:, -1] = False

    rows, cols, vals = [], [], []
    b = np.zeros(N)
    for j, Tb in ((0, T_ROAD), (ny - 1, T_SEWER)):
        rows.append(n[:, j]); cols.append(n[:, j]); vals.append(np.ones(nx))
        b[n[:, j]] = Tb

    diag = np.zeros((nx, ny))
    for a, shift, axis in ((aE, -1, 0), (aW, 1, 0), (aN, 1, 1), (aS, -1, 1)):
        nb = np.roll(n, shift, axis=axis)
        ok = interior & (a > 0)
        if axis == 0:
            if shift == -1:
                ok[-1, :] = False
            else:
                ok[0, :] = False
        rows.append(n[ok]); cols.append(nb[ok]); vals.append(-a[ok])
        diag += a * interior
    rows.append(n[interior]); cols.append(n[interior]); vals.append(diag[interior])

    A = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    T = spla.spsolve(A, b).reshape(nx, ny)

    j_top = int(round((D_MOD - T_MOD / 2) / dy))
    j_bot = int(round((D_MOD + T_MOD / 2) / dy))
    i_end = max(1, int(round(mod_w / dx)))
    dT = float(np.mean(T[:i_end, j_top] - T[:i_end, j_bot]))
    q = K_MOD * dT / T_MOD                        # W per m2 of module
    return T, K, dT, q, mod_w / W


def eff(dTm, Th=328.15, ZT=1.0):
    M = math.sqrt(1 + ZT)
    return (dTm / Th) * (M - 1) / (M + (Th - dTm) / Th)


def one_d(R_riser):
    # No contact or water-film terms: the 2-D grid does not contain them either,
    # so this is the like-for-like twin of the simulation.
    R_tot = D_MOD / K_ASPH + R_MOD_1D + R_riser
    dT = R_MOD_1D / R_tot * (T_ROAD - T_SEWER)
    q = (T_ROAD - T_SEWER) / R_tot
    return dT, q, eff(dT) * q


# ======================================================================
print("GRID CONVERGENCE - does the answer stop moving as the mesh is refined?")
print("=" * 70)
print("  Module and heat pipe cover the full width, so the problem is")
print("  one-dimensional by symmetry and the 1-D model should reproduce it.\n")
dT1, q1, P1 = one_d(R_RISER)
print(f"  {'grid (nx x ny)':>16s} {'dy':>8s} {'dT module':>11s} {'output':>12s} {'vs 1-D':>8s}")
for nx, ny in ((31, 1001), (41, 2001), (61, 4001), (81, 8001)):
    _, _, dT, q, _ = solve(pitch=0.50, mod_w=0.25, pipe_w=0.25, nx=nx, ny=ny)
    P = eff(dT) * q
    print(f"  {nx:6d} x {ny:<7d} {2000/(ny-1):6.2f} mm {dT:9.3f} C {P:9.3f} W/m2 {100*P/P1:6.1f} %")
print(f"\n  1-D model, same geometry:                {dT1:9.3f} C {P1:9.3f} W/m2")

# ======================================================================
print("CASE A - does the solver agree with the simple model?")
print("=" * 68)
print("  Module and heat pipe both cover the full width, which is what the")
print("  1-D model quietly assumes. The two should agree.\n")

T, K, dT2, q2, cov = solve(pitch=0.50, mod_w=0.25, pipe_w=0.25, nx=81, ny=8001)
dT1, q1, P1 = one_d(R_RISER)
P2 = eff(dT2) * q2
print(f"  temperature across module   1-D {dT1:7.3f} C     2-D {dT2:7.3f} C")
print(f"  heat flux                   1-D {q1:7.0f} W/m2  2-D {q2:7.0f} W/m2")
print(f"  electrical output           1-D {P1:7.3f} W/m2  2-D {P2:7.3f} W/m2")
print(f"  the 2-D answer is {100*P2/P1:.0f} % of the 1-D answer")

# ======================================================================
print()
print("CASE B - realistic separate tiles feeding narrow heat pipes")
print("=" * 74)
print("  Now the modules are tiles with gaps between them, each feeding a")
print("  narrow pipe, so heat has to funnel sideways to reach it.")
print("  Two ways of counting: per square metre of MODULE, and per square")
print("  metre of ROAD. The second is what a city would actually care about.\n")
print(f"  {'module':>7s} {'pipe':>6s} {'cover':>6s} {'dT mod':>8s} "
      f"{'W/m2 module':>12s} {'W/m2 road':>11s} {'vs full':>8s}")
base_road = None
for mod_w, pipe_w in ((0.25, 0.25), (0.20, 0.05), (0.20, 0.02),
                      (0.10, 0.02), (0.05, 0.01)):
    T, K, dT, q, cov = solve(pitch=0.50, mod_w=mod_w, pipe_w=pipe_w, nx=81, ny=8001)
    P_mod  = eff(dT) * q          # per m2 of module
    P_road = P_mod * cov          # per m2 of road
    if base_road is None:
        base_road = P_road
    print(f"  {2*mod_w*100:5.0f}cm {2*pipe_w*100:4.0f}cm {100*cov:5.0f}% "
          f"{dT:7.3f}C {P_mod:10.3f}   {P_road:9.3f}   {100*P_road/base_road:6.0f} %")

print()
print("  Counted per square metre of road, nothing beats full coverage. The")
print("  narrow-pipe cases look good per module only because each module is")
print("  drawing heat from a much larger patch of road than it covers.")
print("  Two rules follow: cover as much of the road as the budget allows,")
print("  and make each heat pipe wide compared with the module it serves.")

np.save("sim2d_T.npy", T)
np.save("sim2d_K.npy", K)
print("\n  temperature field saved for the figure")
