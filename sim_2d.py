"""
2D steady-state thermal model for road-embedded thermoelectric generators (TEG).
Solves the 2D conduction equation over a symmetric repeating unit slice to 
evaluate lateral spreading resistance against the simplified 1D layer model.

Author: Kavish Shah
"""

import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Geometric and thermal properties
H = 2.00          # Total depth from road surface to sewer heat sink (m)
D_MOD = 0.05      # Module embed depth below surface (m)
T_MOD = 0.004     # Module thickness (m)
K_ASPH = 1.73     # Asphalt thermal conductivity (W/m-K)
K_SOIL = 1.20     # Surrounding soil conductivity (W/m-K)

# Calibrated module resistance matching 1D formulation
R_MOD_1D = 0.0016 / (1.5 * 0.40)
K_MOD = T_MOD / R_MOD_1D

# Heat pipe / riser equivalent conductivity
R_RISER = 0.020   # Effective riser thermal resistance (m^2-K/W)
K_PIPE = (2.00 - 0.05) / R_RISER

# Boundary temperatures
T_ROAD = 55.0     # Surface temperature (deg C)
T_SEWER = 20.0    # Sewer sink temperature (deg C)


def solve(pitch, mod_w, pipe_w, ny=2001, nx=61):
    """
    Solve steady-state 2D heat conduction for a half-pitch symmetric slice.
    
    Parameters:
        pitch  : Full periodic spacing between modules (m)
        mod_w  : Module half-width (m)
        pipe_w : Riser half-width (m)
        ny, nx : Grid resolutions along vertical and horizontal axes
    """
    W = pitch / 2.0
    dx = W / (nx - 1)
    dy = H / (ny - 1)

    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Map material conductivity field
    K = np.full((nx, ny), K_SOIL)
    K[Y < D_MOD] = K_ASPH
    K[(X <= pipe_w) & (Y > D_MOD)] = K_PIPE
    K[(np.abs(Y - D_MOD) <= T_MOD / 2.0) & (X <= mod_w)] = K_MOD

    # Interface harmonic mean conductivities
    aE = np.zeros_like(K)
    aW = np.zeros_like(K)
    aN = np.zeros_like(K)
    aS = np.zeros_like(K)

    aE[:-1, :] = (2.0 * K[:-1, :] * K[1:, :] / (K[:-1, :] + K[1:, :])) / dx**2
    aW[1:, :]  = (2.0 * K[1:, :] * K[:-1, :] / (K[1:, :] + K[:-1, :])) / dx**2
    aN[:, 1:]  = (2.0 * K[:, 1:] * K[:, :-1] / (K[:, 1:] + K[:, :-1])) / dy**2
    aS[:, :-1] = (2.0 * K[:, :-1] * K[:, 1:] / (K[:, :-1] + K[:, 1:])) / dy**2

    # Linear system assembly
    N = nx * ny
    n = np.arange(N).reshape(nx, ny)
    interior = np.ones((nx, ny), dtype=bool)
    interior[:, 0] = interior[:, -1] = False

    rows, cols, vals = [], [], []
    b = np.zeros(N)

    # Dirichlet boundaries (top surface and bottom sink)
    for j, Tb in ((0, T_ROAD), (ny - 1, T_SEWER)):
        rows.append(n[:, j])
        cols.append(n[:, j])
        vals.append(np.ones(nx))
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
        rows.append(n[ok])
        cols.append(nb[ok])
        vals.append(-a[ok])
        diag += a * interior

    rows.append(n[interior])
    cols.append(n[interior])
    vals.append(diag[interior])

    A = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N)
    )
    T = spla.spsolve(A, b).reshape(nx, ny)

    # Output calculations across active module zone
    j_top = int(round((D_MOD - T_MOD / 2.0) / dy))
    j_bot = int(round((D_MOD + T_MOD / 2.0) / dy))
    i_end = max(1, int(round(mod_w / dx)))

    dT = float(np.mean(T[:i_end, j_top] - T[:i_end, j_bot]))
    q = K_MOD * dT / T_MOD
    coverage = mod_w / W

    return T, K, dT, q, coverage


def eff(dTm, Th=328.15, ZT=1.0):
    """Maximum thermoelectric conversion efficiency."""
    M = math.sqrt(1.0 + ZT)
    return (dTm / Th) * (M - 1.0) / (M + (Th - dTm) / Th)


def one_d(R_riser):
    """Analytical 1D series resistance solution."""
    R_tot = D_MOD / K_ASPH + R_MOD_1D + R_riser
    dT = R_MOD_1D / R_tot * (T_ROAD - T_SEWER)
    q = (T_ROAD - T_SEWER) / R_tot
    return dT, q, eff(dT) * q


if __name__ == "__main__":
    # 1. Grid convergence study
    print("--- Grid Convergence Study ---")
    dT1, q1, P1 = one_d(R_RISER)
    print(f"{'Mesh (nx x ny)':>16s} {'dy (mm)':>9s} {'dT mod (C)':>12s} {'Power (W/m2)':>14s} {'% of 1D':>9s}")
    for nx, ny in ((31, 1001), (41, 2001), (61, 4001), (81, 8001)):
        _, _, dT, q, _ = solve(pitch=0.50, mod_w=0.25, pipe_w=0.25, nx=nx, ny=ny)
        P = eff(dT) * q
        dy_mm = 2000.0 / (ny - 1)
        print(f"{nx:6d} x {ny:<7d} {dy_mm:9.2f} {dT:12.3f} {P:14.3f} {100.0 * P / P1:8.1f}%")
    print(f"Analytical 1D Baseline:             {dT1:12.3f} {P1:14.3f}\n")

    # 2. Benchmark against 1D continuous case
    print("--- Model Benchmark (Continuous Coverage) ---")
    T, K, dT2, q2, _ = solve(pitch=0.50, mod_w=0.25, pipe_w=0.25, nx=81, ny=8001)
    P2 = eff(dT2) * q2
    print(f"Module Delta-T:  1D = {dT1:6.3f} C  |  2D = {dT2:6.3f} C")
    print(f"Heat Flux:       1D = {q1:6.0f} W/m2|  2D = {q2:6.0f} W/m2")
    print(f"Power Output:    1D = {P1:6.3f} W/m2|  2D = {P2:6.3f} W/m2")
    print(f"Agreement Ratio: {100.0 * P2 / P1:.1f}%\n")

    # 3. Discrete module geometry sweep
    print("--- Discrete Module Geometry Sweep ---")
    print(f"{'Module':>8s} {'Pipe':>8s} {'Cover':>7s} {'dT (C)':>8s} {'W/m2 (mod)':>12s} {'W/m2 (road)':>13s} {'% Baseline':>11s}")
    base_road = None
    test_cases = [
        (0.25, 0.25),
        (0.20, 0.05),
        (0.20, 0.02),
        (0.10, 0.02),
        (0.05, 0.01)
    ]

    for mod_w, pipe_w in test_cases:
        T, K, dT, q, cov = solve(pitch=0.50, mod_w=mod_w, pipe_w=pipe_w, nx=81, ny=8001)
        P_mod = eff(dT) * q
        P_road = P_mod * cov
        if base_road is None:
            base_road = P_road
        print(f"{2 * mod_w * 100:6.0f}cm {2 * pipe_w * 100:6.0f}cm {100.0 * cov:6.0f}% "
              f"{dT:8.3f} {P_mod:12.3f} {P_road:13.3f} {100.0 * P_road / base_road:10.1f}%")

    np.save("sim2d_T.npy", T)
    np.save("sim2d_K.npy", K)
    print("\nState arrays saved to sim2d_T.npy and sim2d_K.npy.")
