
import numpy as np


def calculate_thermodynamics(g_unfolded, g_transition, g_native):
    """
    Calculate free-energy differences for a three-state model.
    Units: kJ/mol
    """

    delta_g_fold = g_native - g_unfolded

    delta_g_fold_barrier = g_transition - g_unfolded

    delta_g_unfold_barrier = g_transition - g_native

    return {
        "delta_g_fold": delta_g_fold,
        "delta_g_fold_barrier": delta_g_fold_barrier,
        "delta_g_unfold_barrier": delta_g_unfold_barrier,
        "is_transition_maximum": (
            g_transition > g_unfolded
            and g_transition > g_native
        )
    }


def generate_energy_curve(g_unfolded, g_transition, g_native):
    """
    Generate a smooth illustrative energy profile.
    Reaction coordinate ranges from 0 to 1.
    """

    t = np.linspace(0, 1, 150)

    # Smooth interpolation from unfolded to transition state
    smooth_1 = 3 * t**2 - 2 * t**3

    x1 = 0.5 * t
    y1 = g_unfolded + (
        g_transition - g_unfolded
    ) * smooth_1

    # Smooth interpolation from transition state to native state
    smooth_2 = 3 * t**2 - 2 * t**3

    x2 = 0.5 + 0.5 * t
    y2 = g_transition + (
        g_native - g_transition
    ) * smooth_2

    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])

    return x, y

def simulate_temperature_sweep(
    delta_g_fold_barrier,
    delta_g_unfold_barrier,
    delta_s_fold=-0.25,
    t_ref=298.15,
    t_min=275.0,
    t_max=400.0,
    steps=150
):
    """
    Calculate temperature-dependent folding and unfolding rate constants
    incorporating conformational entropy changes: ΔG‡(T) = ΔG‡(T0) - (T - T0)ΔS‡.
    """
    T = np.linspace(t_min, t_max, steps)
    R = 8.314462618e-3       # kJ / (mol * K)
    k_B = 1.380649e-23       # J / K
    h = 6.62607015e-34       # J * s
    kappa = 1e-6             # Effective transmission factor for protein folding

    # 假設過渡態 (TS) 已喪失約 60% 的未折疊態構形熵 (Tanford beta_T ≈ 0.6)
    delta_s_fold_barrier = 0.6 * delta_s_fold
    delta_s_unfold_barrier = -0.4 * delta_s_fold

    # 計算各溫度下的有效活化自由能障
    dg_f_T = delta_g_fold_barrier - (T - t_ref) * delta_s_fold_barrier
    dg_u_T = delta_g_unfold_barrier - (T - t_ref) * delta_s_unfold_barrier

    prefactor = kappa * (k_B * T / h)

    exp_fold = np.clip(-dg_f_T / (R * T), -100.0, 50.0)
    exp_unfold = np.clip(-dg_u_T / (R * T), -100.0, 50.0)

    k_fold = prefactor * np.exp(exp_fold)
    k_unfold = prefactor * np.exp(exp_unfold)

    # 解析計算熔解溫度 Tm (當 ΔG_fold(Tm) = 0 時，k_fold == k_unfold)
    delta_g_fold_ref = delta_g_fold_barrier - delta_g_unfold_barrier
    if delta_s_fold < 0 and delta_g_fold_ref < 0:
        t_m_exact = t_ref + (delta_g_fold_ref / delta_s_fold)
        t_m = float(t_m_exact) if (t_min <= t_m_exact <= t_max) else None
    else:
        t_m = None

    return {
        "T": T,
        "T_celsius": T - 273.15,
        "inv_T": 1000.0 / T,
        "k_fold": k_fold,
        "k_unfold": k_unfold,
        "ln_k_fold": np.log(k_fold),
        "ln_k_unfold": np.log(k_unfold),
        "t_m": t_m
    }

def analyze_structural_biophysics(ca_coords, sequence, cutoff=8.0):
    """
    Calculate Relative Contact Order (RCO), Long-range Contact Ratio,
    Radius of Gyration (Rg), Hydrophobic Core Burial %, and map them
    to estimated 3-state thermodynamic parameters (U, TS, N).
    """
    coords = np.array(ca_coords, dtype=float)
    L = len(coords)

    if L < 4:
        return None

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    idx_i, idx_j = np.triu_indices(L, k=3)
    pair_dists = dist_matrix[idx_i, idx_j]
    contact_mask = pair_dists <= cutoff

    seq_separations = np.abs(idx_j[contact_mask] - idx_i[contact_mask])
    n_contacts = len(seq_separations)

    if n_contacts > 0:
        rco = float(np.sum(seq_separations) / (L * n_contacts))
        long_range_ratio = float(np.sum(seq_separations >= 12) / n_contacts)
    else:
        rco = 0.0
        long_range_ratio = 0.0

    center_of_mass = np.mean(coords, axis=0)
    radial_dists = np.linalg.norm(coords - center_of_mass, axis=1)
    rg = float(np.sqrt(np.mean(radial_dists**2)))

    hydrophobic_set = set("AVILMFYW")
    hydro_indices = [i for i, aa in enumerate(sequence[:L]) if aa in hydrophobic_set]

    if len(hydro_indices) > 0 and rg > 0:
        buried_count = sum(1 for i in hydro_indices if radial_dists[i] < 0.75 * rg)
        core_burial_ratio = float(buried_count / len(hydro_indices))
    else:
        core_burial_ratio = 0.0

    est_u = 0.0
    est_ts = float(np.clip(round(10.0 + 120.0 * rco, 1), 5.0, 55.0))
    raw_dg = -0.25 * L * (0.6 + 0.8 * core_burial_ratio)
    est_n = float(np.clip(round(raw_dg, 1), -32.0, -5.0))

    return {
        "length": L,
        "n_contacts": int(n_contacts),
        "rco": rco,
        "long_range_ratio": long_range_ratio,
        "rg": rg,
        "core_burial_ratio": core_burial_ratio,
        "est_u": est_u,
        "est_ts": est_ts,
        "est_n": est_n
    }
