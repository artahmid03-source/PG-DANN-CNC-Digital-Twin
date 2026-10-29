import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import skfem
from skfem.models.poisson import laplace, unit_load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MATERIALS = {
    1: {
        'name': 'Cast Iron (FC250)',
        'A_jc': 240.0, 'B_jc': 320.0, 'n_jc': 0.28, 'C_jc': 0.015, 'm_jc': 1.05,
        'T_m': 1450.0, 'T_r': 293.15,
        'rho': 7200.0, 'cp': 460.0, 'k_w': 48.0,
        'phi_deg': 26.5, 'mu_f': 0.40, 'mc_size': 0.22
    },
    2: {
        'name': 'Stainless/Carbon Steel (J45)',
        'A_jc': 553.1, 'B_jc': 600.8, 'n_jc': 0.234, 'C_jc': 0.0134, 'm_jc': 1.00,
        'T_m': 1733.0, 'T_r': 293.15,
        'rho': 7850.0, 'cp': 486.0, 'k_w': 38.0,
        'phi_deg': 21.0, 'mu_f': 0.54, 'mc_size': 0.26
    }
}

V_C = 200.0 / 60.0   # 3.333 m/s (200 m/min at 826 RPM)
R_NOSE_MM = 0.80     # Standard ISO carbide insert corner radius r_epsilon = 0.8 mm
MILL_DUTY = 0.22     # Intermittent face-milling radial immersion duty factor per rev

def solve_2d_tool_fea(q_eff_W_m2, l_contact_m, ap_mm, k_tool=55.0, T_amb=293.15):
    """
    2D Finite-Element Steady-State Thermal Solver (scikit-fem MeshTri) on 4mm x 4mm carbide insert.
    Includes axial fin-cooling sink scaling with DOC (ap_mm).
    """
    L_dom = 0.004
    nx, ny = 45, 45
    x = np.linspace(0, L_dom, nx)
    y = np.linspace(0, L_dom, ny)
    mesh = skfem.MeshTri.init_tensor(x, y)
    basis = skfem.CellBasis(mesh, skfem.ElementTriP1())

    # Effective 3D-to-2D axial lateral heat dissipation factor (thinner DOC dissipates heat faster laterally)
    k_eff = k_tool * (1.0 + 0.28 * (1.50 / ap_mm)**0.5)
    A_mat = k_eff * skfem.asm(laplace, basis)

    def is_contact_zone(pts):
        return (np.abs(pts[1] - L_dom) < 1e-7) & (pts[0] <= max(l_contact_m, 3.0e-4))

    contact_facets = mesh.facets_satisfying(is_contact_zone)
    fbasis = skfem.FacetBasis(mesh, skfem.ElementTriP1(), facets=contact_facets)
    b_vec = q_eff_W_m2 * skfem.asm(unit_load, fbasis)

    def is_holder_sink(pts):
        return (np.abs(pts[1]) < 1e-7) | (np.abs(pts[0] - L_dom) < 1e-7)

    sink_dofs = basis.get_dofs(is_holder_sink)
    x_init = np.full(basis.N, T_amb + 15.0)
    T_sol = skfem.solve(*skfem.condense(A_mat, b_vec, x=x_init, D=sink_dofs))
    return mesh, T_sol, float(np.max(T_sol))

def solve_regime_physics(mat_id, doc_mm, feed_mm_rev):
    mp = MATERIALS[mat_id]
    # Equivalent chip thickness h_eq accounting for nose radius r_eps and DOC (Altintas model)
    kappa_r = np.radians(45.0)
    l_arc_mm = R_NOSE_MM * kappa_r + max(0.0, (doc_mm - R_NOSE_MM * (1.0 - np.cos(kappa_r))) / np.sin(kappa_r))
    fz_eff_mm = feed_mm_rev / 2.0
    h_eq_mm = (doc_mm * fz_eff_mm) / l_arc_mm
    h_m = h_eq_mm * 1e-3
    ap_m = doc_mm * 1e-3

    # Shear angle adjusts slightly with chip load (Merchant-Oxley size effect)
    phi = np.radians(mp['phi_deg'] + 2.2 * (h_eq_mm - 0.08) / 0.10)
    gamma = np.radians(5.0)

    eps_ab = np.cos(gamma) / (2.0 * np.sqrt(3) * np.sin(phi) * np.cos(phi - gamma))
    l_ab = h_m / np.sin(phi)
    eps_dot_ab = (V_C * np.cos(gamma) / np.cos(phi - gamma)) / (0.15 * l_ab)

    def jc_residual(T_ab_arr):
        T_ab = float(T_ab_arr[0])
        T_star = np.clip((T_ab - mp['T_r']) / (mp['T_m'] - mp['T_r']), 0.0, 0.95)
        sigma_eq = (mp['A_jc'] + mp['B_jc'] * (eps_ab ** mp['n_jc'])) * \
                   (1.0 + mp['C_jc'] * np.log(max(eps_dot_ab, 1.0))) * \
                   (1.0 - (T_star ** mp['m_jc']))
        tau_ab = (sigma_eq / np.sqrt(3.0)) * 1e6
        dT_ab = (0.85 * tau_ab * (eps_ab * np.sqrt(3.0))) / (mp['rho'] * mp['cp'])
        return [T_ab - (mp['T_r'] + dT_ab)]

    T_ab_sol = float(fsolve(jc_residual, [mp['T_r'] + 220.0])[0])
    T_star = np.clip((T_ab_sol - mp['T_r']) / (mp['T_m'] - mp['T_r']), 0.0, 0.95)
    sigma_eq_MPa = (mp['A_jc'] + mp['B_jc'] * (eps_ab ** mp['n_jc'])) * \
                   (1.0 + mp['C_jc'] * np.log(max(eps_dot_ab, 1.0))) * \
                   (1.0 - (T_star ** mp['m_jc']))
    tau_ab_MPa = sigma_eq_MPa / np.sqrt(3.0)

    # Kienzle-Oxley size effect on specific cutting energy and contact mechanics
    size_factor = (0.10 / h_eq_mm) ** mp['mc_size']
    beta_f = np.arctan(mp['mu_f'])
    F_shear_N = (tau_ab_MPa * 1e6) * (ap_m * h_m / np.sin(phi)) * size_factor
    F_c_N = F_shear_N * np.cos(beta_f - gamma) / np.cos(phi + beta_f - gamma)
    F_t_N = F_shear_N * np.sin(beta_f - gamma) / np.cos(phi + beta_f - gamma)

    u_c_GPa = (F_c_N / (ap_m * (fz_eff_mm * 1e-3))) * 1e-9

    # Contact length & normal stress on tool-chip/flank zone
    l_c_m = 1.65 * (fz_eff_mm * 1e-3) * (np.cos(phi - gamma) / np.sin(phi)) * ((h_eq_mm / 0.10) ** 0.15)
    N_rake_N = F_c_N * np.cos(gamma) - F_t_N * np.sin(gamma)
    sigma_n_MPa = (N_rake_N / (ap_m * l_c_m)) * 1e-6

    # Intermittent milling effective heat flux into carbide insert
    V_chip = V_C * np.sin(phi) / np.cos(phi - gamma)
    F_fric_N = F_c_N * np.sin(gamma) + F_t_N * np.cos(gamma)
    q_fric_instant = F_fric_N * V_chip / (ap_m * l_c_m)
    R_tool = 0.24 if mat_id == 1 else 0.29
    q_eff = MILL_DUTY * R_tool * q_fric_instant

    mesh, T_field, T_int_K = solve_2d_tool_fea(q_eff, l_c_m, doc_mm)
    return {
        'material': mat_id,
        'mat_name': mp['name'],
        'DOC': doc_mm,
        'feed': feed_mm_rev,
        'h_eq_mm': round(h_eq_mm, 4),
        'F_c_N': round(F_c_N, 1),
        'u_c_GPa': round(u_c_GPa, 4),
        'sigma_n_MPa': round(sigma_n_MPa, 2),
        'T_shear_K': round(T_ab_sol, 1),
        'T_interface_K': round(T_int_K, 1),
        'T_interface_C': round(T_int_K - 273.15, 1)
    }, mesh, T_field

regimes = []
saved_plots = {}
for m_id in [1, 2]:
    for doc in [0.75, 1.50]:
        for fd in [0.25, 0.50]:
            res, mesh, T_f = solve_regime_physics(m_id, doc, fd)
            regimes.append(res)
            if (m_id == 1 and doc == 0.75 and fd == 0.25) or (m_id == 2 and doc == 1.50 and fd == 0.50):
                saved_plots[(m_id, doc, fd)] = (mesh, T_f, res)

df_phys = pd.DataFrame(regimes)
ref_row = df_phys[(df_phys['material']==1) & (df_phys['DOC']==0.75) & (df_phys['feed']==0.25)].iloc[0]
df_phys['scale_uc'] = (df_phys['u_c_GPa'] / ref_row['u_c_GPa']).round(4)
df_phys['scale_sigma_n'] = (df_phys['sigma_n_MPa'] / ref_row['sigma_n_MPa']).round(4)
df_phys['scale_T_int'] = (df_phys['T_interface_K'] / ref_row['T_interface_K']).round(4)

df_phys.to_csv('data/processed/physics_regimes.csv', index=False)
print("--- REFINED 2D FEA & CONTINUUM THERMO-MECHANICAL SCALING TABLE (ALL 8 REGIMES) ---")
print(df_phys[['material', 'DOC', 'feed', 'h_eq_mm', 'u_c_GPa', 'sigma_n_MPa', 'T_interface_K', 'T_interface_C', 'scale_uc', 'scale_sigma_n']].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
keys = [(1, 0.75, 0.25), (2, 1.50, 0.50)]
titles = [
    "Source Regime: Cast Iron (DOC=0.75mm, f=0.25mm/rev)",
    "Target Extreme: Steel J45 (DOC=1.50mm, f=0.50mm/rev)"
]
for ax, k, title in zip(axes, keys, titles):
    msh, Tf, rinfo = saved_plots[k]
    pts = msh.p * 1000.0
    tcf = ax.tricontourf(pts[0], pts[1], msh.t.T, Tf - 273.15, levels=24, cmap='inferno')
    ax.triplot(pts[0], pts[1], msh.t.T, color='white', alpha=0.12, linewidth=0.3)
    cb = fig.colorbar(tcf, ax=ax, label='Temperature (°C)')
    ax.set_title(f"{title}\nPeak $T_{{int}}$ = {rinfo['T_interface_C']} °C ({rinfo['T_interface_K']} K), $\\sigma_n$ = {rinfo['sigma_n_MPa']} MPa", fontsize=9.5, fontweight='bold')
    ax.set_xlabel('Distance from Cutting Edge along Rake X (mm)')
    ax.set_ylabel('Insert Thickness Y (mm)')

plt.tight_layout()
plt.savefig('figures/fig1_fea_thermal_field.png', dpi=300)
plt.close()
print("\nUpdated publication figure: figures/fig1_fea_thermal_field.png (300 DPI)")
