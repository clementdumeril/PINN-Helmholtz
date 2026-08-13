"""
Reference FDM transitoire axisymetrique (r,z) pour l'equation d'onde forcee
du resonateur de Helmholtz -- VERITE-TERRAIN du PINN time-marching.

Meme geometrie / source / murs Neumann que pinn_3d_transient.ipynb :
    ptt = c^2 ( prr + pr/r + pzz ) + F(r,z,t)
    dp/dn = 0 sur toutes les parois, IC de repos (p=pt=0).

Schema : leapfrog explicite, Laplacien axisymetrique en VOLUMES FINIS masques
(flux nul vers une cellule hors-domaine => Neumann homogene exact, gere le
decrochement col/cavite et l'axe r=0 sans cas particulier fragile).

Sortie :
    data/fdm_transient_reference.npz  (champ sous-echantillonne + enveloppe)
    plots/fdm_transient_reference.png
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)

# --- Parametres physiques (identiques au notebook) ---
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_MAX = L_NECK + H_CAV
C = 343.0
F_START, F_END = 50.0, 800.0
T_MAX = 0.025
SRC_ZC, SRC_W, SRC_S0 = 0.006, 0.005, 8.0e7

def forcing(rr, zz, t):
    spatial = np.exp(-(rr**2 + (zz - SRC_ZC)**2) / (2.0 * SRC_W**2))
    phase = 2.0*np.pi*(F_START*t + (F_END - F_START)/(2.0*T_MAX)*t*t)
    temporal = np.sin(phase) * np.exp(-((t - T_MAX/2)**2) / (2.0*(T_MAX/3)**2))
    return SRC_S0 * spatial * temporal

# --- Grille ---
H = 5.0e-4                       # 0.5 mm
Nr = int(round(R_CAV / H)) + 1   # 81
Nz = int(round(Z_MAX / H)) + 1   # 241
r = np.arange(Nr) * H
z = np.arange(Nz) * H
RR, ZZ = np.meshgrid(r, z, indexing="ij")   # (Nr, Nz)

# Masque du domaine (col fin puis cavite large)
dom = ((ZZ < L_NECK - 1e-12) & (RR <= R_NECK + 1e-12)) | \
      ((ZZ >= L_NECK - 1e-12) & (RR <= R_CAV + 1e-12))
print(f"Grille {Nr}x{Nz}, h={H*1e3:.2f} mm, cellules actives = {dom.sum()}")

# Voisins dans le domaine (pour flux Neumann masque)
in_rp = np.zeros_like(dom); in_rp[:-1, :] = dom[1:, :] & dom[:-1, :]   # face vers i+1
in_rm = np.zeros_like(dom); in_rm[1:, :]  = dom[:-1, :] & dom[1:, :]   # face vers i-1
in_zp = np.zeros_like(dom); in_zp[:, :-1] = dom[:, 1:] & dom[:, :-1]   # face vers j+1
in_zm = np.zeros_like(dom); in_zm[:, 1:]  = dom[:, :-1] & dom[:, 1:]   # face vers j-1

# Rayons aux faces radiales (0 si flux coupe)
rp_face = np.where(in_rp, (np.arange(Nr)[:, None] + 0.5) * H, 0.0)
rm_face = np.where(in_rm, (np.arange(Nr)[:, None] - 0.5) * H, 0.0)
r_col = np.where(r > 0, r, 1.0)[:, None]     # evite /0 ; l'axe est traite a part

def laplacian(p):
    lap = np.zeros_like(p)
    # terme radial i>0 : (1/r) d/dr(r dp/dr) en volumes finis
    pr_p = np.zeros_like(p); pr_p[:-1, :] = p[1:, :] - p[:-1, :]
    pr_m = np.zeros_like(p); pr_m[1:, :]  = p[1:, :] - p[:-1, :]
    rad = (rp_face * pr_p - rm_face * pr_m) / (r_col * H * H)
    # axe r=0 : 2 d2p/dr2 = 4(p1-p0)/h^2 (symetrie), si (1,j) dans le domaine
    axis = np.zeros_like(p)
    axis[0, :] = np.where(in_rp[0, :], 4.0 * (p[1, :] - p[0, :]) / (H*H), 0.0)
    rad[0, :] = axis[0, :]
    # terme axial : flux masque
    az_p = np.where(in_zp, np.roll(p, -1, axis=1) - p, 0.0)
    az_m = np.where(in_zm, p - np.roll(p, 1, axis=1), 0.0)
    ax = (az_p - az_m) / (H * H)
    return (rad + ax) * dom

# --- Pas de temps (CFL) ---
CFL = 0.40
dt = CFL * H / (C * np.sqrt(2.0))
Nt = int(np.ceil(T_MAX / dt))
dt = T_MAX / Nt
print(f"dt = {dt*1e6:.3f} us, Nt = {Nt} pas")

# --- Leapfrog ---
p_old = np.zeros((Nr, Nz))
p_cur = np.zeros((Nr, Nz))
c2dt2 = (C * dt) ** 2

# points de sonde : centre col (r=0,z=SRC_ZC) et fond cavite (r=0,z=Z_MAX)
j_neck = int(round(SRC_ZC / H)); j_cav = Nz - 1
probe_neck, probe_cav, t_hist = [], [], []
env_max = np.zeros(Nt + 1)      # enveloppe max|p| domaine
snap_times = np.linspace(0, T_MAX, 60)
snaps, snap_t = [], []
si = 0

t0 = time.time()
for n in range(Nt):
    tn = n * dt
    F = forcing(RR, ZZ, tn) * dom
    p_new = 2.0 * p_cur - p_old + c2dt2 * laplacian(p_cur) + (dt*dt) * F
    p_new *= dom
    p_old, p_cur = p_cur, p_new
    env_max[n + 1] = np.abs(p_cur[dom]).max()
    probe_neck.append(p_cur[0, j_neck]); probe_cav.append(p_cur[0, j_cav])
    t_hist.append(tn + dt)
    if si < len(snap_times) and (tn + dt) >= snap_times[si]:
        snaps.append(p_cur.copy()); snap_t.append(tn + dt); si += 1
print(f"FDM termine en {time.time()-t0:.1f} s")

t_hist = np.array(t_hist)
probe_neck = np.array(probe_neck); probe_cav = np.array(probe_cav)
p_final_env = np.abs(np.array(snaps)).max()

print("\n=== DIAGNOSTIC FDM ===")
print(f"max|p| global (domaine, tout t) : {env_max.max():.3f} Pa")
print(f"max|p| sonde col                : {np.abs(probe_neck).max():.3f} Pa")
print(f"max|p| sonde cavite             : {np.abs(probe_cav).max():.3f} Pa")
i_peak = np.argmax(np.abs(probe_cav))
f_at_peak = F_START + (F_END - F_START) / T_MAX * t_hist[i_peak]
print(f"pic cavite a t={t_hist[i_peak]*1e3:.2f} ms  (f_chirp={f_at_peak:.0f} Hz)")

os.makedirs("data", exist_ok=True); os.makedirs("plots", exist_ok=True)
np.savez_compressed("data/fdm_transient_reference.npz",
    r=r, z=z, dom=dom, t=t_hist, env_max=env_max[1:],
    probe_neck=probe_neck, probe_cav=probe_cav,
    snaps=np.array(snaps), snap_t=np.array(snap_t),
    H=H, dt=dt, T_MAX=T_MAX, SRC_S0=SRC_S0)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(t_hist*1e3, probe_cav, lw=0.6, label="fond cavite (r=0,z=Zmax)")
ax[0].plot(t_hist*1e3, probe_neck, lw=0.6, alpha=0.7, label="col (r=0)")
ax[0].set_xlabel("t (ms)"); ax[0].set_ylabel("p (Pa)")
ax[0].set_title("Signaux sondes"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(t_hist*1e3, env_max[1:], "C3", lw=1)
ax[1].set_xlabel("t (ms)"); ax[1].set_ylabel("max|p| (Pa)")
ax[1].set_title("Enveloppe d'amplitude (domaine)"); ax[1].grid(alpha=0.3)
km = np.argmin(np.abs(np.array(snap_t) - t_hist[i_peak]))
pm = ax[2].pcolormesh(z*1e3, r*1e3, np.where(dom, snaps[km], np.nan),
                      cmap="RdBu_r", shading="auto")
ax[2].set_xlabel("z (mm)"); ax[2].set_ylabel("r (mm)")
ax[2].set_title(f"Champ a t={snap_t[km]*1e3:.1f} ms"); ax[2].set_aspect("equal")
fig.colorbar(pm, ax=ax[2], label="p (Pa)")
fig.tight_layout(); fig.savefig("plots/fdm_transient_reference.png", dpi=110)
print("\nFigure  : plots/fdm_transient_reference.png")
print("Donnees : data/fdm_transient_reference.npz")
