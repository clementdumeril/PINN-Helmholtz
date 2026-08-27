"""
Balayage frequentiel complet du resonateur de Helmholtz a col ouvert.

Resout l'equation de Helmholtz en regime harmonique pour CHAQUE frequence de
1 a 1000 Hz, avec condition d'impedance de rayonnement a la bouche (piston
bafle), et trace la reponse en amplitude. Le pic est la resonance.

C'est la REFERENCE du PINN parametre par la frequence : reseau (r,z,f) -> P.

A chaque frequence la matrice CHANGE : k^2 = (2 pi f / c)^2 vit sur la
diagonale, et le coefficient d'impedance alpha = i w rho / Z_r depend lui
aussi de omega, donc les lignes de la bouche changent aussi. Il n'y a pas de
raccourci : c'est une resolution lineaire complete par frequence (~0,1 s).

Sortie : data/fdm_sweep.npz, plots/fdm_sweep.png
Env : SW_FMIN (1) SW_FMAX (1000) SW_DF (1.0) SW_H (mm, 1.0) SW_TAG ("")
"""
import os, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)

C, RHO = 343.0, 1.204
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV
SRC_Z, SRC_W, SRC_A = L_NECK + 0.5*H_CAV, 0.01, 1.0e4

FMIN = float(os.environ.get("SW_FMIN", 1.0))
FMAX = float(os.environ.get("SW_FMAX", 1000.0))
DF   = float(os.environ.get("SW_DF", 1.0))
H    = float(os.environ.get("SW_H", 1.0))*1e-3
TAG  = os.environ.get("SW_TAG", "")

# ---- geometrie et masque : constants, calcules une seule fois ----
Nr = int(round(R_CAV/H)) + 1
Nz = int(round(Z_TOP/H)) + 1
r = np.arange(Nr)*H; z = np.arange(Nz)*H
RR, ZZ = np.meshgrid(r, z, indexing="ij")
FLUID = ((ZZ < L_NECK-1e-12) & (RR <= R_NECK+1e-12)) | ((ZZ >= L_NECK-1e-12) & (RR <= R_CAV+1e-12))
MOUTH = FLUID & (np.abs(ZZ) < 1e-12) & (RR <= R_NECK+1e-12)
FORC = SRC_A*np.exp(-(RR**2 + (ZZ-SRC_Z)**2)/(2*SRC_W**2))
N = Nr*Nz
J_CAV = int(round(0.08/H))          # sonde : fond de cavite, sur l'axe

print(f"grille {Nr}x{Nz} = {N} noeuds, {FLUID.sum()} fluides | h = {H*1e3:.2f} mm")
print(f"balayage {FMIN:.0f} -> {FMAX:.0f} Hz au pas de {DF:g} Hz "
      f"= {int(round((FMAX-FMIN)/DF))+1} resolutions")


def solve(freq):
    """Une resolution complete a la frequence donnee -> champ complexe P."""
    w = 2*np.pi*freq; k2 = (w/C)**2
    ka = w/C*R_NECK
    Zr = RHO*C*(0.5*ka**2 + 1j*(8/(3*np.pi))*ka)     # piston bafle
    alpha = 1j*w*RHO/Zr
    idx = lambda i, j: i + j*Nr
    ih2 = 1.0/H**2
    rows, cols, dat = [], [], []
    b = np.zeros(N, complex)
    for i in range(Nr):
        for j in range(Nz):
            ci = idx(i, j)
            if not FLUID[i, j]:
                rows += [ci]; cols += [ci]; dat += [1.0]; continue
            diag = k2 + 0j
            if i == 0:                                   # forme limite sur l'axe
                diag += -4*ih2; rows += [ci]; cols += [idx(1, j)]; dat += [4*ih2]
            else:
                lc = ih2*(1-0.5/i); rc = ih2*(1+0.5/i); diag += -2*ih2
                if FLUID[i-1, j]: rows += [ci]; cols += [idx(i-1, j)]; dat += [lc]
                else:             diag += lc             # noeud fantome replie
                if i+1 < Nr and FLUID[i+1, j]: rows += [ci]; cols += [idx(i+1, j)]; dat += [rc]
                else:                          diag += rc
            if MOUTH[i, j]:                              # condition de Robin
                diag += -2*ih2 - 2*alpha/H
                rows += [ci]; cols += [idx(i, j+1)]; dat += [2*ih2]
            else:
                diag += -2*ih2
                for jj in (j-1, j+1):
                    if 0 <= jj < Nz and FLUID[i, jj]:
                        rows += [ci]; cols += [idx(i, jj)]; dat += [ih2]
                    else:
                        diag += ih2
            rows += [ci]; cols += [ci]; dat += [diag]
            b[ci] = -FORC[i, j]
    A = sp.coo_matrix((dat, (rows, cols)), shape=(N, N), dtype=complex).tocsr()
    return spsolve(A, b).reshape((Nz, Nr)).T


freqs = np.arange(FMIN, FMAX + 0.5*DF, DF)
amp_cav = np.empty(freqs.size)          # |P| au fond de cavite
amp_max = np.empty(freqs.size)          # |P| max sur le domaine
phase   = np.empty(freqs.size)
t0 = time.time()
for n, f in enumerate(freqs):
    P = solve(f)
    amp_cav[n] = abs(P[0, J_CAV])
    amp_max[n] = np.abs(np.where(FLUID, P, 0)).max()
    phase[n] = np.angle(P[0, J_CAV])
    if (n+1) % 100 == 0 or n == freqs.size-1:
        print(f"  {n+1:4d}/{freqs.size}  f = {f:7.1f} Hz  |P|cav = {amp_cav[n]:10.2f} "
              f"| {time.time()-t0:5.0f}s")

# ---- pic de resonance : parabole sur les trois points du sommet ----
kpk = int(np.argmax(amp_cav))
if 0 < kpk < freqs.size-1:
    y0, y1, y2 = amp_cav[kpk-1], amp_cav[kpk], amp_cav[kpk+1]
    d = 0.5*(y0 - y2)/(y0 - 2*y1 + y2)
    f_pk = freqs[kpk] + d*DF
else:
    f_pk = freqs[kpk]
# largeur a -3 dB -> facteur de qualite
half = amp_cav[kpk]/np.sqrt(2.0)
lo = np.where(amp_cav[:kpk] < half)[0]
hi = np.where(amp_cav[kpk:] < half)[0]
if lo.size and hi.size:
    f_lo = np.interp(half, amp_cav[lo[-1]:kpk+1], freqs[lo[-1]:kpk+1])
    seg = slice(kpk, kpk+hi[0]+1)
    f_hi = np.interp(half, amp_cav[seg][::-1], freqs[seg][::-1])
    Q = f_pk/(f_hi - f_lo)
else:
    f_lo = f_hi = Q = np.nan

print(f"\n=== RESONANCE ===")
print(f"pic (parabole sur 3 points) : {f_pk:.2f} Hz")
print(f"amplitude au pic            : {amp_cav[kpk]:.1f} Pa")
print(f"bande a -3 dB               : {f_lo:.2f} - {f_hi:.2f} Hz")
print(f"facteur de qualite Q        : {Q:.1f}")
print(f"gain a la resonance         : {amp_cav[kpk]/amp_cav[0]:.1f}x l'amplitude a {FMIN:.0f} Hz")

np.savez_compressed(f"data/fdm_sweep{TAG}.npz", freqs=freqs, amp_cav=amp_cav,
                    amp_max=amp_max, phase=phase, f_pk=f_pk, Q=Q,
                    f_lo=f_lo, f_hi=f_hi, h=H, r=r, z=z, fluid=FLUID)

fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
ax[0].semilogy(freqs, amp_cav, "C0", lw=1.4, label="fond de cavité")
ax[0].semilogy(freqs, amp_max, "C1", lw=0.9, alpha=.6, label="max du domaine")
ax[0].axvline(f_pk, color="C3", ls="--", lw=1.1)
ax[0].annotate(f"f₀ = {f_pk:.1f} Hz\nQ = {Q:.0f}", (f_pk, amp_cav[kpk]),
               xytext=(f_pk+70, amp_cav[kpk]*0.7), color="C3", fontsize=10,
               arrowprops=dict(arrowstyle="->", color="C3", lw=1))
ax[0].set(ylabel="|P| (Pa)", title=f"Balayage fréquentiel — résonateur de Helmholtz à col ouvert "
                                  f"(h = {H*1e3:.1f} mm)")
ax[0].legend(); ax[0].grid(alpha=.3, which="both")
ax[1].plot(freqs, np.degrees(np.unwrap(phase)), "C2", lw=1.2)
ax[1].axvline(f_pk, color="C3", ls="--", lw=1.1)
ax[1].set(xlabel="fréquence (Hz)", ylabel="phase (°)")
ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"plots/fdm_sweep{TAG}.png", dpi=120)
print(f"Figure : plots/fdm_sweep{TAG}.png | Données : data/fdm_sweep{TAG}.npz")
