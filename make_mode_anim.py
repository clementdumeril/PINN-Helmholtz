"""
Animation du MODE RESONANT du resonateur de Helmholtz a col ouvert.

Resout l'equation de Helmholtz en regime harmonique a f0, avec condition
d'impedance de rayonnement a la bouche (piston bafle), puis anime

        p(r,z,t) = Re{ P(r,z) * exp(i*w*t) }

sur plusieurs periodes. On voit la cavite se pressuriser et se depressuriser
pendant que l'air fait des allers-retours dans le col.

Le panneau de droite montre la signature meme d'un resonateur masse-ressort :
la pression de cavite et la vitesse au col sont en QUADRATURE (dephasees de 90
degres). La cavite joue le ressort, le bouchon d'air du col joue la masse.

Sortie : plots/helmholtz_mode.gif, data/helmholtz_mode.npz
Env : MA_F (frequence, 209.84) MA_H (pas mm, 0.5) MA_FRAMES (72) MA_PERIODES (2)
"""
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)

C, RHO = 343.0, 1.204
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV
FREQ = float(os.environ.get("MA_F", 209.84))
H = float(os.environ.get("MA_H", 0.5))*1e-3
NFR = int(os.environ.get("MA_FRAMES", 72))
NPER = float(os.environ.get("MA_PERIODES", 2))
W = 2*np.pi*FREQ; K2 = (W/C)**2
SRC_Z, SRC_W, SRC_A = L_NECK + 0.5*H_CAV, 0.01, 1.0e4
ka = W/C*R_NECK
Zr = RHO*C*(0.5*ka**2 + 1j*(8/(3*np.pi))*ka)
ALPHA = 1j*W*RHO/Zr


def solve(h):
    Nr = int(round(R_CAV/h))+1; Nz = int(round(Z_TOP/h))+1
    r = np.arange(Nr)*h; z = np.arange(Nz)*h
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    fl = ((ZZ < L_NECK-1e-12) & (RR <= R_NECK+1e-12)) | ((ZZ >= L_NECK-1e-12) & (RR <= R_CAV+1e-12))
    mo = fl & (np.abs(ZZ) < 1e-12) & (RR <= R_NECK+1e-12)
    idx = lambda i, j: i + j*Nr
    N = Nr*Nz; ih2 = 1.0/h**2
    rows, cols, dat = [], [], []; b = np.zeros(N, complex)
    F = SRC_A*np.exp(-(RR**2 + (ZZ-SRC_Z)**2)/(2*SRC_W**2))
    for i in range(Nr):
        for j in range(Nz):
            ci = idx(i, j)
            if not fl[i, j]:
                rows += [ci]; cols += [ci]; dat += [1.0]; continue
            diag = K2 + 0j
            if i == 0:
                diag += -4*ih2; rows += [ci]; cols += [idx(1, j)]; dat += [4*ih2]
            else:
                lc = ih2*(1-0.5/i); rc = ih2*(1+0.5/i); diag += -2*ih2
                if fl[i-1, j]: rows += [ci]; cols += [idx(i-1, j)]; dat += [lc]
                else:          diag += lc
                if i+1 < Nr and fl[i+1, j]: rows += [ci]; cols += [idx(i+1, j)]; dat += [rc]
                else:                        diag += rc
            if mo[i, j]:
                diag += -2*ih2 - 2*ALPHA/h
                rows += [ci]; cols += [idx(i, j+1)]; dat += [2*ih2]
            else:
                diag += -2*ih2
                for jj in (j-1, j+1):
                    if 0 <= jj < Nz and fl[i, jj]:
                        rows += [ci]; cols += [idx(i, jj)]; dat += [ih2]
                    else:
                        diag += ih2
            rows += [ci]; cols += [ci]; dat += [diag]
            b[ci] = -F[i, j]
    A = sp.coo_matrix((dat, (rows, cols)), shape=(N, N), dtype=complex).tocsr()
    P = spsolve(A, b).reshape((Nz, Nr)).T
    return r, z, fl, np.where(fl, P, np.nan)


r, z, fl, P = solve(H)
# normalisation : amplitude de cavite = 1 Pa (systeme lineaire)
j_cav = int(round(0.08/H)); P = P/abs(P[0, j_cav])
print(f"f = {FREQ} Hz | grille {fl.shape} | {fl.sum()} noeuds")

# --- sondes ---
j_mouth = 0
P_cav = P[0, j_cav]
# vitesse axiale a la bouche : v = -(1/(i*w*rho)) dP/dz
dPdz_mouth = (P[0, 1] - P[0, 0])/H
V_mouth = -dPdz_mouth/(1j*W*RHO)
print(f"|P| cavite = {abs(P_cav):.3f} Pa | |v| bouche = {abs(V_mouth)*1e3:.3f} mm/s")
dphi = np.angle(V_mouth) - np.angle(P_cav)
dphi = (dphi + np.pi) % (2*np.pi) - np.pi
print(f"dephasage vitesse/pression = {np.degrees(dphi):+.1f} deg  (quadrature attendue)")

# --- miroir pour une coupe meridienne complete ---
r_full = np.concatenate([-r[::-1], r[1:]])
mir = lambda a: np.concatenate([a[::-1, :], a[1:, :]], axis=0)
fl_f = mir(fl); P_f = mir(P)

phases = np.linspace(0, 2*np.pi*NPER, NFR, endpoint=False)
vmax = np.nanmax(np.abs(P_f))*1.05

plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(11, 5.2))
gs = GridSpec(1, 2, width_ratios=[1.25, 1], wspace=0.28)
axF = fig.add_subplot(gs[0]); axS = fig.add_subplot(gs[1])

fld0 = np.where(fl_f, (P_f*np.exp(1j*phases[0])).real, np.nan)
qm = axF.pcolormesh(z*1e3, r_full*1e3, fld0, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
axF.set(xlabel="z (mm)", ylabel="r (mm)"); axF.set_aspect("equal")
fig.colorbar(qm, ax=axF, fraction=0.030, pad=0.02, label="p (Pa)")
for s in (1, -1):
    axF.plot([0, L_NECK*1e3], [s*R_NECK*1e3]*2, "k", lw=1.2)
    axF.plot([L_NECK*1e3]*2, [s*R_NECK*1e3, s*R_CAV*1e3], "k", lw=1.2)
    axF.plot([L_NECK*1e3, Z_TOP*1e3], [s*R_CAV*1e3]*2, "k", lw=1.2)
axF.plot([Z_TOP*1e3]*2, [-R_CAV*1e3, R_CAV*1e3], "k", lw=1.2)
axF.plot([0, 0], [R_NECK*1e3, R_CAV*1e3*1.15], "k", lw=2.5)
axF.plot([0, 0], [-R_NECK*1e3, -R_CAV*1e3*1.15], "k", lw=2.5)
axF.annotate("bouche\n(ouverte)", (2, 26), fontsize=8, color="0.35")
axF.annotate("cavité", (80, 30), fontsize=9, color="0.35")

ph = np.linspace(0, 2*np.pi*NPER, 600)
pc = (P_cav*np.exp(1j*ph)).real
vm = (V_mouth*np.exp(1j*ph)).real
vm_n = vm/np.abs(vm).max()
axS.plot(ph/(2*np.pi), pc, "C3", lw=1.6, label="pression en cavité")
axS.plot(ph/(2*np.pi), vm_n, "C0", lw=1.6, label="vitesse au col (normalisée)")
axS.axhline(0, color="0.7", lw=.8)
axS.set(xlabel="temps (en périodes)", ylabel="amplitude",
        title=f"quadrature : {np.degrees(dphi):+.0f}°")
axS.legend(fontsize=8, loc="upper right"); axS.grid(alpha=.3)
cur = axS.axvline(0, color="k", lw=1.3)
sup = fig.suptitle("", fontsize=12, y=0.97)

def update(k):
    fld = np.where(fl_f, (P_f*np.exp(1j*phases[k])).real, np.nan)
    qm.set_array(fld.ravel())
    cur.set_xdata([phases[k]/(2*np.pi)]*2)
    sup.set_text(f"Mode résonant du résonateur de Helmholtz — f₀ = {FREQ:.1f} Hz   "
                 f"(t = {phases[k]/(2*np.pi):.2f} période)")
    return qm, cur, sup

anim = FuncAnimation(fig, update, frames=NFR, blit=False)
out = "plots/helmholtz_mode.gif"
anim.save(out, writer=PillowWriter(fps=18))
from PIL import Image, ImageSequence
im = Image.open(out)
fr = [f.convert("RGB").quantize(colors=96, method=Image.MEDIANCUT) for f in ImageSequence.Iterator(im)]
fr[0].save(out, save_all=True, append_images=fr[1:], duration=55, loop=0, optimize=True)
np.savez_compressed("data/helmholtz_mode.npz", r=r, z=z, fluid=fl, P=P,
                    freq=FREQ, P_cav=P_cav, V_mouth=V_mouth, dphi=dphi)
print(f"Anime : {out} ({NFR} images, {os.path.getsize(out)/1e6:.1f} Mo)")
