"""
BANC D'ESSAI DES BASES DE FONCTIONS  --  sans aucun entrainement.

Le rapport a etabli que le verrou n'est pas la capacite du reseau mais le
CONDITIONNEMENT de la base : les features tanh representent le mode a 0,40 %
mais exigent ||theta|| = 9,3e8, pour un conditionnement de 3,55e16 et un rang
effectif de 57 sur 97.

Ce script mesure, pour plusieurs familles de fonctions et a initialisation
aleatoire (donc en quelques secondes, sans entrainer) :

  * l'erreur de PROJECTION de la solution FDM sur la base  -> capacite
  * le conditionnement de la base                          -> kappa(Phi)
  * le rang effectif
  * la norme ||theta|| requise
  * et surtout le conditionnement de la MATRICE DE PHYSIQUE A = lap + k^2,
    qui est celle qu'on resout reellement et qui est bien pire.

Si une base fait chuter kappa de 1e16 a 1e7, c'est acquis avant meme de
regarder un L2. C'est le seul test qui separe "capacite" de "conditionnement".

Env : BB_HID (96)  BB_F (209.84)  BB_SEED (0)
"""
import os
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
torch.set_default_dtype(torch.float64)
SEED = int(os.environ.get("BB_SEED", 0))
HID = int(os.environ.get("BB_HID", 96))
FREQ = float(os.environ.get("BB_F", 209.84))

C, RHO = 343.0, 1.204
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV
SRC_Z, SRC_W, SRC_A = L_NECK + 0.5*H_CAV, 0.01, 1.0e4
K2 = (2*np.pi*FREQ/C)**2

# --- reference FDM (reprise de fdm_sweep.py) ---
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def fdm(freq, h=1e-3):
    w = 2*np.pi*freq; k2 = (w/C)**2
    ka = w/C*R_NECK
    Zr = RHO*C*(0.5*ka**2 + 1j*(8/(3*np.pi))*ka); al = 1j*w*RHO/Zr
    Nr = int(round(R_CAV/h))+1; Nz = int(round(Z_TOP/h))+1
    r = np.arange(Nr)*h; z = np.arange(Nz)*h
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    fl = ((ZZ < L_NECK-1e-12) & (RR <= R_NECK+1e-12)) | ((ZZ >= L_NECK-1e-12) & (RR <= R_CAV+1e-12))
    mo = fl & (np.abs(ZZ) < 1e-12) & (RR <= R_NECK+1e-12)
    idx = lambda i, j: i + j*Nr
    N = Nr*Nz; ih2 = 1.0/h**2
    F = SRC_A*np.exp(-(RR**2 + (ZZ-SRC_Z)**2)/(2*SRC_W**2))
    rows, cols, dat = [], [], []; b = np.zeros(N, complex)
    for i in range(Nr):
        for j in range(Nz):
            ci = idx(i, j)
            if not fl[i, j]:
                rows += [ci]; cols += [ci]; dat += [1.0]; continue
            diag = k2 + 0j
            if i == 0:
                diag += -4*ih2; rows += [ci]; cols += [idx(1, j)]; dat += [4*ih2]
            else:
                lc = ih2*(1-0.5/i); rc = ih2*(1+0.5/i); diag += -2*ih2
                if fl[i-1, j]: rows += [ci]; cols += [idx(i-1, j)]; dat += [lc]
                else:          diag += lc
                if i+1 < Nr and fl[i+1, j]: rows += [ci]; cols += [idx(i+1, j)]; dat += [rc]
                else:                        diag += rc
            if mo[i, j]:
                diag += -2*ih2 - 2*al/h
                rows += [ci]; cols += [idx(i, j+1)]; dat += [2*ih2]
            else:
                diag += -2*ih2
                for jj in (j-1, j+1):
                    if 0 <= jj < Nz and fl[i, jj]: rows += [ci]; cols += [idx(i, jj)]; dat += [ih2]
                    else:                          diag += ih2
            rows += [ci]; cols += [ci]; dat += [diag]
            b[ci] = -F[i, j]
    A = sp.coo_matrix((dat, (rows, cols)), shape=(N, N), dtype=complex).tocsr()
    return r, z, fl, spsolve(A, b).reshape((Nz, Nr)).T


r_ref, z_ref, FL, P_REF = fdm(FREQ)
ii, jj = np.where(FL)
RG, ZG = r_ref[ii], z_ref[jj]
PREF = P_REF[FL]
print(f"reference FDM : {FL.sum()} noeuds, max|P| = {np.abs(PREF).max():.1f} Pa, f = {FREQ} Hz")
print(f"{HID} fonctions de base, initialisation aleatoire, AUCUN entrainement\n")

X = torch.tensor(np.stack([RG/R_CAV, ZG/Z_TOP], 1))          # entrees normalisees
XR = torch.tensor(np.stack([RG, ZG], 1))                     # coordonnees physiques


# ==========================================================================
# les familles de bases
# ==========================================================================
def mlp(act, layers=5, w0=1.0, siren=False):
    """Renvoie phi(x) : (N,2) -> (N,HID), a poids aleatoires."""
    torch.manual_seed(SEED)
    mods, d = [], 2
    for i in range(layers):
        lin = nn.Linear(d, HID)
        if siren:
            with torch.no_grad():
                b = (1.0/d) if i == 0 else (np.sqrt(6.0/d)/w0)
                lin.weight.uniform_(-b, b); lin.bias.uniform_(-b, b)
        mods += [lin, act()]
        d = HID
    net = nn.Sequential(*mods)
    return lambda x: net(x)


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__(); self.w0 = w0
    def forward(self, x): return torch.sin(self.w0*x)


def gabor_basis(nb, sig_lo, sig_hi, kmax):
    """Fonctions de Gabor : gaussienne localisee x oscillation."""
    rng = np.random.default_rng(SEED)
    cr = rng.uniform(0, R_CAV, nb); cz = rng.uniform(0, Z_TOP, nb)
    sg = rng.uniform(sig_lo, sig_hi, nb)
    kr = rng.normal(0, kmax, nb); kz = rng.normal(0, kmax, nb)
    ph = rng.uniform(0, 2*np.pi, nb)
    cr, cz, sg = map(lambda a: torch.tensor(a), (cr, cz, sg))
    kr, kz, ph = map(lambda a: torch.tensor(a), (kr, kz, ph))
    def f(xr):
        d2 = (xr[:, 0:1] - cr)**2 + (xr[:, 1:2] - cz)**2
        env = torch.exp(-d2/(2*sg**2))
        osc = torch.cos(kr*xr[:, 0:1] + kz*xr[:, 1:2] + ph)
        return env*osc
    return f


def rbf_basis(nb, sig_lo, sig_hi):
    rng = np.random.default_rng(SEED)
    cr = torch.tensor(rng.uniform(0, R_CAV, nb)); cz = torch.tensor(rng.uniform(0, Z_TOP, nb))
    sg = torch.tensor(rng.uniform(sig_lo, sig_hi, nb))
    def f(xr):
        d2 = (xr[:, 0:1] - cr)**2 + (xr[:, 1:2] - cz)**2
        return torch.exp(-d2/(2*sg**2))
    return f



def local_basis(nb, w0=8.0):
    """DECOMPOSITION DE DOMAINE : un petit reseau par zone, fondu par une
    partition de l'unite. Les zones se recouvrent, donc aucune contrainte
    d'interface n'est necessaire -- c'est l'approche FBPINN."""
    torch.manual_seed(SEED)
    zc = [0.5*L_NECK, L_NECK, L_NECK + 0.55*H_CAV]     # col, jonction, cavite
    zs = [0.9*L_NECK, 0.5*L_NECK, 0.75*H_CAV]          # largeurs, avec recouvrement
    per = nb//3
    nets = []
    for _ in range(3):
        mods, d = [], 2
        for i in range(3):
            lin = nn.Linear(d, per)
            with torch.no_grad():
                b = (1.0/d) if i == 0 else (np.sqrt(6.0/d)/w0)
                lin.weight.uniform_(-b, b); lin.bias.uniform_(-b, b)
            mods += [lin, Sine(w0)]; d = per
        nets.append(nn.Sequential(*mods))
    def f(x):
        z = x[:, 1:2]*Z_TOP                             # entree normalisee -> z physique
        ws = [torch.exp(-0.5*((z - c)/sg)**2) for c, sg in zip(zc, zs)]
        tot = sum(ws)
        return torch.cat([(w/tot)*n(x) for w, n in zip(ws, nets)], 1)
    return f


def trefftz_basis(nb):
    """TREFFTZ : chaque fonction satisfait lap(phi) + k^2 phi = 0 EXACTEMENT.

    En axisymetrique, separation phi = J0(kappa r) Z(z) avec Z'' = (kappa^2-k^2)Z.
    On prend kappa_m = j'_(0,m)/R_CAV, zeros de J0', ce qui satisfait AUSSI la
    paroi laterale de la cavite par construction. Le residu d'EDP ne peut donc
    venir que de la source et des autres bords -- jamais de l'approximation.
    """
    from scipy.special import jn_zeros, j0
    k = np.sqrt(K2)
    kap = np.concatenate([[0.0], jn_zeros(1, nb//2)/R_CAV])   # j'_(0,m) = zeros de J1
    fns = []
    for km in kap:
        d = km**2 - K2
        if d < 0:
            b = np.sqrt(-d); zf = [lambda z, b=b: np.cos(b*z), lambda z, b=b: np.sin(b*z)]
        else:
            b = np.sqrt(d)
            zf = [lambda z, b=b: np.cosh(b*np.clip(z, 0, Z_TOP))/np.cosh(b*Z_TOP),
                  lambda z, b=b: np.sinh(b*np.clip(z, 0, Z_TOP))/np.cosh(b*Z_TOP)]
        for zz in zf:
            fns.append((km, zz))
        if len(fns) >= nb: break
    fns = fns[:nb]
    def f(xr):
        r = xr[:, 0].detach().numpy(); z = xr[:, 1].detach().numpy()
        cols = [torch.tensor(j0(km*r)*zz(z)) for km, zz in fns]
        return torch.stack(cols, 1)
    return f


BASES = [
    ("tanh (reference)",     lambda: mlp(nn.Tanh),                               "norm"),
    ("sine w0=5",            lambda: mlp(lambda: Sine(5.), siren=True, w0=5.),   "norm"),
    ("sine w0=8",            lambda: mlp(lambda: Sine(8.), siren=True, w0=8.),   "norm"),
    ("Gabor etroit",         lambda: gabor_basis(HID, 0.005, 0.02, 60.0),        "phys"),
    ("DECOMPO 3 zones sine", lambda: local_basis(HID, 8.0),                      "norm"),
    ("TREFFTZ (Bessel)",     lambda: trefftz_basis(HID),                      "trefftz"),
]


def laplacian_of(fn, xr_np, mode):
    """lap(phi) en axisymetrique, par autodiff, pour les HID fonctions."""
    xr = torch.tensor(xr_np, requires_grad=True)
    inp = xr if mode == "phys" else torch.stack([xr[:, 0]/R_CAV, xr[:, 1]/Z_TOP], 1)
    out = fn(inp)
    L = torch.zeros_like(out)
    for j in range(out.shape[1]):
        g = torch.autograd.grad(out[:, j].sum(), xr, create_graph=True)[0]
        gr, gz = g[:, 0], g[:, 1]
        grr = torch.autograd.grad(gr.sum(), xr, create_graph=True)[0][:, 0]
        gzz = torch.autograd.grad(gz.sum(), xr, create_graph=True)[0][:, 1]
        r = xr[:, 0]
        inv = torch.where(r < 1e-6, torch.zeros_like(r), 1.0/torch.clamp(r, min=1e-6))
        L[:, j] = grr + torch.where(r < 1e-6, grr, gr*inv) + gzz
    return out.detach(), L.detach()


# sous-echantillonnage pour la matrice de physique (autodiff sur HID sorties)
sel = np.random.default_rng(0).choice(RG.size, size=min(900, RG.size), replace=False)
XSEL = np.stack([RG[sel], ZG[sel]], 1)
FSEL = SRC_A*np.exp(-(XSEL[:, 0]**2 + (XSEL[:, 1]-SRC_Z)**2)/(2*SRC_W**2))

print(f"{'base':22s} {'proj. de P':>11s} {'RESIDU de la proj.':>20s} "
      f"{'kappa(A_phys)':>14s} {'rang A':>8s}")
print("-"*80)
print("  (le residu de la projection est LE chiffre : il dit si la base represente")
print("   correctement lap(P), et non seulement P)")
print("-"*80)

for nom, mk, mode in BASES:
    fn = mk()
    with torch.no_grad():
        inp = X if mode == "norm" else XR
        Phi = fn(inp).numpy()
    # capacite : meilleure projection possible de la solution FDM
    thr, *_ = np.linalg.lstsq(Phi, PREF.real, rcond=None)
    thi, *_ = np.linalg.lstsq(Phi, PREF.imag, rcond=None)
    fit = Phi @ thr + 1j*(Phi @ thi)
    err = np.linalg.norm(fit - PREF)/np.linalg.norm(PREF)
    s = np.linalg.svd(Phi, compute_uv=False)
    kap = s[0]/max(s[-1], 1e-300)
    rk = int((s > s[0]*1e-12).sum())
    nth = np.linalg.norm(np.r_[thr, thi])
    # matrice de PHYSIQUE : c'est elle qu'on resout reellement
    if mode == "trefftz":
        # lap(phi) + k^2 phi = 0 par CONSTRUCTION : le laplacien est analytique,
        # pas besoin d'autodiff, et le residu homogene est nul a la precision
        # machine. Le prix : une base de Trefftz ne peut pas porter la SOURCE.
        # Il lui faut une solution particuliere -- et on l'a deja, c'est la
        # formule de Poisson validee a 4,6 % contre le FDM.
        with torch.no_grad():
            M = fn(torch.tensor(XSEL))
        L = -K2*M
    else:
        M, L = laplacian_of(fn, XSEL, mode)
    A = (L + K2*M).numpy()
    sa = np.linalg.svd(A, compute_uv=False)
    kapa = sa[0]/max(sa[-1], 1e-300)
    rka = int((sa > sa[0]*1e-12).sum())
    # LE test : le champ le mieux represente produit-il un residu acceptable ?
    thr2, *_ = np.linalg.lstsq(M.numpy(), PREF[sel].real, rcond=None)
    thi2, *_ = np.linalg.lstsq(M.numpy(), PREF[sel].imag, rcond=None)
    rr = A @ thr2 + FSEL
    ri = A @ thi2
    resid = np.sqrt(np.mean(rr**2 + ri**2))/np.sqrt(np.mean(FSEL**2))
    print(f"{nom:22s} {err*100:10.2f}% {resid*100:19.1f}% "
          f"{kapa:14.2e} {rka:5d}/{HID:<3d}")

print()
print()
print("Lecture :")
print("  proj. de P          : la base sait-elle representer le CHAMP ?  (petit = bon)")
print("  RESIDU de la proj.  : ce meme champ satisfait-il l'EQUATION ?   (petit = bon)")
print("                        100 % = aussi mauvais que le champ nul.")
print("  kappa(A_phys)       : conditionnement du systeme reellement resolu.")
