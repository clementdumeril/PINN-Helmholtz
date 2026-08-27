"""
BANC D'ESSAI DES BASES  --  cas TRANSITOIRE, col ouvert + exterieur maille.

Pendant que bench_bases.py teste le cas harmonique sur col + cavite, celui-ci
teste le VRAI domaine du transitoire : exterieur maille compris, soit
0,12 x 0,22 m au lieu de 0,04 x 0,12.

Pourquoi c'est mesurable sans derivees temporelles : a temps long, apres le
passage de l'impulsion, le champ transitoire EST harmonique. L'ansatz du PINN
transitoire l'ecrit d'ailleurs explicitement,

    p = g(t) [ A(x) cos(w0 t) + B(x) sin(w0 t) + C(x) ],

et en injectant cela dans  p_tt + sig p_t - c^2 lap(p) = 0  (F = 0 a temps
long, sig = 0 hors eponge), on obtient terme a terme

    lap(A) + k0^2 A = 0        et        lap(B) + k0^2 B = 0.

L'operateur transitoire se REDUIT donc a Helmholtz sur les enveloppes. Un
instantane tardif est une combinaison lineaire de A et B : le projeter et
mesurer le residu de lap + k0^2 mesure exactement la meme difficulte.

Normalisation : a temps long l'equation est HOMOGENE, donc pas de source pour
normaliser. On rapporte le residu a ||k0^2 P||, c'est-a-dire a la taille des
termes qui doivent s'annuler entre eux. Un residu de 100 % signifie que la
quasi-annulation ne se produit pas du tout.

Env : BT_SNAP (indice d'instantane, -1 = le dernier)  BT_HID (96)  BT_SEED (0)
"""
import os
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
torch.set_default_dtype(torch.float64)
SEED = int(os.environ.get("BT_SEED", 0))
HID = int(os.environ.get("BT_HID", 96))
ISNAP = int(os.environ.get("BT_SNAP", -1))

C = 343.0
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV
R_EXT, Z_EXT = 0.12, 0.10
Z_SPAN = Z_TOP + Z_EXT

d = np.load("data/open_resonator.npz")
F0 = float(d["f0_meas"]); K0 = 2*np.pi*F0/C; K02 = K0**2
r_g, z_g, DOM = d["r"], d["z"], d["dom"].astype(bool)
SNAP = d["snaps"][ISNAP]; TSN = float(d["snap_t"][ISNAP])

ii, jj = np.where(DOM)
RG, ZG = r_g[ii], z_g[jj]
PREF = SNAP[DOM]
print(f"instantane a t = {TSN*1e3:.1f} ms (bien apres l'impulsion), f0 = {F0:.2f} Hz")
print(f"domaine transitoire : {DOM.sum()} noeuds, r dans [0, {R_EXT}], z dans [{-Z_EXT}, {Z_TOP}]")
print(f"max|p| = {np.abs(PREF).max():.3e} Pa")
print(f"{HID} fonctions de base, initialisation aleatoire, AUCUN entrainement\n")

# entrees normalisees sur le domaine transitoire
X = torch.tensor(np.stack([RG/R_EXT, (ZG + Z_EXT)/Z_SPAN], 1))
XR = torch.tensor(np.stack([RG, ZG], 1))


class Sine(nn.Module):
    def __init__(self, w0): super().__init__(); self.w0 = w0
    def forward(self, x): return torch.sin(self.w0*x)


def mlp(act, layers=5, w0=1.0, siren=False):
    torch.manual_seed(SEED)
    mods, dd = [], 2
    for i in range(layers):
        lin = nn.Linear(dd, HID)
        if siren:
            with torch.no_grad():
                b = (1.0/dd) if i == 0 else (np.sqrt(6.0/dd)/w0)
                lin.weight.uniform_(-b, b); lin.bias.uniform_(-b, b)
        mods += [lin, act()]
        dd = HID
    net = nn.Sequential(*mods)
    return lambda x: net(x)


def gabor_basis(nb, s_lo, s_hi, kmax):
    rng = np.random.default_rng(SEED)
    cr = torch.tensor(rng.uniform(0, R_EXT, nb)); cz = torch.tensor(rng.uniform(-Z_EXT, Z_TOP, nb))
    sg = torch.tensor(rng.uniform(s_lo, s_hi, nb))
    kr = torch.tensor(rng.normal(0, kmax, nb)); kz = torch.tensor(rng.normal(0, kmax, nb))
    ph = torch.tensor(rng.uniform(0, 2*np.pi, nb))
    def f(xr):
        d2 = (xr[:, 0:1] - cr)**2 + (xr[:, 1:2] - cz)**2
        return torch.exp(-d2/(2*sg**2))*torch.cos(kr*xr[:, 0:1] + kz*xr[:, 1:2] + ph)
    return f


def local_basis(nb, nz_zones, w0=8.0):
    """DECOMPOSITION : un petit reseau par zone, fondu par partition de l'unite.

    Quatre zones alignees sur la geometrie : exterieur, bouche/col, jonction,
    cavite. Les fenetres se recouvrent, donc aucune contrainte d'interface.
    """
    torch.manual_seed(SEED)
    zc, zs = nz_zones
    per = nb//len(zc)
    nets = []
    for _ in range(len(zc)):
        mods, dd = [], 2
        for i in range(3):
            lin = nn.Linear(dd, per)
            with torch.no_grad():
                b = (1.0/dd) if i == 0 else (np.sqrt(6.0/dd)/w0)
                lin.weight.uniform_(-b, b); lin.bias.uniform_(-b, b)
            mods += [lin, Sine(w0)]
            dd = per
        nets.append(nn.Sequential(*mods))
    def f(x):
        z = x[:, 1:2]*Z_SPAN - Z_EXT
        ws = [torch.exp(-0.5*((z - c)/s)**2) for c, s in zip(zc, zs)]
        tot = sum(ws)
        return torch.cat([(w/tot)*n(x) for w, n in zip(ws, nets)], 1)
    return f


ZONES = ([-0.05, 0.0, L_NECK, L_NECK + 0.55*H_CAV],      # exterieur, bouche, jonction, cavite
         [0.055, 0.030, 0.030, 0.050])

BASES = [
    ("tanh (reference)",     lambda: mlp(nn.Tanh),                              "norm"),
    ("sine w0=5",            lambda: mlp(lambda: Sine(5.), siren=True, w0=5.),  "norm"),
    ("sine w0=8",            lambda: mlp(lambda: Sine(8.), siren=True, w0=8.),  "norm"),
    ("Gabor etroit",         lambda: gabor_basis(HID, 0.01, 0.04, 40.0),        "phys"),
    ("DECOMPO 4 zones sine", lambda: local_basis(HID, ZONES, 8.0),              "norm"),
]


def lap_of(fn, xr_np, mode):
    xr = torch.tensor(xr_np, requires_grad=True)
    inp = xr if mode == "phys" else torch.stack(
        [xr[:, 0]/R_EXT, (xr[:, 1] + Z_EXT)/Z_SPAN], 1)
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


sel = np.random.default_rng(0).choice(RG.size, size=min(900, RG.size), replace=False)
XSEL = np.stack([RG[sel], ZG[sel]], 1)
PSEL = PREF[sel]

print(f"{'base':22s} {'proj. de p':>11s} {'RESIDU de la proj.':>20s} "
      f"{'kappa(A)':>12s} {'rang A':>8s}")
print("-"*78)

for nom, mk, mode in BASES:
    fn = mk()
    with torch.no_grad():
        Phi = fn(X if mode == "norm" else XR).numpy()
    th, *_ = np.linalg.lstsq(Phi, PREF, rcond=None)
    err = np.linalg.norm(Phi@th - PREF)/np.linalg.norm(PREF)

    M, L = lap_of(fn, XSEL, mode)
    A = (L + K02*M).numpy()
    sa = np.linalg.svd(A, compute_uv=False)
    kap = sa[0]/max(sa[-1], 1e-300); rk = int((sa > sa[0]*1e-12).sum())

    th2, *_ = np.linalg.lstsq(M.numpy(), PSEL, rcond=None)
    res = A @ th2
    ref = K02*np.linalg.norm(M.numpy() @ th2)     # taille du terme k0^2 P
    rel = np.linalg.norm(res)/max(ref, 1e-300)
    print(f"{nom:22s} {err*100:10.2f}% {rel*100:19.1f}% {kap:12.2e} {rk:5d}/{HID:<3d}")

print()
print("Lecture :")
print("  proj. de p         : la base represente-t-elle le CHAMP ?    (petit = bon)")
print("  RESIDU de la proj. : ce champ satisfait-il lap(p) + k0^2 p = 0 ?")
print("                       rapporte a ||k0^2 p||, la taille des termes qui")
print("                       doivent s'annuler. 100 % = pas d'annulation du tout.")
