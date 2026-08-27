"""
PINN en CHAMP DIFFRACTE sur le resonateur a col ouvert.

    p(x,t)  =  p0(x,t)  +  dp(x,t)

p0 est le rayonnement libre de la source, connu ANALYTIQUEMENT ; le reseau
n'apprend que dp, ce que la geometrie fait de cette impulsion. C'est la
formulation d'Alkhalifah et coll., adaptee a une source gaussienne compacte.

POURQUOI CE CHANGEMENT (mesures des versions precedentes)

1. L'ansatz a enveloppes ne represente qu'une oscillation a w0. Or l'impulsion
   incidente est un Ricker LARGE BANDE. Resultat mesure : cavite 77-115 %,
   exterieur 370-600 %. Ici p0 porte l'impulsion, le reseau n'a plus a
   l'inventer -- et dp, lui, EST mono-frequence : ajuste sur le FDM, il donne
   209,20 Hz avec 4,01 % de residu. L'ansatz lui convient enfin.

2. Le terme source DISPARAIT de l'equation. Le probleme "0,1 % de l'espace-temps
   porte 100 % de l'information d'amplitude" n'existe plus.

3. Et surtout : dp = 0 NE SATISFAIT PLUS LES PAROIS. La condition devient
   d(dp)/dn = -d(p0)/dn, a second membre non nul et connu. Le champ diffracte
   est ENGENDRE par les parois, qui sont des courbes 1D bien echantillonnees,
   au lieu d'etre engendre par une source ponctuelle noyee dans le volume.
   C'est la correction structurelle du piege du zero : le minimum trivial
   n'est plus un minimum.

LE CHAMP INCIDENT

Pour une gaussienne compacte (sigma = 4 mm contre une longueur d'onde >= 0,68 m,
donc k*sigma <= 0,04), la solution en espace libre de p_tt - c^2 lap(p) = F est

    p0(R,t) = (S0/c^2) sigma^2 sqrt(pi/2) erf(u/sqrt2)/u * ricker(t - R/c)

avec u = R/sigma. Exacte en champ proche (ou le retard est negligeable) et en
champ lointain (ou erf -> 1 et l'on retrouve Q/4piR retarde).
Verifiee contre le FDM : 1,558e-4 predit contre 1,487e-4 mesure, soit 4,6 %,
qui est la contribution de l'image du baffle -- absente de p0 par construction.

Env : SC_TMS (60) SC_ITERS (3000) SC_CHECK (100) SC_M (16) SC_NCOL (800)
      SC_NLS (1200) SC_HID (128) SC_LR (1e-3) SC_F0 (209.8) SC_WWALL (5.0)
      SC_BETA (0.15) SC_RIDGE (1e-4) SC_SEED (0) SC_TAG ("")
"""
import os, time
import numpy as np
import torch, torch.nn as nn
from torch.func import jacfwd, vmap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
SEED = int(os.environ.get("SC_SEED", 0))
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cpu"; torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)      # jacfwd construit ses tangentes en double

# ---- geometrie et physique : celles de fdm_open_resonator.py ----
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV
C = 343.0
R_EXT, Z_EXT, L_SP = 0.12, 0.10, 0.03
SIG_MAX = 5.0 * C / L_SP
F_C, Z_SRC, W_SRC, S0 = 250.0, -0.055, 0.004, 8.0e6
T0_R = 1.3 / F_C

# f0 : ici la valeur du FDM a h=1mm, car c'est la reference a laquelle on se
# compare. La valeur A PRIORI de la formule de Helmholtz corrigee vaut
# 205,56 Hz ; l'ecart de 2 % accumule ~110 deg de phase sur 12,6 periodes, donc
# le choix n'est pas neutre. SC_F0 permet de mesurer ce cout.
F0 = float(os.environ.get("SC_F0", 209.8))
W0 = 2.0*np.pi*F0

T_WIN  = float(os.environ.get("SC_TMS", 60.0))*1e-3
ITERS  = int(os.environ.get("SC_ITERS", 3000))
CHECK  = int(os.environ.get("SC_CHECK", 100))
M_CH   = int(os.environ.get("SC_M", 16))
N_COL  = int(os.environ.get("SC_NCOL", 800))
N_LS   = int(os.environ.get("SC_NLS", 1200))
HID    = int(os.environ.get("SC_HID", 128))
LR     = float(os.environ.get("SC_LR", 1e-3))
W_WALL = float(os.environ.get("SC_WWALL", 5.0))
BETA   = float(os.environ.get("SC_BETA", 0.15))
RIDGE  = float(os.environ.get("SC_RIDGE", 1e-4))
TAG    = os.environ.get("SC_TAG", "")

SCALE_D = 3.0e-4                 # echelle du champ DIFFRACTE
TAU_G   = 0.8e-3
Z_SPAN  = Z_TOP + Z_EXT
FLOOR   = W0**2 * SCALE_D        # magnitude attendue de dp_tt et c^2 lap(dp)
N_PER   = max(8, N_COL // M_CH)
EDGES   = np.linspace(0.0, T_WIN, M_CH+1)
DT_CH   = T_WIN / M_CH
EPS_SCHED = [1.0, 3.0, 10.0, 30.0, 100.0]
DELTA   = 0.99

print(f"champ diffracte | fenetre {T_WIN*1e3:.0f} ms = {T_WIN*F0:.1f} periodes")
print(f"w0 = {W0:.1f} rad/s (f0 = {F0} Hz) | scale dp = {SCALE_D:.1e} Pa | floor = {FLOOR:.1f}")


# --------------------------------------------------------------------------
# champ incident analytique
# --------------------------------------------------------------------------
def ricker_t(t):
    a = (np.pi*F_C*(t - T0_R))**2
    return (1.0 - 2.0*a)*torch.exp(-a)


P0_K = (S0/C**2)*W_SRC**2        # 1.088e-3 Pa : valeur au centre de la source


def p0_of(r, z, t):
    """Rayonnement libre de la source gaussienne, temps retarde."""
    R = torch.sqrt(r**2 + (z - Z_SRC)**2)
    u = torch.clamp(R/W_SRC, min=1e-9)
    prof = np.sqrt(np.pi/2)*torch.erf(u/np.sqrt(2.0))/u
    return P0_K*prof*ricker_t(t - R/C)


def sigma_of(r, z):
    dz = torch.clamp((-Z_EXT + L_SP - z)/L_SP, 0.0, 1.0)
    dr = torch.clamp((r - (R_EXT - L_SP))/L_SP, 0.0, 1.0)
    return SIG_MAX*torch.maximum(dz, dr)**2 * (z < 0).double()


# --------------------------------------------------------------------------
# reseau : trois enveloppes LENTES pour le champ diffracte
# --------------------------------------------------------------------------
class EnvNet(nn.Module):
    def __init__(self, hidden=128, layers=5):
        super().__init__()
        net = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(layers-1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*net)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01); self.net[-1].bias.zero_()
    def forward(self, x):
        return self.net(x)


model = EnvNet(HID).to(DEV)
HEAD, OUT = model.net[:-1], model.net[-1]
D_TH = HID + 1


def Tn(a, g=False):
    return torch.tensor(a, dtype=torch.float64, device=DEV).reshape(-1, 1).requires_grad_(bool(g))


def _q_of_t(t):
    """Enveloppes temporelles g*cos, g*sin, g et leurs deux derivees."""
    e = torch.exp(-(t/TAU_G)**2)
    g, gp = 1.0 - e, 2.0*t/TAU_G**2*e
    gpp = (2.0/TAU_G**2 - 4.0*t**2/TAU_G**4)*e
    c, sn = torch.cos(W0*t), torch.sin(W0*t)
    return [(g*c,  gp*c - W0*g*sn,  gpp*c - 2*W0*gp*sn - W0**2*g*c),
            (g*sn, gp*sn + W0*g*c,  gpp*sn + 2*W0*gp*c  - W0**2*g*sn),
            (g,    gp,              gpp)]


def dp_of(r, z, t):
    x = torch.cat([r/R_EXT, (z + Z_EXT)/Z_SPAN, t/T_WIN], 1)
    o = model(x)
    qs = _q_of_t(t)
    return SCALE_D*(o[:, 0:1]*qs[0][0] + o[:, 1:2]*qs[1][0] + o[:, 2:3]*qs[2][0])


def p_of(r, z, t):
    return p0_of(r, z, t) + dp_of(r, z, t)


# --------------------------------------------------------------------------
# echantillonnage
# --------------------------------------------------------------------------
def sample_domain(n):
    out_r, out_z = [], []
    need = n
    while need > 0:
        m = int(need*1.8) + 128
        rr = np.random.uniform(0, R_EXT, m)
        zz = np.random.uniform(-Z_EXT, Z_TOP, m)
        keep = ((zz < 0) |
                ((zz >= 0) & (zz < L_NECK) & (rr <= R_NECK)) |
                ((zz >= L_NECK) & (rr <= R_CAV)))
        out_r.append(rr[keep]); out_z.append(zz[keep])
        need = n - sum(len(a) for a in out_r)
    return np.concatenate(out_r)[:n], np.concatenate(out_z)[:n]


def _t_chunks(n):
    return EDGES[:-1, None] + np.random.uniform(0.0, 1.0, (M_CH, n))*DT_CH


WALL_SEGS = [
    (lambda m: (np.full(m, R_NECK), np.random.uniform(0, L_NECK, m)), 1., 0.),
    (lambda m: (np.random.uniform(R_NECK, R_CAV, m), np.full(m, L_NECK)), 0., 1.),
    (lambda m: (np.full(m, R_CAV), np.random.uniform(L_NECK, Z_TOP, m)), 1., 0.),
    (lambda m: (np.random.uniform(0, R_CAV, m), np.full(m, Z_TOP)), 0., 1.),
    (lambda m: (np.random.uniform(R_NECK, R_EXT, m), np.zeros(m)), 0., 1.),
]


def sample_walls(m_per):
    S = M_CH*m_per
    rs, zs, nxs, nys = [], [], [], []
    for f, nx, ny in WALL_SEGS:
        a, b = f(S)
        rs.append(a.reshape(M_CH, m_per)); zs.append(b.reshape(M_CH, m_per))
        nxs.append(np.full((M_CH, m_per), nx)); nys.append(np.full((M_CH, m_per), ny))
    n_seg = 5*m_per
    return (np.concatenate(rs, 1).ravel(), np.concatenate(zs, 1).ravel(),
            np.concatenate(nxs, 1).ravel(), np.concatenate(nys, 1).ravel(),
            _t_chunks(n_seg).ravel(), n_seg)


# --------------------------------------------------------------------------
# residus par tranche temporelle (ponderation causale)
# --------------------------------------------------------------------------
def residual_chunks():
    """dp_tt + sig*dp_t - c^2 lap(dp) + sig*p0_t = 0, par tranche."""
    rn, zn = sample_domain(N_PER*M_CH)
    r = Tn(rn, 1); z = Tn(zn, 1); t = Tn(_t_chunks(N_PER).ravel(), 1)
    d = dp_of(r, z, t)
    dr = torch.autograd.grad(d, r, torch.ones_like(d), create_graph=True)[0]
    dz = torch.autograd.grad(d, z, torch.ones_like(d), create_graph=True)[0]
    dt = torch.autograd.grad(d, t, torch.ones_like(d), create_graph=True)[0]
    drr = torch.autograd.grad(dr, r, torch.ones_like(dr), create_graph=True)[0]
    dzz = torch.autograd.grad(dz, z, torch.ones_like(dz), create_graph=True)[0]
    dtt = torch.autograd.grad(dt, t, torch.ones_like(dt), create_graph=True)[0]
    inv_r = torch.where(r < 1e-6, torch.zeros_like(r), 1.0/torch.clamp(r, min=1e-6))
    lap = drr + torch.where(r < 1e-6, drr, dr*inv_r) + dzz
    sig = sigma_of(r, z)
    p0t = _p0_t(r.detach(), z.detach(), t.detach())
    num = dtt + sig*dt - C**2*lap + sig*p0t
    # normalisation UNIFORME : le terme source a disparu, donc la dynamique de
    # 1e4 qui imposait une normalisation locale n'existe plus.
    return ((num/FLOOR)**2).reshape(M_CH, N_PER).mean(1)


# p0 est purement analytique : ses derivees se calculent par autodiff, mais il
# faut FORCER le mode gradient car ls_step tourne sous no_grad.
def _p0_t(r, z, t):
    with torch.enable_grad():
        tt = t.detach().clone().requires_grad_(True)
        v = p0_of(r.detach(), z.detach(), tt)
        return torch.autograd.grad(v, tt, torch.ones_like(v))[0].detach()


def _p0_grad(r, z, t):
    with torch.enable_grad():
        rr = r.detach().clone().requires_grad_(True)
        zz = z.detach().clone().requires_grad_(True)
        v = p0_of(rr, zz, t.detach())
        gr = torch.autograd.grad(v, rr, torch.ones_like(v), retain_graph=True)[0]
        gz = torch.autograd.grad(v, zz, torch.ones_like(v))[0]
        return gr.detach(), gz.detach()


def walls_chunks():
    """d(dp)/dn = -d(p0)/dn : c'est CE second membre qui engendre le champ."""
    rn, zn, nx, ny, tn, n_seg = sample_walls(max(4, N_PER//5))
    r = Tn(rn, 1); z = Tn(zn, 1); t = Tn(tn, 1)
    d = dp_of(r, z, t)
    dr = torch.autograd.grad(d, r, torch.ones_like(d), create_graph=True)[0]
    dz = torch.autograd.grad(d, z, torch.ones_like(d), create_graph=True)[0]
    g0r, g0z = _p0_grad(r.detach(), z.detach(), t.detach())
    nxx, nyy = Tn(nx), Tn(ny)
    resid = (nxx*dr + nyy*dz) + (nxx*g0r + nyy*g0z)
    return ((resid/(SCALE_D/R_NECK))**2).reshape(M_CH, n_seg).mean(1)


# --------------------------------------------------------------------------
# couche de sortie par moindres carres
# --------------------------------------------------------------------------
def _hfun(u):
    return HEAD(torch.stack([u[0]/R_EXT, (u[1] + Z_EXT)/Z_SPAN, u[2]/T_WIN]))


_jac  = vmap(jacfwd(_hfun))
_hess = vmap(jacfwd(jacfwd(_hfun)))


def _stack(rn, zn, tn):
    return torch.stack([torch.as_tensor(rn, dtype=torch.float64),
                        torch.as_tensor(zn, dtype=torch.float64),
                        torch.as_tensor(tn, dtype=torch.float64)], 1)


def _design_pde(rn, zn, tn):
    U = _stack(rn, zn, tn)
    h = HEAD(torch.stack([U[:, 0]/R_EXT, (U[:, 1] + Z_EXT)/Z_SPAN, U[:, 2]/T_WIN], 1))
    J, H2 = _jac(U), _hess(U)
    h_r, h_t = J[:, :, 0], J[:, :, 2]
    h_rr, h_zz, h_tt = H2[:, :, 0, 0], H2[:, :, 1, 1], H2[:, :, 2, 2]
    r, z, t = U[:, 0:1], U[:, 1:2], U[:, 2:3]
    inv_r = torch.where(r < 1e-6, torch.zeros_like(r), 1.0/torch.clamp(r, min=1e-6))
    lap_h = h_rr + torch.where(r < 1e-6, h_rr, h_r*inv_r) + h_zz
    one = torch.ones_like(h[:, :1]); zero = torch.zeros_like(one)
    M1  = torch.cat([h,     one ], 1); Mt  = torch.cat([h_t,   zero], 1)
    Mtt = torch.cat([h_tt,  zero], 1); Ml  = torch.cat([lap_h, zero], 1)
    sig = sigma_of(r, z)
    A = torch.cat([SCALE_D*(qpp*M1 + 2.0*qp*Mt + q*Mtt + sig*(qp*M1 + q*Mt) - C**2*q*Ml)
                   for q, qp, qpp in _q_of_t(t)], 1)
    b = -sig*_p0_t(r, z, t)          # non nul dans la couche absorbante seulement
    return A, b


def _design_walls(rn, zn, tn, nx, ny):
    U = _stack(rn, zn, tn)
    J = _jac(U)
    h_r, h_z = J[:, :, 0], J[:, :, 1]
    zero = torch.zeros_like(h_r[:, :1])
    Mr = torch.cat([h_r, zero], 1); Mz = torch.cat([h_z, zero], 1)
    nxx = torch.as_tensor(nx, dtype=torch.float64).reshape(-1, 1)
    nyy = torch.as_tensor(ny, dtype=torch.float64).reshape(-1, 1)
    r, z, t = U[:, 0:1], U[:, 1:2], U[:, 2:3]
    A = torch.cat([SCALE_D*q*(nxx*Mr + nyy*Mz) for q, _, _ in _q_of_t(t)], 1)
    g0r, g0z = _p0_grad(r, z, t)
    return A, -(nxx*g0r + nyy*g0z)   # second membre NON NUL : dp = 0 est exclu


def ls_step(w_ch):
    n_per = max(4, N_LS // M_CH)
    rn, zn = sample_domain(n_per*M_CH)
    tn = _t_chunks(n_per).ravel()
    with torch.no_grad():
        A, b = _design_pde(rn, zn, tn)
        sw = torch.sqrt(w_ch).repeat_interleave(n_per).reshape(-1, 1)
        A = A*sw/FLOOR; b = b*sw/FLOOR

        m = max(2, n_per//5)
        rw, zw, nx, ny, tw, n_seg = sample_walls(m)
        Aw, bw = _design_walls(rw, zw, tw, nx, ny)
        swW = torch.sqrt(w_ch).repeat_interleave(n_seg).reshape(-1, 1)
        Aw = Aw*swW; bw = bw*swW
        rms = lambda X: (X.norm()/np.sqrt(max(X.shape[0], 1))).clamp(min=1e-30)
        k = (rms(A)/rms(Aw))*np.sqrt(W_WALL)
        Aw = Aw*k; bw = bw*k

        A = torch.cat([A, Aw], 0); b = torch.cat([b, bw], 0)
        AtA = A.T @ A; Atb = A.T @ b
        lam = RIDGE*torch.diagonal(AtA).mean().clamp(min=1e-30)
        th = torch.linalg.solve(AtA + lam*torch.eye(AtA.shape[0]), Atb).reshape(3, D_TH)
        if not torch.isfinite(th).all():
            return False
        cur = torch.cat([OUT.weight, OUT.bias.reshape(3, 1)], 1)
        mix = (1.0 - BETA)*cur + BETA*th
        OUT.weight.copy_(mix[:, :HID]); OUT.bias.copy_(mix[:, HID])
    return True


# --------------------------------------------------------------------------
# BILAN D'ENERGIE  --  la seule contrainte qui fixe l'amplitude
# --------------------------------------------------------------------------
# Tout ce qu'on a impose jusqu'ici est HOMOGENE en dp : residu d'EDP, parois,
# ancrage. Remplacer dp par lambda*dp multiplie chaque residu par lambda, donc
# reduire l'amplitude reduit la perte. Mesure : correlation perte/amplitude
# = +0,908. La perte physique est un amperemetre, pas un indicateur de justesse.
#
# L'identite d'energie casse cette degenerescence. En multipliant l'equation du
# champ diffracte par dp_t et en integrant sur le domaine :
#
#   d/dt E  =  - INT sig*p0_t*dp_t dV  -  INT sig*dp_t^2 dV  -  c^2 OINT dp_t (dp0/dn) dS
#
#   avec   E = INT [ dp_t^2/2 + c^2 |grad dp|^2 /2 ] dV
#
# Le terme de bord ne s'annule PAS ici : sur les parois rigides la condition
# est d(dp)/dn = -dp0/dn, non nulle. C'est justement lui qui alimente dp.
#
# Comportement sous dp -> lambda*dp :
#     d/dt E              -> lambda^2
#     INT sig*p0_t*dp_t   -> lambda      (p0 est FIXE, analytique)
#     INT sig*dp_t^2      -> lambda^2
#     OINT dp_t dp0/dn    -> lambda      (p0 est FIXE)
#
# Des termes en lambda ET en lambda^2 : l'identite n'est vraie qu'a lambda = 1.
# Elle fixe donc l'amplitude de facon ABSOLUE. C'est la version generale de
# l'identite <p>'' = <F> qui avait sauve la cavite fermee -- celle-la etait
# lineaire et ne fixait que la composante continue, celle-ci fixe tout.
#
# Integree de 0 a T_k avec E(0) = 0 (repos initial impose par la porte) :
#
#   E(T_k) + INT_0^Tk [ ... ] dt  =  0
#
# Les integrales sont evaluees sur une GRILLE FIXE, pas par Monte-Carlo : une
# contrainte integrale bruitee ne contraint rien, et le bruit d'echantillonnage
# est deja le defaut principal du pipeline.
N_EG   = int(os.environ.get("SC_NEG", 46))     # resolution radiale de la grille
N_ET   = int(os.environ.get("SC_NET", 8))      # instants pour l'integrale de puissance
N_EW   = int(os.environ.get("SC_NEW", 24))     # points par segment de paroi
W_ENER = float(os.environ.get("SC_WENER", 30.0))
E_EVERY = int(os.environ.get("SC_EEVERY", 3))

# ---- grille fixe du plan meridien, ponderee par r (mesure axisymetrique) ----
def _build_grid():
    nr, nz = N_EG, int(round(N_EG*(Z_SPAN/R_EXT)))
    rr = (np.arange(nr) + 0.5)*R_EXT/nr
    zz = -Z_EXT + (np.arange(nz) + 0.5)*Z_SPAN/nz
    RR, ZZ = np.meshgrid(rr, zz, indexing="ij")
    keep = ((ZZ < 0) |
            ((ZZ >= 0) & (ZZ < L_NECK) & (RR <= R_NECK)) |
            ((ZZ >= L_NECK) & (RR <= R_CAV)))
    dA = (R_EXT/nr)*(Z_SPAN/nz)
    return RR[keep], ZZ[keep], 2.0*np.pi*RR[keep]*dA      # poids = 2 pi r dA

EG_R, EG_Z, EG_W = _build_grid()
V_DOM = float(EG_W.sum())
E_REF = 0.5*(W0*SCALE_D)**2 * V_DOM        # echelle d'energie attendue, CONSTANTE

# ---- quadrature fixe sur les parois, avec la vraie longueur de chaque segment ----
_SEG_LEN = [L_NECK, R_CAV - R_NECK, H_CAV, R_CAV, R_EXT - R_NECK]

def _build_walls():
    rs, zs, nxs, nys, ws = [], [], [], [], []
    for (f, nx, ny), Lseg in zip(WALL_SEGS, _SEG_LEN):
        a, b = f(N_EW)
        a = np.linspace(a.min(), a.max(), N_EW) if np.ptp(a) > 0 else a
        b = np.linspace(b.min(), b.max(), N_EW) if np.ptp(b) > 0 else b
        rs.append(a); zs.append(b)
        nxs.append(np.full(N_EW, nx)); nys.append(np.full(N_EW, ny))
        ws.append(2.0*np.pi*a*(Lseg/N_EW))          # poids = 2 pi r dl
    return (np.concatenate(rs), np.concatenate(zs), np.concatenate(nxs),
            np.concatenate(nys), np.concatenate(ws))

EW_R, EW_Z, EW_NX, EW_NY, EW_W = _build_walls()


# ---- frontiere EXTERIEURE du domaine ----
# Le theoreme de Green porte sur TOUTE la frontiere. Les 5 parois rigides ne
# la couvrent pas : il manque r = R_EXT et z = -Z_EXT, ou l'eponge rend le
# champ petit mais PAS nul. Le test d'identite le montrait crument -- 143,6 %
# d'ecart, avec un terme manquant 12x plus grand que le terme de paroi.
# Sur ces faces, d(dp)/dn n'est pas connu : il vient du reseau.
def _build_outer():
    a = np.full(N_EW, R_EXT)                       # face r = R_EXT, z < 0
    b = np.linspace(-Z_EXT, 0.0, N_EW)
    wa = 2.0*np.pi*a*(Z_EXT/N_EW)
    c_ = np.linspace(0.0, R_EXT, N_EW)             # face z = -Z_EXT
    d_ = np.full(N_EW, -Z_EXT)
    wc = 2.0*np.pi*c_*(R_EXT/N_EW)
    return (np.concatenate([a, c_]), np.concatenate([b, d_]),
            np.concatenate([np.ones(N_EW), np.zeros(N_EW)]),      # normale sortante
            np.concatenate([np.zeros(N_EW), -np.ones(N_EW)]),
            np.concatenate([wa, wc]))


EO_R, EO_Z, EO_NX, EO_NY, EO_W = _build_outer()
print(f"bilan d'energie | grille {EG_R.size} points, V = {V_DOM*1e6:.1f} cm3 "
      f"| parois {EW_R.size} + exterieur {EO_R.size} points | E_ref = {E_REF:.3e}")


def _dp_derivs(rn, zn, tn):
    """dp, dp_t, dp_r, dp_z aux points donnes."""
    r = Tn(rn, 1); z = Tn(zn, 1); t = Tn(tn, 1)
    d = dp_of(r, z, t)
    g = torch.autograd.grad(d, [r, z, t], torch.ones_like(d), create_graph=True)
    return d, g[2], g[0], g[1]


def energy_residual():
    """Residu du bilan d'energie, un par instant de controle -> (N_ET,)."""
    nP = EG_R.size
    tk = np.linspace(T_WIN/N_ET, T_WIN, N_ET)          # instants de controle
    # --- E(T_k) : grille evaluee a chaque instant de controle ---
    rr = np.tile(EG_R, N_ET); zz = np.tile(EG_Z, N_ET)
    tt = np.repeat(tk, nP)
    _, dt_, dr_, dz_ = _dp_derivs(rr, zz, tt)
    wv = Tn(np.tile(EG_W, N_ET))
    e_dens = 0.5*dt_**2 + 0.5*C**2*(dr_**2 + dz_**2)
    E_k = (wv*e_dens).reshape(N_ET, nP).sum(1)

    # --- integrale de puissance de 0 a T_k, quadrature du point milieu ---
    dtq = T_WIN/(N_ET*max(1, N_ET))                     # pas fin non necessaire
    nq = N_ET*2
    tq = (np.arange(nq) + 0.5)*T_WIN/nq                 # instants de quadrature
    rr2 = np.tile(EG_R, nq); zz2 = np.tile(EG_Z, nq); tt2 = np.repeat(tq, nP)
    _, dtq_, _, _ = _dp_derivs(rr2, zz2, tt2)
    sig = sigma_of(Tn(rr2), Tn(zz2))
    p0t = _p0_t(Tn(rr2), Tn(zz2), Tn(tt2))
    wv2 = Tn(np.tile(EG_W, nq))
    integ = (wv2*(sig*p0t*dtq_ + sig*dtq_**2)).reshape(nq, nP).sum(1)

    # --- terme de bord : c^2 * OINT dp_t * (dp0/dn) dS ---
    nW = EW_R.size
    rw = np.tile(EW_R, nq); zw = np.tile(EW_Z, nq); tw = np.repeat(tq, nW)
    _, dtw_, _, _ = _dp_derivs(rw, zw, tw)
    g0r, g0z = _p0_grad(Tn(rw), Tn(zw), Tn(tw))
    dn0 = Tn(np.tile(EW_NX, nq))*g0r + Tn(np.tile(EW_NY, nq))*g0z
    ww = Tn(np.tile(EW_W, nq))
    integ = integ + C**2*(ww*dtw_*dn0).reshape(nq, nW).sum(1)

    # frontiere EXTERIEURE : d(dp)/dn vient du reseau, pas de p0
    nO = EO_R.size
    ro = np.tile(EO_R, nq); zo = np.tile(EO_Z, nq); to = np.repeat(tq, nO)
    _, dto_, dro_, dzo_ = _dp_derivs(ro, zo, to)
    dnO = Tn(np.tile(EO_NX, nq))*dro_ + Tn(np.tile(EO_NY, nq))*dzo_
    wo = Tn(np.tile(EO_W, nq))
    integ = integ - C**2*(wo*dto_*dnO).reshape(nq, nO).sum(1)

    # cumul de 0 a T_k  (nq = 2*N_ET instants, deux par intervalle de controle)
    dtq = T_WIN/nq
    cum = torch.cumsum(integ*dtq, 0)[1::2]              # aux instants tk
    return ((E_k + cum)/E_REF)**2


def modal_residual(n=400):
    """Apres l'impulsion, la cavite oscille librement : p_tt + w0^2 p ~ 0.

    w0 vient de la formule de Helmholtz (geometrie seule). Cela ne fixe pas
    l'amplitude mais contraint fortement forme et phase -- ce qui reste faux.
    """
    rn = np.random.uniform(0, R_CAV, n)
    zn = np.random.uniform(L_NECK, Z_TOP, n)
    tn = np.random.uniform(T_MODAL, T_WIN, n)
    r = Tn(rn, 1); z = Tn(zn, 1); t = Tn(tn, 1)
    pp = p_of(r, z, t)
    pt = torch.autograd.grad(pp, t, torch.ones_like(pp), create_graph=True)[0]
    ptt = torch.autograd.grad(pt, t, torch.ones_like(pt), create_graph=True)[0]
    return torch.mean(((ptt + W0**2*pp)/FLOOR)**2)


T_MODAL = float(os.environ.get("SC_TMOD", 18.0))*1e-3
W_MODAL = float(os.environ.get("SC_WMOD", 2.0))


# --------------------------------------------------------------------------
# reference FDM et selection sans reference
# --------------------------------------------------------------------------
ref = np.load("data/open_resonator.npz")
msk = ref["t"] <= T_WIN + 1e-12
t_ref = ref["t"][msk]; pc_ref = ref["p_cav"][msk]; po_ref = ref["p_out"][msk]
Z_CAV_PROBE, Z_OUT_PROBE = 0.08, -0.02


# --------------------------------------------------------------------------
# VERIFICATION DE L'IDENTITE D'ENERGIE  (SC_TEST=1)
# --------------------------------------------------------------------------
# L'identite d/dt E = -(...)  n'est vraie que pour une SOLUTION. On ne peut donc
# pas la verifier sur un champ quelconque. Mais la relation dont elle derive,
# elle, vaut pour TOUT champ : en multipliant le residu par dp_t et en integrant,
#
#   INT R*dp_t dV = d/dt E + INT sig*dp_t^2 + INT sig*p0_t*dp_t - c^2 OINT dp_t (d dp/dn)
#
# soit   d/dt E = INT R*dp_t - INT sig*dp_t^2 - INT sig*p0_t*dp_t + c^2 OINT dp_t (d dp/dn)
#
# Ceci est une identite MATHEMATIQUE, valable meme sur un reseau non entraine.
# Si mes quadratures sont justes, les deux membres coincident. C'est la meme
# demarche que la MMS du volet 2 : verifier l'instrument avant de croire sa mesure.
def _energy_at(tval):
    nP = EG_R.size
    _, dt_, dr_, dz_ = _dp_derivs(EG_R, EG_Z, np.full(nP, tval))
    w = Tn(EG_W)
    return float((w*(0.5*dt_**2 + 0.5*C**2*(dr_**2 + dz_**2))).sum().detach())


def test_energy():
    tv, hdt = 0.030, 2.0e-6
    dEdt = (_energy_at(tv + hdt) - _energy_at(tv - hdt))/(2*hdt)

    nP = EG_R.size
    r = Tn(EG_R, 1); z = Tn(EG_Z, 1); t = Tn(np.full(nP, tv), 1)
    d = dp_of(r, z, t)
    g = torch.autograd.grad(d, [r, z, t], torch.ones_like(d), create_graph=True)
    dr_, dz_, dt_ = g[0], g[1], g[2]
    drr = torch.autograd.grad(dr_, r, torch.ones_like(dr_), create_graph=True)[0]
    dzz = torch.autograd.grad(dz_, z, torch.ones_like(dz_), create_graph=True)[0]
    dtt = torch.autograd.grad(dt_, t, torch.ones_like(dt_), create_graph=True)[0]
    inv_r = torch.where(r < 1e-6, torch.zeros_like(r), 1.0/torch.clamp(r, min=1e-6))
    lap = drr + torch.where(r < 1e-6, drr, dr_*inv_r) + dzz
    sig = sigma_of(r, z); p0t = _p0_t(r, z, t)
    R = dtt + sig*dt_ - C**2*lap + sig*p0t
    w = Tn(EG_W)
    T1 = float((w*R*dt_).sum().detach())
    T2 = float((w*sig*dt_**2).sum().detach())
    T3 = float((w*sig*p0t*dt_).sum().detach())

    nW = EW_R.size
    _, dtw, drw, dzw = _dp_derivs(EW_R, EW_Z, np.full(nW, tv))
    dndp = Tn(EW_NX)*drw + Tn(EW_NY)*dzw
    ww = Tn(EW_W)
    T4 = float((ww*dtw*dndp).sum().detach())
    nO = EO_R.size
    _, dto, dro, dzo = _dp_derivs(EO_R, EO_Z, np.full(nO, tv))
    dnO = Tn(EO_NX)*dro + Tn(EO_NY)*dzo
    T4o = float((Tn(EO_W)*dto*dnO).sum().detach())
    T4 = T4 + T4o

    rhs = T1 - T2 - T3 + C**2*T4
    ech = max(abs(dEdt), abs(rhs), abs(T1), abs(C**2*T4), 1e-300)
    print(chr(10) + "=== VERIFICATION DE L'IDENTITE D'ENERGIE ===")
    print(f"  d/dt E  (differences finies) = {dEdt:+.6e}")
    print(f"  membre de droite            = {rhs:+.6e}")
    print(f"    INT R*dp_t                = {T1:+.6e}")
    print(f"    INT sig*dp_t^2            = {T2:+.6e}")
    print(f"    INT sig*p0_t*dp_t         = {T3:+.6e}")
    print(f"    c^2 * OINT dp_t d(dp)/dn  = {C**2*T4:+.6e}  "
          f"(dont exterieur {C**2*T4o:+.3e})")
    print(f"  ecart relatif               = {abs(dEdt-rhs)/ech*100:.3f} %")
    return abs(dEdt-rhs)/ech


if os.environ.get("SC_TEST", "0") == "1":
    test_energy(); raise SystemExit(0)


# --------------------------------------------------------------------------
# BILAN D'ENERGIE PAR REGION
# --------------------------------------------------------------------------
# Le bilan global est UN SEUL SCALAIRE : il dit combien d'energie, pas OU.
# Mesure : avec lui seul, le terme d'energie descend a 0,11 pendant que la
# sonde cavite reste 5x trop faible -- le reseau met la bonne energie totale
# dans l'exterieur au lieu de la cavite, et les comptes sont justes.
#
# La meme identite vaut sur n'importe quel sous-domaine, a condition d'ajouter
# le flux a travers sa frontiere interne. On l'ecrit donc sur trois regions :
#
#     d/dt E_i = - INT_i sig*p0_t*dp_t - INT_i sig*dp_t^2
#                + c^2 OINT_i dp_t (d dp/dn)
#
# Distinction essentielle sur le bord :
#   * PAROI RIGIDE   -> d(dp)/dn = -dp0/dn, valeur CONNUE et fixe. C'est elle
#                       qui rend le terme lineaire en dp (et non quadratique),
#                       donc qui fixe l'amplitude.
#   * INTERFACE ou FRONTIERE EXTERIEURE -> d(dp)/dn vient du reseau.
#
# Les interfaces apparaissent deux fois avec des normales opposees : sommer les
# trois identites redonne exactement l'identite globale. C'est un test de
# coherence gratuit, et il tourne avant l'entrainement.
def _grid(r0, r1, z0, z1, nr, nz):
    rr = r0 + (np.arange(nr) + 0.5)*(r1 - r0)/nr
    zz = z0 + (np.arange(nz) + 0.5)*(z1 - z0)/nz
    RR, ZZ = np.meshgrid(rr, zz, indexing="ij")
    w = 2.0*np.pi*RR*((r1 - r0)/nr)*((z1 - z0)/nz)
    return RR.ravel(), ZZ.ravel(), w.ravel()


def _bnd(axis, v0, v1, fixed, nx, ny, n, wall):
    """Segment de bord. axis='r' : r varie a z=fixed. axis='z' : z varie a r=fixed."""
    q = v0 + (np.arange(n) + 0.5)*(v1 - v0)/n
    if axis == "r":
        r, z = q, np.full(n, fixed)
    else:
        r, z = np.full(n, fixed), q
    w = 2.0*np.pi*r*abs(v1 - v0)/n
    return dict(r=r, z=z, nx=np.full(n, nx), ny=np.full(n, ny), w=w, wall=wall)


NB = int(os.environ.get("SC_NB", 40))     # points par segment de bord
NG = int(os.environ.get("SC_NG", 26))     # finesse des grilles de region

REGIONS = [
    dict(nom="exterieur",
         grid=_grid(0.0, R_EXT, -Z_EXT, 0.0, NG, NG),
         bnd=[_bnd("r", R_NECK, R_EXT, 0.0, 0., +1., NB, True),    # baffle (paroi)
              _bnd("r", 0.0, R_NECK, 0.0, 0., +1., NB, False),     # bouche (interface)
              _bnd("z", -Z_EXT, 0.0, R_EXT, +1., 0., NB, False),   # bord lointain
              _bnd("r", 0.0, R_EXT, -Z_EXT, 0., -1., NB, False)]), # bord lointain
    dict(nom="col",
         grid=_grid(0.0, R_NECK, 0.0, L_NECK, NG, NG),
         bnd=[_bnd("z", 0.0, L_NECK, R_NECK, +1., 0., NB, True),   # paroi du col
              _bnd("r", 0.0, R_NECK, L_NECK, 0., +1., NB, False),  # vers cavite
              _bnd("r", 0.0, R_NECK, 0.0, 0., -1., NB, False)]),   # bouche
    dict(nom="cavite",
         grid=_grid(0.0, R_CAV, L_NECK, Z_TOP, NG, NG),
         bnd=[_bnd("z", L_NECK, Z_TOP, R_CAV, +1., 0., NB, True),  # paroi laterale
              _bnd("r", 0.0, R_CAV, Z_TOP, 0., +1., NB, True),     # fond
              _bnd("r", R_NECK, R_CAV, L_NECK, 0., -1., NB, True), # epaulement
              _bnd("r", 0.0, R_NECK, L_NECK, 0., -1., NB, False)]),# vers col
]

for _rg in REGIONS:
    _rg["V"] = float(_rg["grid"][2].sum())
    _rg["Eref"] = 0.5*(W0*SCALE_D)**2 * _rg["V"]
    for _b in _rg["bnd"]:
        _b["N"] = _b["r"].size
print("bilan par region | " + " · ".join(
    f"{g['nom']} {g['grid'][0].size} pts, V={g['V']*1e6:.1f} cm3" for g in REGIONS))


def _region_terms(rg, tq, physique=True):
    """Termes du bilan de la region, aux instants tq. -> (dEdt_integrand, flux)."""
    gr, gz, gw = rg["grid"]
    nP = gr.size; nq = tq.size

    # --- puissance dissipee et injectee dans le volume ---
    rr = np.tile(gr, nq); zz = np.tile(gz, nq); tt = np.repeat(tq, nP)
    _, dt_, _, _ = _dp_derivs(rr, zz, tt)
    sig = sigma_of(Tn(rr), Tn(zz))
    p0t = _p0_t(Tn(rr), Tn(zz), Tn(tt))
    wv = Tn(np.tile(gw, nq))
    vol = (wv*(sig*p0t*dt_ + sig*dt_**2)).reshape(nq, nP).sum(1)

    # --- flux au bord ---
    flux = torch.zeros(nq, dtype=torch.float64)
    for b in rg["bnd"]:
        nB = b["N"]
        rb = np.tile(b["r"], nq); zb = np.tile(b["z"], nq); tb = np.repeat(tq, nB)
        _, dtb, drb, dzb = _dp_derivs(rb, zb, tb)
        if physique and b["wall"]:
            # paroi rigide : d(dp)/dn = -dp0/dn, valeur CONNUE. C'est ce terme
            # lineaire en dp qui brise l'invariance d'echelle.
            g0r, g0z = _p0_grad(Tn(rb), Tn(zb), Tn(tb))
            dn = -(Tn(np.tile(b["nx"], nq))*g0r + Tn(np.tile(b["ny"], nq))*g0z)
        else:
            dn = Tn(np.tile(b["nx"], nq))*drb + Tn(np.tile(b["ny"], nq))*dzb
        wb = Tn(np.tile(b["w"], nq))
        flux = flux + (wb*dtb*dn).reshape(nq, nB).sum(1)
    return vol, C**2*flux


def _region_energy(rg, tval):
    gr, gz, gw = rg["grid"]
    _, dt_, dr_, dz_ = _dp_derivs(gr, gz, np.full(gr.size, tval))
    w = Tn(gw)
    return (w*(0.5*dt_**2 + 0.5*C**2*(dr_**2 + dz_**2))).sum()


def energy_regions():
    """Residu du bilan, par region et par instant de controle -> (3*N_ET,)."""
    tk = np.linspace(T_WIN/N_ET, T_WIN, N_ET)
    nq = 2*N_ET
    tq = (np.arange(nq) + 0.5)*T_WIN/nq
    dtq = T_WIN/nq
    out = []
    for rg in REGIONS:
        vol, flux = _region_terms(rg, tq, physique=True)
        cum = torch.cumsum((vol - flux)*dtq, 0)[1::2]
        Ek = torch.stack([_region_energy(rg, float(t)) for t in tk])
        out.append(((Ek + cum)/rg["Eref"])**2)
    return torch.cat(out)


def test_regions():
    """Verifie l'identite MATHEMATIQUE region par region, sur un reseau non entraine.

    d/dt E_i = INT_i R*dp_t - INT_i sig*dp_t^2 - INT_i sig*p0_t*dp_t
               + c^2 OINT_i dp_t (d dp/dn)
    """
    tv, hdt = 0.030, 2.0e-6
    print(chr(10) + "=== IDENTITE D'ENERGIE, REGION PAR REGION ===")
    pires = 0.0
    for rg in REGIONS:
        dEdt = float((_region_energy(rg, tv + hdt) - _region_energy(rg, tv - hdt)).detach())/(2*hdt)
        gr, gz, gw = rg["grid"]; nP = gr.size
        r = Tn(gr, 1); z = Tn(gz, 1); t = Tn(np.full(nP, tv), 1)
        d = dp_of(r, z, t)
        g = torch.autograd.grad(d, [r, z, t], torch.ones_like(d), create_graph=True)
        dr_, dz_, dt_ = g[0], g[1], g[2]
        drr = torch.autograd.grad(dr_, r, torch.ones_like(dr_), create_graph=True)[0]
        dzz = torch.autograd.grad(dz_, z, torch.ones_like(dz_), create_graph=True)[0]
        dtt = torch.autograd.grad(dt_, t, torch.ones_like(dt_), create_graph=True)[0]
        inv_r = torch.where(r < 1e-6, torch.zeros_like(r), 1.0/torch.clamp(r, min=1e-6))
        lap = drr + torch.where(r < 1e-6, drr, dr_*inv_r) + dzz
        sig = sigma_of(r, z); p0t = _p0_t(r, z, t)
        R = dtt + sig*dt_ - C**2*lap + sig*p0t
        w = Tn(gw)
        T1 = float((w*R*dt_).sum().detach())
        T2 = float((w*sig*dt_**2).sum().detach())
        T3 = float((w*sig*p0t*dt_).sum().detach())
        _, flux = _region_terms(rg, np.array([tv]), physique=False)
        T4 = float(flux[0].detach())
        rhs = T1 - T2 - T3 + T4
        ech = max(abs(dEdt), abs(rhs), abs(T1), abs(T4), 1e-300)
        err = abs(dEdt - rhs)/ech
        pires = max(pires, err)
        print(f"  {rg['nom']:10s} dE/dt = {dEdt:+.4e}   membre droit = {rhs:+.4e}   "
              f"ecart = {err*100:6.3f} %")
    print(f"  pire ecart : {pires*100:.3f} %")
    return pires


if os.environ.get("SC_TESTR", "0") == "1":
    test_regions(); raise SystemExit(0)


def probes():
    with torch.no_grad():
        zc = Tn(np.full_like(t_ref, Z_CAV_PROBE)); zo = Tn(np.full_like(t_ref, Z_OUT_PROBE))
        r0 = Tn(np.zeros_like(t_ref)); tt = Tn(t_ref)
        return (p_of(r0, zc, tt).numpy().ravel(), p_of(r0, zo, tt).numpy().ravel())


def l2(a, b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))


VAL_SEED = 12345


def phys_val():
    """Critere de selection : lot FIXE, aucune reference FDM."""
    st = np.random.get_state(); np.random.seed(VAL_SEED)
    v = float((residual_chunks().mean() + W_WALL*walls_chunks().mean()
               + W_ENER*energy_regions().mean()
               + W_MODAL*modal_residual()).detach())
    np.random.set_state(st)
    return v


# --------------------------------------------------------------------------
# entrainement
# --------------------------------------------------------------------------
opt = torch.optim.Adam(list(HEAD.parameters()), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
ie = 0; EPS = EPS_SCHED[ie]
best = np.inf; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
hist = []; t0 = time.time()
print(f"{M_CH} tranches de {DT_CH*1e3:.2f} ms x {N_PER} pts | {ITERS} iterations\n")

for it in range(1, ITERS+1):
    opt.zero_grad()
    L, Lw = residual_chunks(), walls_chunks()
    with torch.no_grad():
        Ld = L.detach()
        cum = (torch.cumsum(Ld, 0) - Ld)/Ld.sum().clamp(min=1e-30)
        w = torch.exp(-EPS*cum)
    if not ls_step(w):
        print(f"  it {it:5d} : moindres carres non finis, pas ignore")
    L, Lw = residual_chunks(), walls_chunks()
    loss = (w*L).mean() + W_WALL*(w*Lw).mean() + W_MODAL*modal_residual()
    if it % E_EVERY == 0:
        loss = loss + W_ENER*energy_regions().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()

    if it % CHECK == 0:
        phys = phys_val()
        pc, po = probes()
        e_c, e_o = l2(pc, pc_ref), l2(po, po_ref)
        amp = np.abs(pc).max(); wmin = float(w.min())
        if phys < best:
            best = phys
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = " *"
        else:
            flag = "  "
        with torch.no_grad(): pass
        e_ener = float(energy_regions().mean().detach())
        e_mod  = float(modal_residual().detach())
        hist.append((it, EPS, wmin, float(L.mean().detach()), float(Lw.mean().detach()),
                     phys, e_c, e_o, amp, e_ener, e_mod))
        print(f"  it {it:5d} | phys={phys:.3e}{flag}| EDP={float(L.mean().detach()):.2e} "
              f"Paroi={float(Lw.mean().detach()):.1e} Ener={e_ener:.2e} Mod={e_mod:.2e} "
              f"| L2 cav={e_c*100:6.1f} % ext={e_o*100:6.1f} % | amp={amp:.3e} "
              f"(FDM {np.abs(pc_ref).max():.3e}) | {time.time()-t0:5.0f}s")
        if wmin > DELTA and ie < len(EPS_SCHED)-1:
            ie += 1; EPS = EPS_SCHED[ie]
            print(f"        -> fenetre entierement entrainee, eps monte a {EPS:.0f}")

model.load_state_dict(best_state)
print(f"\nmodele retenu : perte physique = {best:.3e} sur lot fixe (sans reference)")

pc, po = probes()
print("\n=== VERDICT (champ diffracte) ===")
print(f"L2 sonde cavite     : {l2(pc, pc_ref)*100:.1f} %")
print(f"L2 sonde exterieure : {l2(po, po_ref)*100:.1f} %")
print(f"amplitude cavite    : {np.abs(pc).max():.3e} Pa   (FDM {np.abs(pc_ref).max():.3e} Pa)")

torch.save({"state": model.state_dict(), "scale": SCALE_D, "w0": W0,
            "hid": HID, "T_WIN": T_WIN}, f"models/pinn_scat{TAG}.pth")
np.savez(f"data/pinn_scat{TAG}.npz", t=t_ref, pc=pc, pc_ref=pc_ref, po=po, po_ref=po_ref,
         hist=np.array(hist), T_WIN=T_WIN, F0=F0)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, (y, yr, nom) in zip(ax, [(pc, pc_ref, "cavité"), (po, po_ref, "extérieure")]):
    a.plot(t_ref*1e3, yr, "k", lw=1.2, label="FDM")
    a.plot(t_ref*1e3, y, "C1", lw=1.0, label="PINN champ diffracté")
    a.set(xlabel="t (ms)", ylabel="p (Pa)", title=f"sonde {nom} | L2 = {l2(y, yr)*100:.1f} %")
    a.legend(); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"plots/pinn_scat{TAG}.png", dpi=110)
print(f"Figure : plots/pinn_scat{TAG}.png | Modele : models/pinn_scat{TAG}.pth")
