"""
PINN transitoire sur le resonateur a COL OUVERT (extérieur maille).

Objectif : reproduire le scenario complet — impulsion qui se propage, entree par
le col, remplissage de la cavite, puis resonance libre — avec un reseau, pour en
tirer une animation.

C'est le regime ou les PINN echouent classiquement (biais spectral : ~13 periodes
d'oscillation a 210 Hz sur la fenetre visee). Deux dispositifs pour lui donner
une chance :

1. ANSATZ A ENVELOPPES. Au lieu de faire apprendre l'oscillation, on la factorise :

       p = scale * g(t) * [ A(x)*cos(w0*t) + B(x)*sin(w0*t) + C(x) ]

   Le reseau ne produit que trois fonctions LENTES (A, B, C). Les derivees des
   cos/sin sont exactes via la differentiation automatique. g(t) = 1-exp(-(t/tau)^2)
   impose le repos initial par construction (g(0)=g'(0)=0).

2. RESIDU NORMALISE PAR UNE ECHELLE CONNUE. La source est intense et tres
   localisee : en residu absolu elle ecrase tout le reste. On divise donc par
   |F| + FLOOR, ou F est le forcage ANALYTIQUE et FLOOR = w0^2 * scale la
   magnitude d'onde attendue. Ces deux quantites sont connues d'avance et ne
   dependent PAS de la solution -- une normalisation batie sur |p_tt|, |lap p|
   rendrait la perte invariante d'echelle, et un champ 100x trop petit
   obtiendrait le meme score qu'un champ correct.

Physique identique a fdm_open_resonator.py : meme geometrie, meme couche absorbante
(le terme sigma*p_t fait partie de l'EDP), meme impulsion de Ricker.

Verification : signaux de sonde compares a data/open_resonator.npz (FDM h=1mm).

Env : OP_MODE ("causal" | "curriculum")   OP_TMS (fenetre ms, 60)
      causal     : OP_M (tranches, 32) OP_ITERS_TOT (4000) OP_CHECK (100) OP_DELTA (0.99)
      moindres carres : OP_LS (1) OP_NLS (1200) OP_LS_EVERY (1) OP_RIDGE (1e-8)
                        OP_ALPHA (0.5, normalisation) OP_BETA (0.15, amortissement)
      ancrage         : OP_RANCH (3.0, en sigma) OP_WANCH (20.0, poids)
      moyenne         : OP_EMA (0.01, taux de Polyak)
      composante cont.: OP_WDC (10.0) OP_NDC (300) OP_NTDC (3)
      curriculum : OP_ITERS (par palier, 400) OP_K (paliers, 8)
      commun     : OP_NCOL (1800) OP_HID (128) OP_LR (1e-3) OP_SEED (0) OP_TAG ("")
"""
import os, time
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
SEED = int(os.environ.get("OP_SEED", 0))
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cpu"; torch.set_num_threads(4)
# float64 partout : jacfwd construit ses tangentes en double, et le systeme
# normal des moindres carres (387 inconnues) y gagne en conditionnement.
torch.set_default_dtype(torch.float64)

# ---- geometrie et physique : strictement celles de fdm_open_resonator.py ----
R_NECK, R_CAV, L_NECK, H_CAV = 0.01, 0.04, 0.04, 0.08
Z_TOP = L_NECK + H_CAV            # 0.12
C = 343.0
R_EXT, Z_EXT, L_SP = 0.12, 0.10, 0.03
SIG_MAX = 5.0 * C / L_SP
F_C, Z_SRC, W_SRC, S0 = 250.0, -0.055, 0.004, 8.0e6
T0_R = 1.3 / F_C

F0 = 209.8                        # frequence propre mesuree par le FDM
W0 = 2.0 * np.pi * F0

T_WIN = float(os.environ.get("OP_TMS", 60.0)) * 1e-3
ITERS = int(os.environ.get("OP_ITERS", 400))
K_PAL = int(os.environ.get("OP_K", 8))
N_COL = int(os.environ.get("OP_NCOL", 1800))
HID   = int(os.environ.get("OP_HID", 128))
LR    = float(os.environ.get("OP_LR", 1e-3))
TAG   = os.environ.get("OP_TAG", "")
MODE  = os.environ.get("OP_MODE", "causal")   # "causal" | "curriculum"
USE_LS   = os.environ.get("OP_LS", "1") == "1"   # couche de sortie par moindres carres
# Normalisation du residu, famille continue a un parametre :
#     den = FLOOR * ((|F|+FLOOR)/FLOOR)**ALPHA
# ALPHA=1 -> "local"  : |F|+FLOOR. Les lignes de source sont ecrasees,
#                       la solution nulle devient presque optimale.
# ALPHA=0 -> "floor"  : constante. La zone source ecrase tout, la cavite
#                       (fraction infime du residu) derive librement.
# Entre les deux, chaque region pese de facon comparable.
ALPHA = float(os.environ.get("OP_ALPHA", 0.5))

SCALE_P = 3.0e-4                  # amplitude cavite du FDM
FLOOR   = None                    # echelle d'onde attendue, fixee plus bas
TAU_G   = 0.8e-3                  # ouverture de la porte de repos
Z_SPAN  = Z_TOP + Z_EXT           # 0.22

print(f"fenetre {T_WIN*1e3:.0f} ms = {T_WIN*F0:.1f} periodes | {K_PAL} paliers x {ITERS} it")
FLOOR = W0**2 * SCALE_P           # magnitude attendue de p_tt et c^2 lap p
print(f"ansatz enveloppes a w0 = {W0:.1f} rad/s (f0 = {F0} Hz) | scale = {SCALE_P:.1e} Pa")
print(f"normalisation du residu : |F| + {FLOOR:.1f}")


# --------------------------------------------------------------------------
# source et couche absorbante (identiques au FDM)
# --------------------------------------------------------------------------
def ricker_t(t):
    a = (np.pi * F_C * (t - T0_R)) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)

def forcing(r, z, t):
    sp = torch.exp(-(r**2 + (z - Z_SRC)**2) / (2.0 * W_SRC**2))
    return S0 * sp * ricker_t(t)

def sigma_of(r, z):
    dz = torch.clamp((-Z_EXT + L_SP - z) / L_SP, 0.0, 1.0)
    dr = torch.clamp((r - (R_EXT - L_SP)) / L_SP, 0.0, 1.0)
    d = torch.maximum(dz, dr)
    return SIG_MAX * d**2 * (z < 0).float()      # couche seulement dans l'exterieur


# --------------------------------------------------------------------------
# reseau : features de Fourier en ESPACE (pas en temps : l'oscillation est
# deja portee par l'ansatz), trois sorties A, B, C
# --------------------------------------------------------------------------
class EnvNet(nn.Module):
    """MLP lisse, SANS features de Fourier spatiales.

    A 210 Hz, lambda = 1,63 m pour un domaine de 0,22 m : le champ est
    SOUS-LONGUEUR D'ONDE, donc quasi uniforme en espace. Des features de
    Fourier a sigma=3 donnaient des nombres d'onde 22 a 41 fois trop grands,
    d'ou un terme c^2*lap(p) 500 a 1700 fois trop grand : la perte n'etait
    plus que du bruit. Les enveloppes A, B, C sont des fonctions LENTES,
    qu'un simple MLP a tangente hyperbolique represente sans peine.
    """
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

def Tn(a, g=False):
    return torch.tensor(a, dtype=torch.float64, device=DEV).reshape(-1, 1).requires_grad_(bool(g))

def p_of(r, z, t):
    x = torch.cat([r/R_EXT, (z + Z_EXT)/Z_SPAN, t/T_WIN], 1)
    o = model(x)
    A, Bc, Cc = o[:, 0:1], o[:, 1:2], o[:, 2:3]
    g = 1.0 - torch.exp(-(t/TAU_G)**2)
    return SCALE_P * g * (A*torch.cos(W0*t) + Bc*torch.sin(W0*t) + Cc)


# --------------------------------------------------------------------------
# echantillonnage du domaine (exterieur + col + cavite)
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


def loss_pde(H, Hprev, frac_src=0.20):
    n_src = int(N_COL*frac_src); n_uni = N_COL - n_src
    r1, z1 = sample_domain(n_uni)
    r2 = np.clip(np.abs(np.random.normal(0, W_SRC, n_src)), 0, R_EXT)
    z2 = np.clip(np.random.normal(Z_SRC, 2*W_SRC, n_src), -Z_EXT, -1e-4)
    r = np.concatenate([r1, r2]); z = np.concatenate([z1, z2])
    # --- temps des points UNIFORMES : horizon courant + emphase causale ---
    t = np.empty(r.size)
    n_new = int(n_uni*0.35)
    t[:n_uni-n_new] = np.random.uniform(0, H, n_uni-n_new)
    t[n_uni-n_new:n_uni] = np.random.uniform(Hprev, H, n_new)
    # --- temps des points SOURCE : cibles sur la fenetre ou la source existe ---
    # Sans cela, la source (0,87 % du volume x 11 % du temps = 0,1 % du volume
    # espace-temps) n'est quasiment jamais tiree : la plupart des lots ne
    # contiennent AUCUN point porteur du forcage, et la perte devient aveugle
    # a ce qui pilote toute la solution.
    W_RICK = 1.5/(np.pi*F_C)
    ts = np.random.normal(T0_R, W_RICK, n_src)
    t[n_uni:] = np.clip(ts, 0.0, max(H, 1e-6))

    r = Tn(r, 1); z = Tn(z, 1); t = Tn(t, 1)
    p = p_of(r, z, t)
    pr = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    pz = torch.autograd.grad(p, z, torch.ones_like(p), create_graph=True)[0]
    pt = torch.autograd.grad(p, t, torch.ones_like(p), create_graph=True)[0]
    prr = torch.autograd.grad(pr, r, torch.ones_like(pr), create_graph=True)[0]
    pzz = torch.autograd.grad(pz, z, torch.ones_like(pz), create_graph=True)[0]
    ptt = torch.autograd.grad(pt, t, torch.ones_like(pt), create_graph=True)[0]
    inv_r = torch.where(r < 1e-6, torch.zeros_like(r), 1.0/torch.clamp(r, min=1e-6))
    lap = prr + torch.where(r < 1e-6, prr, pr*inv_r) + pzz

    F = forcing(r, z, t)
    sig = sigma_of(r, z)
    num = ptt + sig*pt - C**2*lap - F
    # Denominateur construit UNIQUEMENT a partir de quantites connues d'avance
    # (le forcage analytique et l'echelle d'onde attendue) : il ne depend pas de
    # la solution, donc la perte n'est PAS invariante d'echelle et l'amplitude
    # reste contrainte. Un denominateur bati sur |p_tt|, |lap p| rendait au
    # contraire un champ 100x trop petit aussi bon qu'un champ correct.
    den = F.abs() + FLOOR
    return torch.mean((num/den)**2)


def loss_walls(H):
    m = N_COL//6
    segs = [
        (np.full(m, R_NECK), np.random.uniform(0, L_NECK, m), 1., 0.),          # paroi du col
        (np.random.uniform(R_NECK, R_CAV, m), np.full(m, L_NECK), 0., 1.),      # epaulement
        (np.full(m, R_CAV), np.random.uniform(L_NECK, Z_TOP, m), 1., 0.),       # paroi cavite
        (np.random.uniform(0, R_CAV, m), np.full(m, Z_TOP), 0., 1.),            # fond
        (np.random.uniform(R_NECK, R_EXT, m), np.zeros(m), 0., 1.),             # BAFFLE
    ]
    rw = np.concatenate([s[0] for s in segs]); zw = np.concatenate([s[1] for s in segs])
    nx = np.concatenate([np.full(m, s[2]) for s in segs])
    ny = np.concatenate([np.full(m, s[3]) for s in segs])
    r = Tn(rw, 1); z = Tn(zw, 1); t = Tn(np.random.uniform(0, H, rw.size), 1)
    p = p_of(r, z, t)
    pr = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    pz = torch.autograd.grad(p, z, torch.ones_like(p), create_graph=True)[0]
    gn = (Tn(nx)*pr + Tn(ny)*pz) / (SCALE_P/R_NECK)
    return torch.mean(gn**2)


# --------------------------------------------------------------------------
# PONDERATION CAUSALE  (Wang, Sankaran & Perdikaris 2022, arXiv:2203.07404)
# --------------------------------------------------------------------------
# Un PINN entraine tous les instants SIMULTANEMENT : rien ne l'empeche
# d'ajuster t = 55 ms avant d'avoir compris t = 1 ms. Sur 12,6 periodes
# d'oscillation, c'est la violation la plus couteuse du projet.
#
# Le remede : decouper [0, T_WIN] en M tranches, calculer le residu moyen
# L_i de chacune, et ponderer
#
#       w_i = exp( -eps * somme_{j<i} L_j )
#
# Une tranche ne pese que si TOUTES celles qui la precedent sont deja bien
# apprises. C'est un curriculum CONTINU et ADAPTATIF : il suit la qualite
# reellement atteinte, au lieu des paliers fixes de la version precedente.
#
# Deux points critiques :
#  * le poids est DETACHE du graphe. Sinon le reseau apprendrait a degrader
#    le passe pour alleger le futur -- exactement l'inverse du but.
#  * eps monte par paliers, quand min(w) > DELTA, c'est-a-dire quand toute
#    la fenetre temporelle est effectivement entrainee. C'est le critere
#    propose par les auteurs.
M_CH  = int(os.environ.get("OP_M", 32))
N_PER = max(8, N_COL // M_CH)
EPS_SCHED = [1.0, 3.0, 10.0, 30.0, 100.0]   # eps relatif : cum est ramene a [0,1]
DELTA = float(os.environ.get("OP_DELTA", 0.99))
EDGES = np.linspace(0.0, T_WIN, M_CH+1)
DT_CH = T_WIN / M_CH


def _t_chunks(n):
    """n instants par tranche, tires uniformement DANS leur tranche -> (M_CH, n)."""
    return EDGES[:-1, None] + np.random.uniform(0.0, 1.0, (M_CH, n))*DT_CH


def _grads(r, z, t):
    p = p_of(r, z, t)
    pr = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    pz = torch.autograd.grad(p, z, torch.ones_like(p), create_graph=True)[0]
    return p, pr, pz


def residual_chunks(frac_src=0.20):
    """Residu d'EDP moyen par tranche temporelle -> tenseur (M_CH,)."""
    n_src = int(N_PER*frac_src); n_uni = N_PER - n_src
    r1, z1 = sample_domain(n_uni*M_CH)
    r1 = r1.reshape(M_CH, n_uni); z1 = z1.reshape(M_CH, n_uni)
    # Emphase SPATIALE sur la source ; le temps reste celui de la tranche, et
    # l'enveloppe de Ricker eteint F d'elle-meme sur les tranches tardives.
    r2 = np.clip(np.abs(np.random.normal(0, W_SRC, (M_CH, n_src))), 0, R_EXT)
    z2 = np.clip(np.random.normal(Z_SRC, 2*W_SRC, (M_CH, n_src)), -Z_EXT, -1e-4)
    rn = np.concatenate([r1, r2], 1).ravel()
    zn = np.concatenate([z1, z2], 1).ravel()
    tn = _t_chunks(N_PER).ravel()

    r = Tn(rn, 1); z = Tn(zn, 1); t = Tn(tn, 1)
    p, pr, pz = _grads(r, z, t)
    pt  = torch.autograd.grad(p,  t, torch.ones_like(p),  create_graph=True)[0]
    prr = torch.autograd.grad(pr, r, torch.ones_like(pr), create_graph=True)[0]
    pzz = torch.autograd.grad(pz, z, torch.ones_like(pz), create_graph=True)[0]
    ptt = torch.autograd.grad(pt, t, torch.ones_like(pt), create_graph=True)[0]
    inv_r = torch.where(r < 1e-6, torch.zeros_like(r), 1.0/torch.clamp(r, min=1e-6))
    lap = prr + torch.where(r < 1e-6, prr, pr*inv_r) + pzz

    F = forcing(r, z, t)
    num = ptt + sigma_of(r, z)*pt - C**2*lap - F
    den = FLOOR*((F.abs() + FLOOR)/FLOOR)**ALPHA
    return ((num/den)**2).reshape(M_CH, N_PER).mean(1)


def walls_chunks():
    """Residu de paroi moyen par tranche -> tenseur (M_CH,).

    Les parois portent les MEMES poids causaux : imposer le flux nul a t
    tardif avant d'avoir appris t precoce est exactement la meme violation.
    """
    m = max(4, N_PER//5); S = M_CH*m
    segs = [
        (np.full(S, R_NECK), np.random.uniform(0, L_NECK, S), 1., 0.),      # paroi du col
        (np.random.uniform(R_NECK, R_CAV, S), np.full(S, L_NECK), 0., 1.),  # epaulement
        (np.full(S, R_CAV), np.random.uniform(L_NECK, Z_TOP, S), 1., 0.),   # paroi cavite
        (np.random.uniform(0, R_CAV, S), np.full(S, Z_TOP), 0., 1.),        # fond
        (np.random.uniform(R_NECK, R_EXT, S), np.zeros(S), 0., 1.),         # baffle
    ]
    nseg = 5*m
    rw = np.concatenate([s[0].reshape(M_CH, m) for s in segs], 1).ravel()
    zw = np.concatenate([s[1].reshape(M_CH, m) for s in segs], 1).ravel()
    nx = np.concatenate([np.full((M_CH, m), s[2]) for s in segs], 1).ravel()
    ny = np.concatenate([np.full((M_CH, m), s[3]) for s in segs], 1).ravel()
    tw = _t_chunks(nseg).ravel()
    r = Tn(rw, 1); z = Tn(zw, 1); t = Tn(tw, 1)
    _, pr, pz = _grads(r, z, t)
    gn = (Tn(nx)*pr + Tn(ny)*pz) / (SCALE_P/R_NECK)
    return (gn**2).reshape(M_CH, nseg).mean(1)


# --------------------------------------------------------------------------
# ANCRAGE ANALYTIQUE PRES DE LA SOURCE
# --------------------------------------------------------------------------
# Le residu d'EDP ne peut PAS fixer l'amplitude a lui seul, et ce n'est pas un
# probleme de ponderation mais de MESURE : l'information d'amplitude vit
# uniquement la ou F != 0, soit 0,1 % du domaine espace-temps. Partout ailleurs
# l'equation est homogene, donc sans echelle -- p, 2p et 0 la satisfont
# identiquement. Toute repondering du residu revient a choisir qui sacrifier :
# poids fort sur la source -> la cavite derive ; poids faible -> le zero gagne.
# (Mesure : alpha=0 derive a 3x, alpha=0.3 s'effondre a 1.5e-5 Pa.)
#
# Huang & Alkhalifah (2023) repondent par un terme de perte SEPARE, qui ne se
# fait pas moyenner contre les 99,9 % de lignes qui preferent zero.
#
# Notre valeur d'ancrage est derivee, pas empruntee au FDM. Pres de la source,
# p_tt est negligeable devant c^2 lap(p) : la source fait 4 mm pour une longueur
# d'onde de 1,37 m, le rapport vaut 3e-4. L'equation se reduit a un Poisson,
# lap(p) = -F/c^2, dont la solution pour une gaussienne s'integre exactement :
#
#     p(u,t) = (S0/c^2) * sigma^2 * sqrt(pi/2) * erf(u/sqrt(2))/u * ricker(t)
#
# avec u = |x - x_src|/sigma. Au centre : (S0/c^2)*sigma^2 = 1,088e-3 Pa.
# Verifie contre le FDM a 35 mm : 1,558e-4 predit contre 1,487e-4 mesure, soit
# 4,6 % d'ecart -- exactement la contribution de l'image du baffle.
#
# Deux ponderations, toutes deux physiques :
#  * en espace, exp(-u^2/2) : la formule est exacte au centre et se degrade en
#    s'eloignant, quand le champ re-rayonne par le resonateur devient sensible ;
#  * en temps, |ricker(t)| : on n'affirme la contrainte QUE pendant que le
#    forcage domine. Apres le passage de l'impulsion, le champ pres de la source
#    est celui que le resonateur renvoie, et la formule ne le decrit plus.
R_ANCH = float(os.environ.get("OP_RANCH", 3.0))*W_SRC     # rayon d'ancrage
W_ANCH = float(os.environ.get("OP_WANCH", 20.0))          # poids du bloc
P_ANCH = S0*W_SRC**2/C**2                                 # 1.088e-3 Pa au centre


def p_near(dist, t):
    """Solution de Poisson de la source gaussienne, en champ proche."""
    u = torch.clamp(dist/W_SRC, min=1e-6)
    prof = np.sqrt(np.pi/2)*torch.erf(u/np.sqrt(2.0))/u
    return P_ANCH*prof*ricker_t(t)


def _sample_anchor(n_per):
    """n_per points par tranche, dans une boule de rayon R_ANCH autour de la source."""
    out_r, out_z = [], []
    need = M_CH*n_per
    while need > 0:
        m = int(need*2.5) + 128
        rr = np.random.uniform(0, R_ANCH, m)
        zz = np.random.uniform(Z_SRC - R_ANCH, Z_SRC + R_ANCH, m)
        keep = (rr**2 + (zz - Z_SRC)**2) < R_ANCH**2
        out_r.append(rr[keep]); out_z.append(zz[keep])
        need = M_CH*n_per - sum(len(a) for a in out_r)
    N = M_CH*n_per
    return np.concatenate(out_r)[:N], np.concatenate(out_z)[:N]


def anchor_chunks():
    """Ecart a la solution analytique de champ proche, par tranche -> (M_CH,)."""
    n_per = max(4, N_PER//4)
    rn, zn = _sample_anchor(n_per)
    tn = _t_chunks(n_per).ravel()
    r = Tn(rn); z = Tn(zn); t = Tn(tn)
    dist = torch.sqrt(r**2 + (z - Z_SRC)**2)
    pred = p_of(r, z, t)
    targ = p_near(dist, t)
    u = dist/W_SRC
    wgt = torch.exp(-0.5*u**2) * ricker_t(t).abs()
    return (wgt*((pred - targ)/P_ANCH)**2).reshape(M_CH, n_per).mean(1)


def _design_anchor(rn, zn, tn):
    """Bloc d'ancrage pour les moindres carres : p = A_a . theta, cible p_near.

    Aucune derivee ici -- p est directement lineaire en theta, donc le bloc se
    construit avec les seules features cachees.
    """
    U = torch.stack([torch.as_tensor(rn, dtype=torch.float64),
                     torch.as_tensor(zn, dtype=torch.float64),
                     torch.as_tensor(tn, dtype=torch.float64)], 1)
    h = HEAD(torch.stack([U[:, 0]/R_EXT, (U[:, 1] + Z_EXT)/Z_SPAN, U[:, 2]/T_WIN], 1))
    M1 = torch.cat([h, torch.ones_like(h[:, :1])], 1)
    r = U[:, 0:1]; z = U[:, 1:2]; t = U[:, 2:3]
    blocks = [SCALE_P*q*M1 for q, _, _ in _q_of_t(t)]
    A = torch.cat(blocks, 1)
    dist = torch.sqrt(r**2 + (z - Z_SRC)**2)
    b = p_near(dist, t)
    u = dist/W_SRC
    wgt = torch.sqrt(torch.exp(-0.5*u**2) * ricker_t(t).abs() + 1e-30)
    return A*wgt, b*wgt


# --------------------------------------------------------------------------
# PENALITE DE COMPOSANTE CONTINUE
# --------------------------------------------------------------------------
# Le piege du zero une fois mort, un SECOND minimum trivial prend sa place :
# le champ constant. Un p uniforme en espace et en temps verifie tout --
# p_tt = 0, lap(p) = 0, donc l'equation homogene, et grad(p).n = 0, donc les
# parois. Mesure sur le run precedent : le PINN oscillait entre 0 et 2,4e-4
# au lieu de +/-2,95e-4, avec une moyenne de +1,18e-4 contre 1,47e-6 au FDM,
# et EXACTEMENT la meme moyenne aux deux sondes -- signature d'un socle
# uniforme en espace. Retirer les moyennes faisait tomber le L2 de 86,5 a
# 55,2 %, soit 31 points imputables au seul socle.
#
# On ne peut pas simplement annuler C(x) : cette sortie porte aussi le
# transitoire de remplissage, qui est physique. C'est la MOYENNE TEMPORELLE
# du champ qui doit s'annuler, et c'est demontrable : le Ricker est la derivee
# seconde d'une gaussienne, donc son integrale temporelle est nulle. Aucune
# masse nette n'est injectee, donc aucune pression continue ne subsiste.
#
#        < p(x, .) >_t  =  0     pour tout x
#
# La contrainte est LINEAIRE en theta -- c'est la moyenne des lignes psi --
# donc elle entre aussi dans les moindres carres, la ou l'amplitude se decide.
# Mesure : avec W_DC=10, le L2 brut median passe de 107,8 % a 126,7 % et les
# points sous 100 % tombent de 7/16 a 1/16. Les 31 points que le retrait des
# moyennes faisait gagner etaient une DECOMPOSITION de l'erreur, pas une
# prediction : contraindre la moyenne pendant l'optimisation ne retire pas le
# socle d'une solution acquise, ca fait converger ailleurs -- et plus mal.
# Desactivee par defaut ; le terme reste disponible pour experimenter.
W_DC   = float(os.environ.get("OP_WDC", 0.0))
N_DC   = int(os.environ.get("OP_NDC", 300))      # points d'espace
NT_DC  = int(os.environ.get("OP_NTDC", 3))       # instants par tranche


def _sample_dc(n_x):
    # n_x points d'espace, chacun evalue a M_CH*NT_DC instants couvrant la fenetre
    rx, zx = sample_domain(n_x)
    n_t = M_CH*NT_DC
    r = np.repeat(rx, n_t); z = np.repeat(zx, n_t)
    t = np.tile(_t_chunks(NT_DC).ravel(), n_x)
    return r, z, t, n_t


def dc_loss():
    # moyenne temporelle du champ en chaque point d'espace, rapportee a SCALE_P
    rn, zn, tn, n_t = _sample_dc(N_DC)
    p = p_of(Tn(rn), Tn(zn), Tn(tn)).reshape(N_DC, n_t)
    return torch.mean((p.mean(1)/SCALE_P)**2)


def _design_dc(n_x):
    # bloc de moindres carres : < psi >_t . theta = 0
    rn, zn, tn, n_t = _sample_dc(n_x)
    U = torch.stack([torch.as_tensor(rn, dtype=torch.float64),
                     torch.as_tensor(zn, dtype=torch.float64),
                     torch.as_tensor(tn, dtype=torch.float64)], 1)
    h = HEAD(torch.stack([U[:, 0]/R_EXT, (U[:, 1] + Z_EXT)/Z_SPAN, U[:, 2]/T_WIN], 1))
    M1 = torch.cat([h, torch.ones_like(h[:, :1])], 1)
    t = U[:, 2:3]
    A = torch.cat([SCALE_P*q*M1 for q, _, _ in _q_of_t(t)], 1)
    return A.reshape(n_x, n_t, -1).mean(1)


# --------------------------------------------------------------------------
# COUCHE DE SORTIE PAR MOINDRES CARRES  (arXiv:2504.16553)
# --------------------------------------------------------------------------
# Le piege du zero vient de la descente de gradient : avec un desequilibre de
# 1e8 entre les points de source et les points de champ libre, elle entend
# "lisse p" et n'entend pas "respecte la source", donc elle rabote l'amplitude.
#
# Or l'amplitude, c'est exactement l'echelle de la DERNIERE couche -- et cette
# couche est LINEAIRE. Comme l'operateur d'onde est lui aussi lineaire :
#
#     p = psi(x) . theta        et        L[p] = L[psi](x) . theta
#
# avec theta les poids de sortie. Minimiser ||L[psi].theta - F||^2 sur theta
# est un moindres carres ORDINAIRE, resolu exactement en un coup. Les moindres
# carres trouvent l'optimum GLOBAL en theta : ils ne peuvent pas choisir
# theta = 0 sauf si c'est reellement optimal. Le piege devient inatteignable.
#
# Structure exploitee : dans l'ansatz
#     p = S * g(t) * [ A(x) cos(w0 t) + B(x) sin(w0 t) + C(x) ]
# les coordonnees r et z n'entrent QUE par les features cachees h(r,z,t).
# On n'a donc besoin que de h et de ses derivees ; g, cos et sin sont
# analytiques. Et avec 3 entrees pour HID sorties, le mode FORWARD d'autodiff
# est le bon outil (3 passes pour le jacobien, 9 pour la hessienne).
from torch.func import jacfwd, vmap

HEAD = model.net[:-1]         # (r,z,t) normalises -> h  (HID,)
OUT  = model.net[-1]          # h -> (A, B, C)
D_TH = HID + 1                # une base par neurone cache, plus le biais
N_LS = int(os.environ.get("OP_NLS", 1200))
LS_EVERY = int(os.environ.get("OP_LS_EVERY", 1))
RIDGE = float(os.environ.get("OP_RIDGE", 1e-8))
W_WALL_LS = 2.0
# Amortissement de theta. Chaque iteration resout sur un lot ALEATOIRE NEUF :
# A et b changent, donc le theta optimal change aussi, et il ajuste en partie
# le bruit d'echantillonnage. On moyenne les lots successifs au lieu de les
# suivre :  theta <- (1-beta)*theta_ancien + beta*theta_LS.
BETA = float(os.environ.get("OP_BETA", 0.15))


def _hfun(u):
    """u = (r, z, t) brut -> features cachees h (HID,)."""
    xn = torch.stack([u[0]/R_EXT, (u[1] + Z_EXT)/Z_SPAN, u[2]/T_WIN])
    return HEAD(xn)


_jac  = vmap(jacfwd(_hfun))            # (N,3) -> (N,HID,3)
_hess = vmap(jacfwd(jacfwd(_hfun)))    # (N,3) -> (N,HID,3,3)


def _q_of_t(t):
    """Les trois enveloppes temporelles et leurs deux premieres derivees.

    q1 = g.cos(w0 t)    q2 = g.sin(w0 t)    q3 = g
    avec g(t) = 1 - exp(-(t/tau)^2), qui impose g(0) = g'(0) = 0.
    """
    e = torch.exp(-(t/TAU_G)**2)
    g   = 1.0 - e
    gp  = 2.0*t/TAU_G**2 * e
    gpp = (2.0/TAU_G**2 - 4.0*t**2/TAU_G**4) * e
    c, sn = torch.cos(W0*t), torch.sin(W0*t)
    return [
        (g*c,  gp*c - W0*g*sn,  gpp*c - 2*W0*gp*sn - W0**2*g*c),
        (g*sn, gp*sn + W0*g*c,  gpp*sn + 2*W0*gp*c  - W0**2*g*sn),
        (g,    gp,              gpp),
    ]


def _design(rn, zn, tn):
    """Matrice de conception de l'EDP : A (N, 3*D_TH) et second membre b (N,1).

    Le residu s'ecrit exactement  A.theta - b,  ou theta empile les poids de
    sortie sous la forme (3, D_TH) : ligne k = [w_k, b_k].
    """
    U = torch.stack([torch.as_tensor(rn, dtype=torch.float64),
                     torch.as_tensor(zn, dtype=torch.float64),
                     torch.as_tensor(tn, dtype=torch.float64)], 1)
    h  = HEAD(torch.stack([U[:, 0]/R_EXT, (U[:, 1] + Z_EXT)/Z_SPAN, U[:, 2]/T_WIN], 1))
    J  = _jac(U)                    # (N, HID, 3)
    H2 = _hess(U)                   # (N, HID, 3, 3)
    h_r, h_t = J[:, :, 0], J[:, :, 2]
    h_rr, h_zz, h_tt = H2[:, :, 0, 0], H2[:, :, 1, 1], H2[:, :, 2, 2]

    r = U[:, 0:1]
    inv_r = torch.where(r < 1e-6, torch.zeros_like(r), 1.0/torch.clamp(r, min=1e-6))
    # limite sur l'axe : lap = 2*h_rr + h_zz  (voir la note sur la forme limite)
    lap_h = h_rr + torch.where(r < 1e-6, h_rr, h_r*inv_r) + h_zz

    one  = torch.ones_like(h[:, :1]); zero = torch.zeros_like(one)
    M1   = torch.cat([h,     one ], 1)          # base : [h_j , 1]
    Mt   = torch.cat([h_t,   zero], 1)
    Mtt  = torch.cat([h_tt,  zero], 1)
    Ml   = torch.cat([lap_h, zero], 1)

    t = U[:, 2:3]
    sig = sigma_of(r, U[:, 1:2])
    blocks = []
    for q, qp, qpp in _q_of_t(t):
        blocks.append(SCALE_P*(qpp*M1 + 2.0*qp*Mt + q*Mtt
                               + sig*(qp*M1 + q*Mt) - C**2*q*Ml))
    A = torch.cat(blocks, 1)                    # (N, 3*D_TH)
    b = forcing(r, U[:, 1:2], t)                # (N, 1)
    return A, b, forcing(r, U[:, 1:2], t)


def _design_walls(rn, zn, tn, nx, ny):
    """Matrice de conception des parois : n.grad(p) = A_w . theta, cible 0."""
    U = torch.stack([torch.as_tensor(rn, dtype=torch.float64),
                     torch.as_tensor(zn, dtype=torch.float64),
                     torch.as_tensor(tn, dtype=torch.float64)], 1)
    J = _jac(U)
    h_r, h_z = J[:, :, 0], J[:, :, 1]
    zero = torch.zeros_like(h_r[:, :1])
    Mr = torch.cat([h_r, zero], 1); Mz = torch.cat([h_z, zero], 1)
    nxx = torch.as_tensor(nx, dtype=torch.float64).reshape(-1, 1)
    nyy = torch.as_tensor(ny, dtype=torch.float64).reshape(-1, 1)
    t = U[:, 2:3]
    blocks = [SCALE_P*q*(nxx*Mr + nyy*Mz) for q, _, _ in _q_of_t(t)]
    return torch.cat(blocks, 1)


def ls_step(w_ch):
    """Resout les moindres carres pour la couche de sortie et l'affecte.

    w_ch : poids causaux par tranche (M_CH,), detaches.
    """
    n_per = max(4, N_LS // M_CH)
    n_src = int(n_per*0.20); n_uni = n_per - n_src
    r1, z1 = sample_domain(n_uni*M_CH)
    r1 = r1.reshape(M_CH, n_uni); z1 = z1.reshape(M_CH, n_uni)
    r2 = np.clip(np.abs(np.random.normal(0, W_SRC, (M_CH, n_src))), 0, R_EXT)
    z2 = np.clip(np.random.normal(Z_SRC, 2*W_SRC, (M_CH, n_src)), -Z_EXT, -1e-4)
    rn = np.concatenate([r1, r2], 1).ravel()
    zn = np.concatenate([z1, z2], 1).ravel()
    tn = _t_chunks(n_per).ravel()

    with torch.no_grad():
        A, b, F = _design(rn, zn, tn)
        den = FLOOR*((F.abs() + FLOOR)/FLOOR)**ALPHA
        # ponderation causale : la meme que celle de la descente de gradient
        sw = torch.sqrt(w_ch).repeat_interleave(n_per).reshape(-1, 1)
        A = A*(sw/den); b = b*(sw/den)

        # parois
        m = max(2, n_per//10); S = M_CH*m
        segs = [
            (np.full(S, R_NECK), np.random.uniform(0, L_NECK, S), 1., 0.),
            (np.random.uniform(R_NECK, R_CAV, S), np.full(S, L_NECK), 0., 1.),
            (np.full(S, R_CAV), np.random.uniform(L_NECK, Z_TOP, S), 1., 0.),
            (np.random.uniform(0, R_CAV, S), np.full(S, Z_TOP), 0., 1.),
            (np.random.uniform(R_NECK, R_EXT, S), np.zeros(S), 0., 1.),
        ]
        nseg = 5*m
        rw = np.concatenate([g[0].reshape(M_CH, m) for g in segs], 1).ravel()
        zw = np.concatenate([g[1].reshape(M_CH, m) for g in segs], 1).ravel()
        nx = np.concatenate([np.full((M_CH, m), g[2]) for g in segs], 1).ravel()
        ny = np.concatenate([np.full((M_CH, m), g[3]) for g in segs], 1).ravel()
        tw = _t_chunks(nseg).ravel()
        Aw = _design_walls(rw, zw, tw, nx, ny)
        swW = torch.sqrt(w_ch).repeat_interleave(nseg).reshape(-1, 1)
        Aw = Aw*swW
        rms = lambda X: (X.norm()/np.sqrt(max(X.shape[0], 1))).clamp(min=1e-30)
        # Les moindres carres minimisent la somme des carres de TOUTES les lignes :
        # l'echelle relative des deux blocs decide de l'arbitrage paroi / EDP. Un
        # facteur fixe ne tient pas, car passer de den="local" a den="floor"
        # multiplie les lignes d'EDP par |F|/FLOOR ~ 1.5e4 et noie les parois.
        # On ramene donc chaque bloc a une norme de ligne unitaire, PUIS on
        # applique le poids voulu.
        Aw = Aw * (rms(A)/rms(Aw)) * np.sqrt(W_WALL_LS)

        # bloc d'ancrage : c'est LUI qui fixe l'amplitude, parce qu'il porte
        # un second membre non nul independant du residu d'EDP.
        n_a = max(4, n_per//4)
        ra, za = _sample_anchor(n_a)
        ta = _t_chunks(n_a).ravel()
        Aa, ba = _design_anchor(ra, za, ta)
        swA = torch.sqrt(w_ch).repeat_interleave(n_a).reshape(-1, 1)
        Aa = Aa*swA; ba = ba*swA
        ka = (rms(A)/rms(Aa))*np.sqrt(W_ANCH)
        Aa = Aa*ka; ba = ba*ka

        # bloc de composante continue : < p >_t = 0, cible nulle
        n_d = max(32, n_per*2)
        Ad = _design_dc(n_d)
        Ad = Ad*(rms(A)/rms(Ad))*np.sqrt(W_DC)

        A = torch.cat([A, Aw, Aa, Ad], 0)
        b = torch.cat([b, torch.zeros(Aw.shape[0], 1), ba,
                       torch.zeros(Ad.shape[0], 1)], 0)

        # equations normales + ridge relatif (387 inconnues : resolution triviale)
        AtA = A.T @ A; Atb = A.T @ b
        lam = RIDGE * torch.diagonal(AtA).mean().clamp(min=1e-30)
        theta = torch.linalg.solve(AtA + lam*torch.eye(AtA.shape[0]), Atb)
        theta = theta.reshape(3, D_TH)
        if not torch.isfinite(theta).all():
            return False
        cur = torch.cat([OUT.weight, OUT.bias.reshape(3, 1)], 1)
        mix = (1.0 - BETA)*cur + BETA*theta
        OUT.weight.copy_(mix[:, :HID]); OUT.bias.copy_(mix[:, HID])
    return True


# --------------------------------------------------------------------------
# reference FDM
# --------------------------------------------------------------------------
ref = np.load("data/open_resonator.npz")
t_ref_all = ref["t"]; msk = t_ref_all <= T_WIN + 1e-12
t_ref = t_ref_all[msk]
pc_ref = ref["p_cav"][msk]; po_ref = ref["p_out"][msk]
Z_CAV_PROBE, Z_OUT_PROBE = 0.08, -0.02

def probes():
    with torch.no_grad():
        pc = p_of(Tn(np.zeros_like(t_ref)), Tn(np.full_like(t_ref, Z_CAV_PROBE)), Tn(t_ref)).numpy().ravel()
        po = p_of(Tn(np.zeros_like(t_ref)), Tn(np.full_like(t_ref, Z_OUT_PROBE)), Tn(t_ref)).numpy().ravel()
    return pc, po

def l2(a, b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))


# --------------------------------------------------------------------------
# entrainement
# --------------------------------------------------------------------------
if USE_LS:
    # la couche de sortie appartient aux moindres carres : Adam ne
    # gouverne plus que les couches cachees (schema alterne).
    _p_head = [q for q in model.net[:-1].parameters()]
    opt = torch.optim.Adam(_p_head, lr=LR)
else:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
t0 = time.time()
hist = []

# --------------------------------------------------------------------------
# MOYENNE DE POLYAK  +  SELECTION SUR LA PHYSIQUE
# --------------------------------------------------------------------------
# Deux defauts a corriger.
#
# (1) Le residu est estime par MONTE-CARLO sur un lot aleatoire neuf a chaque
#     iteration, et les moindres carres re-resolvent dessus. Le systeme change
#     donc a chaque coup et theta ajuste en partie le bruit d'echantillonnage :
#     le L2 oscille entre 83 % et 642 % sans que le reseau s'ameliore ou se
#     degrade vraiment. Le remede standard est de moyenner les poids le long
#     de la trajectoire (Polyak) au lieu de garder le dernier iterate.
#
# (2) Choisir le meilleur modele d'apres le L2, ce serait le choisir AVEC la
#     reponse : le L2 se calcule contre le FDM. La selection se fait donc sur
#     la perte PHYSIQUE seule -- EDP + parois + ancrage -- qui n'utilise
#     aucune reference. Le L2 ne sert qu'a RAPPORTER, jamais a decider.
# Mesure sur deux runs de 3000 iterations : la moyenne de Polyak donne un L2
# median 3,83x (sans DC) et 2,27x (avec DC) PIRE que les poids bruts, a chaque
# point de controle. Moyenner les poids d'un reseau non lineaire ne vaut que si
# les iteres tournent autour d'UN minimum ; ici ils sautent entre des solutions
# distinctes, et leur moyenne n'est pas un bon reseau. Desactivee par defaut :
# OP_EMA=0 fait porter selection et sauvegarde sur les poids BRUTS.
EMA_D = float(os.environ.get("OP_EMA", 0.0))


def _ema_init():
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _ema_update(st):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            st[k].mul_(1.0 - EMA_D).add_(v.detach(), alpha=EMA_D)


# Le critere de selection doit etre stable. En le calculant sur un lot
# aleatoire neuf, il herite du bruit de Monte-Carlo et selectionne autant le
# tirage que le modele. On rejoue donc TOUJOURS le meme lot, en fixant la
# graine le temps de l'evaluation -- sans toucher a l'entrainement, et sans
# faire entrer la moindre reference FDM.
VAL_SEED = 12345


def phys_val():
    st = np.random.get_state(); np.random.seed(VAL_SEED)
    v = float((residual_chunks().mean() + 2.0*walls_chunks().mean()
               + W_ANCH*anchor_chunks().mean() + W_DC*dc_loss()).detach())
    np.random.set_state(st)
    return v


def _with_state(st, fn):
    # evalue fn() avec les poids st, puis restaure les poids courants
    cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(st); out = fn(); model.load_state_dict(cur)
    return out


if MODE == "causal":
    ITERS_TOT = int(os.environ.get("OP_ITERS_TOT", 4000))
    CHECK     = int(os.environ.get("OP_CHECK", 100))
    ema = _ema_init(); best_phys = np.inf; best_state = _ema_init()
    ie = 0; EPS = EPS_SCHED[ie]
    print(f"mode CAUSAL | {M_CH} tranches de {DT_CH*1e3:.2f} ms x {N_PER} pts "
          f"| {ITERS_TOT} iterations | eps initial {EPS:.0e}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS_TOT)
    for it in range(1, ITERS_TOT+1):
        opt.zero_grad()
        L  = residual_chunks()
        Lw = walls_chunks()
        La = anchor_chunks()
        with torch.no_grad():
            Ld = L.detach()
            cum = torch.cumsum(Ld, 0) - Ld     # somme des tranches STRICTEMENT anterieures
            # eps RELATIF : l'article suppose des residus deja normalises. Ici
            # ils varient de 1e-2 a 1e7 selon la normalisation choisie, et un
            # eps absolu saturerait exp() -- seule la tranche 0 resterait
            # active. On rapporte donc la somme cumulee au total.
            cum = cum / Ld.sum().clamp(min=1e-30)
            w = torch.exp(-EPS*cum)            # w[0] = 1 par construction
        if USE_LS and it % LS_EVERY == 0:
            if not ls_step(w):
                print(f"  it {it:5d} : moindres carres non finis, pas ignore")
            L  = residual_chunks()             # residu recalcule avec la nouvelle sortie
            Lw = walls_chunks()
            La = anchor_chunks()
        Ldc = dc_loss()
        ((w*L).mean() + 2.0*(w*Lw).mean() + W_ANCH*(w*La).mean()
         + W_DC*Ldc).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if EMA_D > 0: _ema_update(ema)
        if it % CHECK == 0:
            wmin = float(w.min()); lp = float(L.mean().detach()); lw = float(Lw.mean().detach())
            la = float(La.mean().detach())
            # perte physique sur le lot FIXE : aucune reference FDM n'y entre
            phys = phys_val()
            if EMA_D > 0:
                pc_e, po_e = _with_state(ema, probes)
                e_e = l2(pc_e, pc_ref); amp_e = np.abs(pc_e).max()
            else:
                e_e, amp_e = np.nan, np.nan
            pc, po = probes(); e_c = l2(pc, pc_ref); amp = np.abs(pc).max()
            if phys < best_phys:
                best_phys = phys
                src = ema if EMA_D > 0 else model.state_dict()
                best_state = {k: v.detach().clone() for k, v in src.items()}
                flag = " *"
            else:
                flag = "  "
            hist.append((it, EPS, wmin, lp, lw, la, e_c, amp, phys, e_e, amp_e))
            print(f"  it {it:5d} | phys={phys:.3e}{flag}| PDE={lp:.2e} Wall={lw:.1e} "
                  f"Anch={la:.2e} DC={float(Ldc.detach()):.1e} | L2={e_c*100:6.1f} % "
                  f"amp={amp:.3e} (FDM {np.abs(pc_ref).max():.3e}) | {time.time()-t0:5.0f}s")
            if wmin > DELTA and ie < len(EPS_SCHED)-1:
                ie += 1; EPS = EPS_SCHED[ie]
                print(f"        -> fenetre entierement entrainee, eps monte a {EPS:.0e}")
else:
    horizons = [(k+1)*T_WIN/K_PAL for k in range(K_PAL)]
    print(f"mode CURRICULUM | {K_PAL} paliers x {ITERS} it")
    for k, H in enumerate(horizons):
        Hprev = horizons[k-1] if k > 0 else 0.0
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
        for g_ in opt.param_groups: g_["lr"] = LR
        for it in range(1, ITERS+1):
            opt.zero_grad()
            lp = loss_pde(H, Hprev); lw = loss_walls(H)
            (lp + 2.0*lw).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        pc, po = probes()
        e_c = l2(pc, pc_ref); amp = np.abs(pc).max()
        hist.append((H, lp.item(), lw.item(), e_c, amp))
        print(f"  palier {k+1:2d}/{K_PAL} H={H*1e3:5.1f}ms | PDE={lp.item():.3e} Wall={lw.item():.2e} "
              f"| L2 cavite={e_c*100:6.1f} % | max|p|={amp:.2e} (FDM {np.abs(pc_ref).max():.2e}) "
              f"| {time.time()-t0:5.0f}s")

if MODE == "causal":
    # le modele retenu est celui de meilleure perte PHYSIQUE, poids moyennes
    model.load_state_dict(best_state)
    print(f"\nmodele retenu : perte physique = {best_phys:.3e} (selection sans reference)")
torch.save({"state": model.state_dict(), "scale": SCALE_P, "w0": W0, "tau": TAU_G,
            "hid": HID, "T_WIN": T_WIN}, f"models/pinn_open{TAG}.pth")

pc, po = probes()
print("\n=== VERDICT (col ouvert) ===")
print(f"L2 sonde cavite     : {l2(pc, pc_ref)*100:.1f} %")
print(f"L2 sonde exterieure : {l2(po, po_ref)*100:.1f} %")
print(f"amplitude cavite    : {np.abs(pc).max():.3e} Pa   (FDM {np.abs(pc_ref).max():.3e} Pa)")

np.savez(f"data/pinn_open{TAG}.npz", t=t_ref, pc=pc, pc_ref=pc_ref, po=po, po_ref=po_ref,
         hist=np.array(hist), T_WIN=T_WIN)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(t_ref*1e3, pc_ref, "k", lw=1.2, label="FDM")
ax[0].plot(t_ref*1e3, pc, "C1", lw=1.0, label="PINN")
ax[0].set(xlabel="t (ms)", ylabel="p (Pa)", title=f"sonde cavite | L2 = {l2(pc,pc_ref)*100:.1f} %")
ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(t_ref*1e3, po_ref, "k", lw=1.2, label="FDM")
ax[1].plot(t_ref*1e3, po, "C1", lw=1.0, label="PINN")
ax[1].set(xlabel="t (ms)", ylabel="p (Pa)", title=f"sonde exterieure | L2 = {l2(po,po_ref)*100:.1f} %")
ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"plots/pinn_open{TAG}.png", dpi=110)
print(f"Figure : plots/pinn_open{TAG}.png | Modele : models/pinn_open{TAG}.pth")
