"""
Solution manufacturee pour le solveur TRANSITOIRE (verification de code).

Comble une lacune du projet : la MMS du volet frequentiel tourne sur un domaine
rectangulaire a bords DIRICHLET. Elle ne teste donc ni le masque, ni le
traitement des parois rigides. Ici on verifie explicitement :
  * le schema saute-mouton en temps ;
  * le laplacien axisymetrique en VOLUMES FINIS MASQUES ;
  * le traitement de l'AXE r = 0 ;
  * les PAROIS RIGIDES obtenues par annulation du flux (rayon de face = 0).

--------------------------------------------------------------------------
CE QUE CE TEST A REVELE
--------------------------------------------------------------------------
En volumes finis masques, la paroi n'est PAS sur le dernier noeud : elle se
trouve a une DEMI-MAILLE au-dela (c'est la face de la derniere cellule).
Le domaine effectivement simule est donc :

        rayon    R_eff = R + h/2          (une paroi laterale)
        hauteur  H_eff = H + h            (deux parois, une a chaque bout)

Consequence mesuree ci-dessous : si la solution manufacturee annule sa derivee
en R (geometrie nominale), l'ordre observe tombe a 1 -- le decalage de h/2 est
une erreur geometrique du premier ordre. En la calant sur R + h/2, on retrouve
l'ordre 2 attendu, avec des erreurs ~27x plus faibles.

Ce n'est pas un defaut du solveur : c'est sa convention. Mais elle implique un
BIAIS GEOMETRIQUE en 1/h sur les dimensions effectives, dont il faut tenir
compte quand on compare une frequence propre a une theorie analytique.

--------------------------------------------------------------------------
Solution manufacturee, cylindre plein, toutes parois rigides :

    p(r,z,t) = J0(lambda*r) * cos(kz*(z + h/2)) * T(t)
    lambda = j / R_eff   avec J1(j) = 0     ->  dp/dr = 0 en r = R_eff
    kz     = n*pi / H_eff                   ->  dp/dz = 0 en z = -h/2 et H+h/2

    laplacien(p) = -(lambda^2 + kz^2) * p          (exact)
    source F     = p_tt - c^2 * laplacien(p)

T(t) = u^3 exp(-u), u = t/tau, avec tau = 1/(c*K) : cale sur l'echelle de temps
du probleme, pour que T'' et c^2 K^2 T soient du meme ordre (sinon la solution
est une quasi-annulation de deux grands termes et la constante d'erreur explose).
T(0) = T'(0) = T''(0) = 0 : le demarrage au repos du saute-mouton est exact a
l'ordre 3, donc sous l'erreur du schema.

Sortie : data/mms_transient.npz, plots/mms_transient.png
"""
import os, time
import numpy as np
from scipy.special import j0, jn_zeros
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)

C = 343.0
R_DOM, H_DOM = 0.04, 0.12
N_MODE = 2
CFL = 0.40
J1Z = jn_zeros(1, 1)[0]


def run(h, half_cell):
    """Un calcul. half_cell=True : solution calee sur la geometrie EFFECTIVE."""
    R_eff = R_DOM + (0.5*h if half_cell else 0.0)
    H_eff = H_DOM + (1.0*h if half_cell else 0.0)
    z_off = 0.5*h if half_cell else 0.0

    LAM = J1Z / R_eff
    KZ = N_MODE*np.pi / H_eff
    K2 = LAM**2 + KZ**2
    TAU = 1.0/(C*np.sqrt(K2))
    T_END = 4.0*TAU

    Nr = int(round(R_DOM/h)) + 1
    Nz = int(round(H_DOM/h)) + 1
    r = np.arange(Nr)*h; z = np.arange(Nz)*h
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    dom = np.ones((Nr, Nz), bool)

    # machinerie de flux masques, identique a fdm_transient_reference.py
    in_rp = np.zeros_like(dom); in_rp[:-1, :] = dom[1:, :] & dom[:-1, :]
    in_rm = np.zeros_like(dom); in_rm[1:, :]  = dom[:-1, :] & dom[1:, :]
    in_zp = np.zeros_like(dom); in_zp[:, :-1] = dom[:, 1:] & dom[:, :-1]
    in_zm = np.zeros_like(dom); in_zm[:, 1:]  = dom[:, :-1] & dom[:, 1:]
    rp_face = np.where(in_rp, (np.arange(Nr)[:, None] + 0.5)*h, 0.0)
    rm_face = np.where(in_rm, (np.arange(Nr)[:, None] - 0.5)*h, 0.0)
    r_col = np.where(r > 0, r, 1.0)[:, None]

    def lap(p):
        a = np.zeros_like(p); a[:-1, :] = p[1:, :] - p[:-1, :]
        b = np.zeros_like(p); b[1:, :]  = p[1:, :] - p[:-1, :]
        rad = (rp_face*a - rm_face*b) / (r_col*h*h)
        rad[0, :] = np.where(in_rp[0, :], 4.0*(p[1, :] - p[0, :])/(h*h), 0.0)
        ap = np.where(in_zp, np.roll(p, -1, axis=1) - p, 0.0)
        am = np.where(in_zm, p - np.roll(p, 1, axis=1), 0.0)
        return (rad + (ap - am)/(h*h)) * dom

    shape = j0(LAM*RR) * np.cos(KZ*(ZZ + z_off))
    T   = lambda t: (t/TAU)**3 * np.exp(-t/TAU)
    Tpp = lambda t: (6*(t/TAU) - 6*(t/TAU)**2 + (t/TAU)**3)*np.exp(-t/TAU)/TAU**2

    dt = CFL*h/(C*np.sqrt(2.0))
    Nt = int(np.ceil(T_END/dt)); dt = T_END/Nt
    p_old = np.zeros((Nr, Nz)); p_cur = np.zeros((Nr, Nz))
    for n in range(Nt):
        tn = n*dt
        p_new = 2.0*p_cur - p_old + (C*dt)**2*lap(p_cur) \
                + dt*dt*shape*(Tpp(tn) + C**2*K2*T(tn))
        p_old, p_cur = p_cur, p_new

    ex = shape*T(T_END)
    w = np.maximum(RR, 0.5*h)
    err = np.sqrt(np.sum(w*(p_cur - ex)**2)/np.sum(w))
    return Nt, err


HS = [2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4]
res = {}
print("=== MMS TRANSITOIRE : role de la convention de paroi ===\n")
for half_cell in (False, True):
    errs = []
    for h in HS:
        t0 = time.time()
        Nt, e = run(h, half_cell)
        errs.append(e)
    errs = np.array(errs)
    orders = np.log(errs[:-1]/errs[1:])/np.log(2.0)
    res["eff" if half_cell else "nom"] = (errs, orders)
    lab = "geometrie EFFECTIVE (paroi a R + h/2)" if half_cell else \
          "geometrie NOMINALE  (paroi a R)"
    print(f"{lab}")
    for h, e in zip(HS, errs):
        print(f"    h = {h*1e3:5.2f} mm   erreur L2 = {e:.4e}")
    print(f"    ordres observes : {'  '.join(f'{o:.3f}' for o in orders)}"
          f"   ->  moyenne {orders.mean():.3f}\n")

en, on_ = res["nom"]; ee, oe = res["eff"]
print("=== CONCLUSION ===")
print(f"  nominale  : ordre {on_.mean():.2f}  -> le decalage de h/2 domine (erreur d'ordre 1)")
print(f"  effective : ordre {oe.mean():.2f}  -> schema VERIFIE a l'ordre 2")
print(f"  gain sur l'erreur a h = 2 mm : facteur {en[0]/ee[0]:.0f}")

os.makedirs("data", exist_ok=True); os.makedirs("plots", exist_ok=True)
np.savez("data/mms_transient.npz", h=np.array(HS), err_nominal=en,
         err_effective=ee, orders_nominal=on_, orders_effective=oe)

fig, ax = plt.subplots(figsize=(6.8, 4.6))
hh = np.array(HS)*1e3
ax.loglog(hh, en, "s--", color="#a8442a", label=f"paroi en R (nominal) — ordre {on_.mean():.2f}")
ax.loglog(hh, ee, "o-", color="#1d5480", label=f"paroi en R + h/2 — ordre {oe.mean():.2f}")
ax.loglog(hh, ee[0]*(hh/hh[0])**2, "k:", lw=1, label="pente 2")
ax.loglog(hh, en[0]*(hh/hh[0])**1, ":", color="0.6", lw=1, label="pente 1")
ax.set_xlabel("pas h (mm)"); ax.set_ylabel("erreur L2 (Pa)")
ax.set_title("MMS transitoire — la convention de paroi fixe l'ordre")
ax.grid(True, which="both", ls=":"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("plots/mms_transient.png", dpi=110)
print("\nFigure : plots/mms_transient.png | Donnees : data/mms_transient.npz")
