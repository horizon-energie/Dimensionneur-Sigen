import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from excel_generator import get_catalog, generate_workbook_bytes

# ----------------------------------------------------
# CONFIG STREAMLIT
# ----------------------------------------------------
st.set_page_config(
    page_title="Dimensionneur Solaire Sigen",
    layout="wide",
)

# ----------------------------------------------------
# CATALOGUE
# ----------------------------------------------------
PANELS, INVERTERS, BATTERIES = get_catalog()
PANEL_IDS = [p[0] for p in PANELS]


# ----------------------------------------------------
# FONCTIONS CATALOGUE
# ----------------------------------------------------
def get_panel_elec(panel_id: str):
    for p in PANELS:
        if p[0] == panel_id:
            return {
                "id": p[0],
                "Pstc": float(p[1]),
                "Voc": float(p[2]),
                "Vmp": float(p[3]),
                "Isc": float(p[4]),
                "alpha_V": float(p[6]),  # %/°C
            }
    return None


def get_inverter_elec(inv_id: str):
    for inv in INVERTERS:
        # (ID, P_AC_nom, P_DC_max, V_MPP_min, V_MPP_max,
        #  V_DC_max, I_MPPT, Nb_MPPT, Type_reseau, Famille, V_nom_dc)
        if inv[0] == inv_id:
            return {
                "id": inv[0],
                "P_ac": float(inv[1]),
                "P_dc_max": float(inv[2]),
                "Vmpp_min": float(inv[3]),
                "Vmpp_max": float(inv[4]),
                "Vdc_max": float(inv[5]),
                "Impp_max": float(inv[6]),
                "nb_mppt": int(inv[7]),
                "type_reseau": inv[8],
                "famille": inv[9],
                "V_nom_dc": float(inv[10]),
            }
    return None


# ----------------------------------------------------
# PROFILS CONSOMMATION / PRODUCTION
# ----------------------------------------------------
def monthly_pv_profile_kwh_kwp():
    """Profil mensuel PV Belgique (kWh/an/kWc)."""
    annual_kwh_kwp = 1034.0
    distribution = np.array([3.8, 5.1, 8.7, 11.5, 12.1, 11.8,
                             11.9, 10.8, 9.7, 7.0, 4.3, 3.3])
    return annual_kwh_kwp * distribution / 100.0


def monthly_consumption_profile(annual_kwh: float, profile: str):
    profiles = {
        "Standard":   [7, 7, 8, 9, 9, 9, 9, 9, 8, 8, 8, 9],
        "Hiver fort": [10,10,10, 9, 8, 7, 6, 6, 7, 8, 9,10],
        "Été fort":   [6, 6, 7, 8, 9,10,11,11,10, 8, 7, 7],
    }
    arr = np.array(profiles[profile], dtype=float)
    arr = arr / arr.sum()
    return annual_kwh * arr


def hourly_profile(profile_name: str):
    """Profil de consommation horaire (24 valeurs qui somment à 1)."""
    if profile_name == "Uniforme":
        return np.ones(24) / 24

    if profile_name == "Classique (matin + soir)":
        prof = np.array([
            0.02,0.02,0.02,0.02,0.02,
            0.04,0.06,0.08,0.06,0.03,
            0.02,0.02,0.02,0.02,0.03,
            0.04,0.06,0.08,0.07,0.04,
            0.02,0.01,0.01,0.01
        ])
        return prof / prof.sum()

    if profile_name == "Travail journée (soir fort)":
        prof = np.array([
            0.01,0.01,0.01,0.01,0.01,
            0.02,0.03,0.03,0.03,0.02,
            0.01,0.01,0.01,0.01,0.02,
            0.04,0.07,0.09,0.10,0.10,
            0.05,0.02,0.01,0.01
        ])
        return prof / prof.sum()

    if profile_name == "Télétravail":
        prof = np.array([
            0.02,0.02,0.03,0.03,0.03,
            0.04,0.05,0.06,0.06,0.06,
            0.05,0.05,0.05,0.05,0.05,
            0.05,0.05,0.06,0.06,0.06,
            0.05,0.03,0.02,0.02
        ])
        return prof / prof.sum()

    return np.ones(24) / 24


# ----------------------------------------------------
# OPTIMISATION DES STRINGS
# ----------------------------------------------------
def get_nominal_dc_voltage(inverter: dict) -> float:
    """Tension DC nominale issue des fiches techniques Sigen."""
    if "V_nom_dc" in inverter and inverter["V_nom_dc"] > 0:
        return inverter["V_nom_dc"]

    grid = inverter["type_reseau"]
    if grid == "Mono":
        return 350.0
    if grid == "Tri 3x230":
        return 360.0
    if grid == "Tri 3x400":
        return 600.0

    # fallback
    return 0.5 * (inverter["Vmpp_min"] + inverter["Vmpp_max"])


def optimize_strings(
    N_tot: int,
    panel: dict,
    inverter: dict,
    T_min: float,
    T_max: float,
    ratio_dc_ac_target: float = 1.35,
    ratio_dc_ac_min: float = 0.80,
    ratio_dc_ac_max: float = 2.00,
):
    """
    Optimisation automatique des strings (générique pour tous les onduleurs) :

    - 0 ou 1 string par MPPT (conforme fiches Sigen).
    - Longueurs de strings éventuellement différentes sur chaque MPPT.
    - Chaque string doit vérifier :
        * Voc_froid <= Vdc_max
        * Vmp_chaud dans [Vmpp_min, Vmpp_max]
    - Le total de modules utilisés <= N_tot.
    - Le ratio DC/AC doit être dans [ratio_dc_ac_min, ratio_dc_ac_max].
    - La fonction renvoie la meilleure combinaison selon :
        * Nombre de modules utilisés (max)
        * Nombre de MPPT utilisés (max)
        * Tension moyenne des strings proche de la tension DC nominale
        * Ratio DC/AC proche de ratio_dc_ac_target
    """

    Voc = panel["Voc"]
    Vmp = panel["Vmp"]
    Isc = panel["Isc"]
    alpha_V = panel["alpha_V"] / 100.0  # %/°C -> 1/°C
    Pstc = panel["Pstc"]

    Vdc_max = inverter["Vdc_max"]
    Vmpp_min = inverter["Vmpp_min"]
    Vmpp_max = inverter["Vmpp_max"]
    Impp_max = inverter["Impp_max"]
    nb_mppt = inverter["nb_mppt"]
    P_ac = inverter["P_ac"]
    P_dc_max = inverter.get("P_dc_max", 1e9)

    voc_factor_cold = (1 + alpha_V * (T_min - 25.0))
    vmp_factor_hot = (1 + alpha_V * (T_max - 25.0))

    if voc_factor_cold <= 0 or vmp_factor_hot <= 0:
        return None

    # Courant : 1 string par MPPT => courant = Isc
    if Isc > Impp_max:
        return None

    # Bornes sur le nombre de modules en série (même si on autorise des strings de longueur différente)
    # - Limite par Vdc_max (Voc froid)
    # - Limite par plage MPPT (Vmp chaud)
    # - Minimum 3 modules pour éviter des tensions ridicules
    Vnom = get_nominal_dc_voltage(inverter)

    # borne max par Voc froid
    N_series_max_voc = math.floor(Vdc_max / (Voc * voc_factor_cold))
    # borne min / max par Vmp chaud
    if Vmp * vmp_factor_hot > 0:
        N_series_min_vmp = math.ceil(Vmpp_min / (Vmp * vmp_factor_hot))
        N_series_max_vmp = math.floor(Vmpp_max / (Vmp * vmp_factor_hot))
    else:
        N_series_min_vmp = 1
        N_series_max_vmp = N_series_max_voc

    N_series_min = max(3, N_series_min_vmp)
    N_series_max = min(N_series_max_voc, N_series_max_vmp)

    if N_series_min > N_series_max:
        return None

    best = None
    best_score = -1e9

    # Pré-calcul des tensions pour toutes les longueurs possibles
    def vmp_hot_for(L):
        return L * Vmp * vmp_factor_hot

    def voc_cold_for(L):
        return L * Voc * voc_factor_cold

    # Génération récursive de toutes les combinaisons de lengths par MPPT
    def search(mppt_index, remaining_modules, lengths):
        nonlocal best, best_score

        if mppt_index == nb_mppt:
            # Fin : on évalue la configuration si au moins un string est utilisé
            N_used = sum(lengths)
            if N_used == 0:
                return

            P_dc = N_used * Pstc
            if P_dc > P_dc_max:
                return

            ratio_dc_ac = P_dc / P_ac
            if not (ratio_dc_ac_min <= ratio_dc_ac <= ratio_dc_ac_max):
                return

            # Score : on maximise N_used, le nombre de MPPT utilisés,
            # et on rapproche la tension moyenne de la tension nominale
            used_lengths = [L for L in lengths if L > 0]
            n_used_mppt = len(used_lengths)

            if n_used_mppt == 0:
                return

            vmp_mean = sum(vmp_hot_for(L) for L in used_lengths) / n_used_mppt
            score = (
                1000 * N_used
                + 100 * n_used_mppt
                - 2.0 * abs(vmp_mean - Vnom)
                - 50.0 * abs(ratio_dc_ac - ratio_dc_ac_target)
            )

            if score > best_score:
                # On choisit N_series_main comme la longueur la + proche de la tension nominale
                idx_best = min(
                    range(len(used_lengths)),
                    key=lambda i: abs(vmp_hot_for(used_lengths[i]) - Vnom)
                )
                N_series_main = used_lengths[idx_best]

                best = {
                    "strings": lengths[:],
                    "N_used": N_used,
                    "N_series_main": N_series_main,
                    "P_dc": P_dc,
                    "ratio_dc_ac": ratio_dc_ac,
                }
                best_score = score

            return

        # Choix de la longueur de string pour ce MPPT : 0 (non utilisé) ou L dans [N_series_min, N_series_max]
        # avec contrainte de ne pas dépasser N_tot
        # 0 : MPPT non utilisé
        search(mppt_index + 1, remaining_modules, lengths + [0])

        # Strings actifs
        for L in range(N_series_min, N_series_max + 1):
            if L > remaining_modules:
                break

            # Vérif électrique immédiate pour ce string
            if voc_cold_for(L) > Vdc_max:
                continue
            Vmp_hot_L = vmp_hot_for(L)
            if not (Vmpp_min <= Vmp_hot_L <= Vmpp_max):
                continue

            search(mppt_index + 1, remaining_modules - L, lengths + [L])

    search(0, N_tot, [])

    return best

# ----------------------------------------------------
# CHOIX AUTOMATIQUE DU MEILLEUR ONDULEUR
# ----------------------------------------------------
def select_best_inverter(
    panel: dict,
    n_panels: int,
    grid_type: str,
    max_dc_ac: float,
    fam_pref: str | None,
    T_min: float,
    T_max: float,
):
    """
    Parcourt tous les onduleurs compatibles (type réseau + famille éventuelle),
    optimise les strings pour chacun, et choisit celui qui :

    - respecte le type de réseau + famille (Store / Hybride),
    - respecte P_dc <= P_DC_max,
    - respecte le ratio DC/AC <= max_dc_ac (slider utilisateur, ex: 1.35),
    - maximise la puissance DC installée.

    Attention : c'est seulement pour la sélection AUTO.
    Ensuite, lorsqu'un onduleur est choisi (auto ou manuel), on recalcule
    les strings avec un ratio DC/AC physique plus large (0.8–2.0).
    """
    best = None
    best_score = -1e9

    for inv in INVERTERS:
        # (ID, P_AC_nom, P_DC_max, V_MPP_min, V_MPP_max,
        #  V_DC_max, I_MPPT, Nb_MPPT, Type_reseau, Famille, V_nom_dc)
        inv_id, p_ac, p_dc_max, vmin, vmax, vdcmax, imppt, nb_mppt, inv_type, inv_family, v_nom_dc = inv

        if inv_type != grid_type:
            continue
        if fam_pref is not None and inv_family != fam_pref:
            continue

        inv_elec = get_inverter_elec(inv_id)
        if inv_elec is None:
            continue

        opt = optimize_strings(
            N_tot=n_panels,
            panel=panel,
            inverter=inv_elec,
            T_min=T_min,
            T_max=T_max,
            ratio_dc_ac_min=0.8,
            ratio_dc_ac_max=max_dc_ac,  # borne issue du slider
        )
        if opt is None:
            continue

        P_dc = opt["P_dc"]
        ratio = P_dc / p_ac

        # Respect P_DC_max
        if P_dc > p_dc_max:
            continue

        score = P_dc  # on maximise la puissance DC installée

        if score > best_score:
            best_score = score
            best = {
                "inv_id": inv_id,
                "opt": opt,
                "P_dc": P_dc,
                "ratio": ratio,
                "P_ac": p_ac,
            }

    return best


# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### 🔧 Paramètres généraux")

    # Sélection panneau
    panel_id = st.selectbox("Panneau", options=PANEL_IDS, index=0)
    n_modules = st.number_input("Nombre de panneaux", min_value=3, max_value=100, value=12)

    panel_elec = get_panel_elec(panel_id)
    if panel_elec is None:
        st.error("Panneau introuvable dans le catalogue.")
        st.stop()

    # Type réseau
    grid_type = st.selectbox("Type de réseau", options=["Mono", "Tri 3x230", "Tri 3x400"], index=0)

    # Mode Store / Hybride
    sigenstore_mode = st.selectbox(
        "Installation compatible SigenStore ?",
        options=["Auto", "Oui (Store)", "Non (Hybride)"],
        index=0,
    )
    if sigenstore_mode == "Oui (Store)":
        fam_pref = "Store"
    elif sigenstore_mode == "Non (Hybride)":
        fam_pref = "Hybride"
    else:
        fam_pref = None

    # Ratio DC/AC (pour la sélection auto uniquement)
    max_dc_ac = st.slider("Ratio DC/AC max (sélection auto)", min_value=1.0, max_value=2.0, value=1.35, step=0.01)

    # Batterie
    battery_enabled = st.checkbox("Batterie", value=False)
    if battery_enabled:
        battery_kwh = st.slider("Capacité batterie (kWh)", 6.0, 50.0, 6.0, 0.5)
    else:
        battery_kwh = 0.0

    st.markdown("---")
    st.markdown("### Profil de consommation")

    annual_consumption = st.number_input("Conso annuelle (kWh)", 500, 20000, 3500, 100)
    consumption_profile = st.selectbox("Profil mensuel", ["Standard", "Hiver fort", "Été fort"], 0)

    hourly_profile_choice = st.selectbox(
        "Profil horaire",
        ["Uniforme", "Classique (matin + soir)", "Travail journée (soir fort)", "Télétravail"],
        index=1
    )

    month_for_hours = st.slider("Mois pour le profil horaire", 1, 12, 6)

    st.markdown("---")
    st.markdown("### Températures de calcul")
    t_min = st.number_input("Température min (°C)", -30, 10, -10)
    t_max = st.number_input("Température max (°C)", 30, 90, 70)

    st.markdown("---")
    st.markdown("### Choix de l’onduleur")

    best = select_best_inverter(
        panel=panel_elec,
        n_panels=int(n_modules),
        grid_type=grid_type,
        max_dc_ac=float(max_dc_ac),
        fam_pref=fam_pref,
        T_min=float(t_min),
        T_max=float(t_max),
    )

    if best is None:
        st.error("Aucun onduleur compatible trouvé avec cette configuration (en sélection auto).")
        st.stop()

    auto_inv_id = best["inv_id"]

    compatible_inv = [
        inv[0] for inv in INVERTERS
        if inv[8] == grid_type and (fam_pref is None or inv[9] == fam_pref)
    ]

    inv_options = [f"(Auto) {auto_inv_id}"] + compatible_inv

    selected_inv_label = st.selectbox("Onduleur", inv_options, index=0)

    if selected_inv_label.startswith("(Auto)"):
        inverter_id = auto_inv_id
    else:
        inverter_id = selected_inv_label

import numpy as np

def generate_pv_profile_hourly(pv_monthly):
    """
    Génère une production PV horaire sur 8760h à partir des productions mensuelles.
    Profil irradiance type (normalisé) issu d'un PV moyen européen.
    """
    # Profil de production typique sur une journée
    pv_day_profile = np.array([
        0,0,0,0,0,
        0.01,0.04,0.09,0.14,0.18,0.20,0.18,
        0.14,0.10,0.06,0.03,0.01,
        0,0,0,0,0,0,0
    ])
    pv_day_profile /= pv_day_profile.sum()

    # Nombre d'heures par mois
    hours_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24,
                   31*24, 31*24, 30*24, 31*24, 30*24, 31*24]

    pv_hourly = []
    for m in range(12):
        days = hours_month[m] // 24
        prod_day = pv_monthly[m] / days  # kWh/jour
        day_profile = pv_day_profile * prod_day
        pv_hourly.extend(list(day_profile) * days)

    return np.array(pv_hourly)  # 8760 valeurs


def generate_consumption_hourly(cons_monthly, cons_frac):
    """
    Distribue la consommation mensuelle sur un profil horaire utilisateur.
    """
    hours_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24,
                   31*24, 31*24, 30*24, 31*24, 30*24, 31*24]

    cons_hourly = []
    for m in range(12):
        days = hours_month[m] // 24
        cons_day = cons_monthly[m] / days
        day_profile = cons_frac * cons_day
        cons_hourly.extend(list(day_profile) * days)

    return np.array(cons_hourly)  # 8760 valeurs


def simulate_battery_hourly(
    pv_hourly,
    cons_hourly,
    battery_capacity_kwh,
    charge_eff=0.95,
    discharge_eff=0.95,
    max_charge_power_kw=3.6,   # 16A × 230 V typique
    max_discharge_power_kw=3.6
):
    """
    Simulation physique précise :
    - 8760 heures
    - SOC persistant
    - charge/décharge limitées en puissance
    - rendement charge/décharge
    """

    hours = len(pv_hourly)
    soc = 0.0
    soc_series = np.zeros(hours)
    ac_direct = np.zeros(hours)
    ac_batt = np.zeros(hours)
    grid_export = np.zeros(hours)
    grid_import = np.zeros(hours)

    for h in range(hours):

        prod = pv_hourly[h]   # kWh
        conso = cons_hourly[h]  # kWh

        # Autoconsommation directe
        direct = min(prod, conso)
        ac_direct[h] = direct

        surplus = prod - direct
        deficit = conso - direct

        # Puissance max charge/décharge → conversion en énergie horaire
        max_charge_kwh = max_charge_power_kw  # sur 1h
        max_discharge_kwh = max_discharge_power_kw  # sur 1h

        # Charge batterie
        charge_possible = min(surplus, max_charge_kwh)
        charge_effective = charge_possible * charge_eff
        soc = min(battery_capacity_kwh, soc + charge_effective)

        # Décharge batterie
        discharge_possible = min(deficit, max_discharge_kwh)
        discharge_effective = min(discharge_possible / discharge_eff, soc)

        ac_batt[h] = discharge_effective * discharge_eff
        soc -= discharge_effective

        # Flux réseau
        grid_export[h] = surplus - charge_possible
        grid_import[h] = deficit - ac_batt[h]

        soc_series[h] = soc

    return soc_series, ac_direct, ac_batt, grid_export, grid_import

# ----------------------------------------------------
# CALCULS PRINCIPAUX
# ----------------------------------------------------
inv_elec = get_inverter_elec(inverter_id)
if inv_elec is None:
    st.error("Spécifications onduleur introuvables.")
    st.stop()

# Ici, on NE LIMITE PLUS par le slider utilisateur, mais par la physique (ratio <= 2.0)
opt_result = optimize_strings(
    N_tot=int(n_modules),
    panel=panel_elec,
    inverter=inv_elec,
    T_min=float(t_min),
    T_max=float(t_max),
    ratio_dc_ac_min=0.8,
    ratio_dc_ac_max=2.0,  # borne physique, indépendante du slider
)


if opt_result is None:
    st.error(
        f"Aucun câblage valide trouvé pour l'onduleur {inverter_id}. "
        "Vérifiez les températures ou le nombre de modules."
    )
    st.stop()

P_dc = opt_result["P_dc"]
ratio_dc_ac = opt_result["ratio_dc_ac"]
p_dc_kwp = P_dc / 1000.0

# Profils mensuels
pv_kwh_per_kwp = monthly_pv_profile_kwh_kwp()
pv_monthly = pv_kwh_per_kwp * p_dc_kwp

cons_monthly = monthly_consumption_profile(annual_consumption, consumption_profile)

# ----------------------------------------------------
# SIMULATION HORAIRE COMPLETE (PV + conso + batterie)
# ----------------------------------------------------

# Profil consommation horaire (normalisé)
cons_frac = hourly_profile(hourly_profile_choice)

# 1. Génération PV horaire (8760h)
pv_hourly = generate_pv_profile_hourly(pv_monthly)

# 2. Génération consommation horaire (8760h)
cons_hourly = generate_consumption_hourly(cons_monthly, cons_frac)

# 3. Simulation batterie
if battery_enabled and battery_kwh > 0:
    soc, ac_direct_h, ac_batt_h, export_h, import_h = simulate_battery_hourly(
        pv_hourly,
        cons_hourly,
        battery_capacity_kwh=battery_kwh,
        charge_eff=0.95,
        discharge_eff=0.95,
        max_charge_power_kw=3.6,
        max_discharge_power_kw=3.6
    )
else:
    soc = np.zeros_like(pv_hourly)
    ac_direct_h = np.minimum(pv_hourly, cons_hourly)
    ac_batt_h = np.zeros_like(pv_hourly)
    export_h = pv_hourly - ac_direct_h
    import_h = cons_hourly - ac_direct_h

# Agrégation annuelle
pv_year = pv_hourly.sum()
cons_year = cons_hourly.sum()
ac_direct_year = ac_direct_h.sum()
ac_batt_year = ac_batt_h.sum()
ac_total_year = ac_direct_year + ac_batt_year

# Garantir AC ≤ PV
ac_total_year = min(ac_total_year, pv_year)

taux_auto = ac_total_year / pv_year * 100
taux_couv = ac_total_year / cons_year * 100

# ----------------------------------------------------
# EN-TÊTE / METRICS
# ----------------------------------------------------
col_logo, col_title = st.columns([1, 3])

with col_logo:
    if os.path.exists("logo_horizon.png"):
        st.image("logo_horizon.png", use_column_width=True)

with col_title:
    st.title("Dimensionneur Solaire Sigen")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Puissance DC installée", f"{P_dc:.0f} Wc")
    st.metric("Panneaux câblés", f"{opt_result['N_used']} / {int(n_modules)}")

with col2:
    st.metric("Prod PV annuelle", f"{pv_year:.0f} kWh")
    st.metric("Conso annuelle", f"{cons_year:.0f} kWh")

with col3:
    st.metric("Taux autocons. total", f"{taux_auto:.1f} %")
    st.metric("Taux couverture", f"{taux_couv:.1f} %")

with col4:
    st.metric("Onduleur choisi", inverter_id)
    st.metric("Ratio DC/AC réel", f"{ratio_dc_ac:.2f}")
    if battery_capacity_kwh > 0:
        st.metric("Autocons. via batterie", f"{autocons_year_batt:.0f} kWh")

# Avertissement si le ratio dépasse le souhait utilisateur
if ratio_dc_ac > float(max_dc_ac):
    st.warning(
        f"Le câblage retenu a un ratio DC/AC de {ratio_dc_ac:.2f}, "
        f"supérieur au ratio max souhaité ({max_dc_ac:.2f})."
    )

if opt_result["N_used"] < int(n_modules):
    st.warning(
        f"{opt_result['N_used']} panneaux seulement peuvent être câblés proprement sur cet onduleur "
        f"(sur {int(n_modules)} demandés)."
    )

# ----------------------------------------------------
# 🔌 Câblage des strings – affichage
# ----------------------------------------------------
st.markdown("## 🔌 Câblage des strings")

strings = opt_result["strings"]
cols_strings = st.columns(len(strings))

for i, s in enumerate(strings):
    with cols_strings[i]:
        if s > 0:
            st.metric(
                label=f"MPPT {i+1}",
                value=f"{s} modules",
                delta=f"{s * panel_elec['Vmp']:.0f} V"
            )
        else:
            st.metric(label=f"MPPT {i+1}", value="Non utilisé")

# ----------------------------------------------------
# PROFIL MENSUEL
# ----------------------------------------------------
st.markdown("## 📊 Production vs Consommation – Profil mensuel")

df_month = pd.DataFrame({
    "Mois": months_labels,
    "Consommation (kWh)": cons_monthly,
    "Production PV (kWh)": pv_monthly,
    "Autocons. directe (kWh)": autocons_monthly_direct,
    "Autocons. batterie (kWh)": autocons_batt_monthly,
    "Autocons. totale (kWh)": autocons_monthly_total,
})

fig = px.bar(
    df_month,
    x="Mois",
    y=["Consommation (kWh)", "Production PV (kWh)", "Autocons. totale (kWh)"],
    barmode="group",
    labels={"value": "kWh", "variable": ""},
)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(df_month)

# ----------------------------------------------------
# PROFIL HORAIRE – JOUR TYPE
# ----------------------------------------------------
st.markdown("## 🕒 Profil horaire – jour type")

idx = month_for_hours - 1

day_cons = cons_monthly[idx] / days_in_month[idx]
day_pv = pv_monthly[idx] / days_in_month[idx]

cons_hour = day_cons * cons_frac
pv_hour = day_pv * pv_frac

autocons_hour_direct = np.minimum(cons_hour, pv_hour)

df_hour = pd.DataFrame({
    "Heure": np.arange(24),
    "Consommation (kWh)": cons_hour,
    "Production PV (kWh)": pv_hour,
    "Autocons. directe (kWh)": autocons_hour_direct,
})

fig2 = px.line(
    df_hour,
    x="Heure",
    y=["Consommation (kWh)", "Production PV (kWh)", "Autocons. directe (kWh)"],
    markers=True,
    labels={"value": "kWh", "variable": ""},
)
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(df_hour)

# ----------------------------------------------------
# EXPORT EXCEL
# ----------------------------------------------------
st.markdown("## 📥 Export Excel complet")

config = {
    "panel_id": panel_id,
    "n_modules": int(n_modules),
    "grid_type": grid_type,
    "battery_enabled": battery_enabled,
    "battery_kwh": float(battery_kwh),
    "max_dc_ac": float(max_dc_ac),
    "annual_consumption": float(annual_consumption),
    "consumption_profile": consumption_profile,
    "t_min": float(t_min),
    "t_max": float(t_max),
    "n_series": int(opt_result["N_series_main"]),
    "inverter_id": inverter_id,
}

if st.button("Générer l’Excel"):
    xlsx_bytes = generate_workbook_bytes(config)
    st.download_button(
        "Télécharger le fichier Excel",
        data=xlsx_bytes,
        file_name="Dimensionnement_Sigen_Complet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
