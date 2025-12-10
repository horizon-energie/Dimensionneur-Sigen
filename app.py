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
    ratio_dc_ac_min: float = 1.00,
    ratio_dc_ac_max: float = 2.00,
):
    """
    Optimisation automatique des strings :
    - calcule une répartition par MPPT (1 string max par MPPT)
    - strings éventuellement de longueurs différentes
    - au moins un MPPT proche de la tension nominale DC
    - respect des contraintes Voc froid, Vmp chaud, courant MPPT et ratio DC/AC.
    """
    Voc = panel["Voc"]
    Vmp = panel["Vmp"]
    Isc = panel["Isc"]
    alpha_V = panel["alpha_V"] / 100.0
    Pstc = panel["Pstc"]

    Vdc_max = inverter["Vdc_max"]
    Vmpp_min = inverter["Vmpp_min"]
    Vmpp_max = inverter["Vmpp_max"]
    Impp_max = inverter["Impp_max"]
    nb_mppt = inverter["nb_mppt"]
    P_ac = inverter["P_ac"]

    voc_factor_cold = (1 + alpha_V * (T_min - 25.0))
    vmp_factor_hot = (1 + alpha_V * (T_max - 25.0))

    if voc_factor_cold <= 0 or vmp_factor_hot <= 0:
        return None

    Vnom = get_nominal_dc_voltage(inverter)

    # Bornes sur le nombre de modules en série
    N_series_max = math.floor(Vdc_max / (Voc * voc_factor_cold))
    N_series_min = 3

    if N_series_min > N_series_max:
        return None

    # Nombre de modules idéal sur le string "principal"
    if Vnom > 0:
        N_series_ideal = max(
            N_series_min,
            min(N_series_max, int(round(Vnom / (Vmp * vmp_factor_hot))))
        )
    else:
        N_series_ideal = N_series_min

    best = None
    best_score = -1e9

    # On teste tous les N_series possibles
    for N_series_main in range(N_series_min, N_series_max + 1):
        # Vérif électrique de ce string "principal"
        Voc_cold_main = N_series_main * Voc * voc_factor_cold
        Vmp_hot_main = N_series_main * Vmp * vmp_factor_hot

        if Voc_cold_main > Vdc_max:
            continue
        if not (Vmpp_min <= Vmp_hot_main <= Vmpp_max):
            continue

        # Maximum de strings de cette taille
        max_full_strings = N_tot // N_series_main
        if max_full_strings == 0:
            continue

        # 1 string par MPPT
        max_full_strings = min(max_full_strings, nb_mppt)

        for n_full in range(1, max_full_strings + 1):
            used_full = n_full * N_series_main
            remaining = N_tot - used_full

            # Cas 1 : aucun reste exploitable
            possible_remainders = [0]

            # Cas 2 : un string "reste" sur un MPPT libre si suffisamment de modules
            if remaining >= N_series_min and n_full < nb_mppt:
                possible_remainders.append(remaining)

            for rem in possible_remainders:
                strings_sizes = []

                # On met éventuellement le string "reste" en premier MPPT
                if rem > 0:
                    Voc_cold_rem = rem * Voc * voc_factor_cold
                    Vmp_hot_rem = rem * Vmp * vmp_factor_hot
                    if Voc_cold_rem > Vdc_max:
                        continue
                    if not (Vmpp_min <= Vmp_hot_rem <= Vmpp_max):
                        continue

                    strings_sizes.append(rem)

                # Puis les strings "principaux"
                for _ in range(n_full):
                    strings_sizes.append(N_series_main)

                # Complète les MPPT inutilisés
                while len(strings_sizes) < nb_mppt:
                    strings_sizes.append(0)

                # Vérif courant par MPPT (1 string max par MPPT)
                if Isc > Impp_max:
                    continue

                N_used = sum(strings_sizes)
                P_dc = N_used * Pstc
                ratio_dc_ac = P_dc / P_ac

                if ratio_dc_ac < ratio_dc_ac_min or ratio_dc_ac > ratio_dc_ac_max:
                    continue

                # Score :
                # - max modules utilisés
                # - max MPPT utilisés
                # - string principal proche de N_series_ideal
                score = (
                    1000 * N_used
                    + 200 * sum(1 for s in strings_sizes if s > 0)
                    - 50 * abs(N_series_main - N_series_ideal)
                    - 10 * abs(ratio_dc_ac - ratio_dc_ac_target)
                )

                if score > best_score:
                    best_score = score
                    best = {
                        "strings": strings_sizes,       # liste de taille nb_mppt
                        "N_used": N_used,
                        "N_series_main": N_series_main,
                        "P_dc": P_dc,
                        "ratio_dc_ac": ratio_dc_ac,
                    }

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
    - respecte les contraintes
    - respecte le ratio DC/AC <= max_dc_ac (slider utilisateur)
    - respecte P_dc <= P_DC_max
    - maximise la puissance DC installée
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

        # On limite par le slider ici (choix auto)
        opt = optimize_strings(
            N_tot=n_panels,
            panel=panel,
            inverter=inv_elec,
            T_min=T_min,
            T_max=T_max,
            ratio_dc_ac_min=1.0,
            ratio_dc_ac_max=max_dc_ac,
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
    ratio_dc_ac_min=1.0,
    ratio_dc_ac_max=2.0,
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

# Autoconsommation directe (sans batterie) au niveau mensuel
autocons_monthly_direct = np.minimum(pv_monthly, cons_monthly)

months_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                 "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# Profil horaire de consommation (normalisé)
cons_frac = hourly_profile(hourly_profile_choice)

# Profil horaire de production PV type (normalisé)
pv_frac = np.array([
    0, 0, 0, 0, 0,
    0.01, 0.04, 0.07, 0.10, 0.13, 0.14, 0.14,
    0.13, 0.10, 0.07, 0.04, 0.02,
    0, 0, 0, 0, 0, 0, 0,
])
if pv_frac.sum() > 0:
    pv_frac = pv_frac / pv_frac.sum()

# ----------------------------------------------------
# MODÈLE SIMPLE DE BATTERIE (SOC persistant dans le mois)
# ----------------------------------------------------
battery_capacity = float(battery_kwh)  # kWh
autocons_batt_monthly = np.zeros(12)

if battery_capacity > 0:
    for m in range(12):
        days = int(days_in_month[m])
        if days <= 0:
            continue

        pv_day = pv_monthly[m] / days
        cons_day = cons_monthly[m] / days

        pv_h = pv_frac * pv_day
        cons_h = cons_frac * cons_day

        autocons_batt = 0.0

        # SOC persistant sur le mois
        battery_soc = 0.0

        for _ in range(days):

            for h in range(24):
                prod = pv_h[h]
                conso = cons_h[h]

                direct = min(prod, conso)
                surplus = prod - direct
                deficit = conso - direct

                # Charge batterie
                battery_soc = min(battery_soc + surplus, battery_capacity)

                # Décharge batterie
                discharge = min(deficit, battery_soc)
                battery_soc -= discharge

                autocons_batt += discharge

        autocons_batt_monthly[m] = autocons_batt

# Autocons totale
autocons_monthly_total = autocons_monthly_direct + autocons_batt_monthly

pv_year = float(pv_monthly.sum())
cons_year = float(annual_consumption)
autocons_year_direct = float(autocons_monthly_direct.sum())
autocons_year_batt = float(autocons_batt_monthly.sum())
autocons_year = autocons_year_direct + autocons_year_batt

taux_auto = (autocons_year / pv_year * 100) if pv_year > 0 else 0.0
taux_couv = (autocons_year / cons_year * 100) if cons_year > 0 else 0.0

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
    if battery_capacity > 0:
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
