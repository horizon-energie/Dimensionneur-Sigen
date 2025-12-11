# 🌞 Horizon Énergie – Dimensionneur Solaire Sigen
🔗 **Accès direct à l’outil en ligne :**  
https://dimensionneur-sigen-dvsf9uyr5lpcbedjy468qt.streamlit.app

# Mode d’emploi

Ce guide explique :

1. Comment encoder les panneaux, onduleurs et batteries dans `excel_generator.py`  
2. Comment fonctionne les calculs (strings, MPPT, tensions, ratio DC/AC, batterie)  
3. Comment utiliser l’application Streamlit pour dimensionner une installation photovoltaïque Sigen.

---

# 1. Encodage du matériel dans `excel_generator.py`

Tous les panneaux, onduleurs et batteries utilisés par l’application sont définis dans :

```
excel_generator.py
```

La fonction centrale :

```python
def get_catalog():
    panels = [...]
    inverters = [...]
    batteries = [...]
    return panels, inverters, batteries
```

L’application lit automatiquement ce catalogue au lancement.

---

# 2. Ajouter / modifier des panneaux photovoltaïques

Les panneaux sont listés dans :

```python
panels = [
    ["Trina450", 450, 52.9, 44.6, 10.74, 10.09, -0.24],
]
```

Format obligatoire :

| Champ | Description |
|-------|-------------|
| 0 | ID du panneau |
| 1 | Puissance STC (W) |
| 2 | Voc (V) |
| 3 | Vmp (V) |
| 4 | Isc (A) |
| 5 | Imp (A) |
| 6 | αV (%/°C — coefficient température tension) |

### Exemple d’ajout :

```python
["JA550", 550, 49.8, 41.5, 13.10, 12.50, -0.25],
```

---

# 3. Ajouter / modifier des onduleurs Sigen

Les onduleurs sont définis comme :

```python
inverters = [
    (ID, P_AC_nom, P_DC_max, V_MPP_min, V_MPP_max,
     V_DC_max, I_MPPT, Nb_MPPT, Type_reseau, Famille, V_nom_dc)
]
```

Signification des champs :

| # | Champ | Description |
|---|-------|-------------|
| 0 | ID | Nom interne |
| 1 | P_AC_nom | Puissance AC nominale (W) |
| 2 | P_DC_max | Puissance DC admissible (W) |
| 3 | V_MPP_min | Tension minimale MPPT (V) |
| 4 | V_MPP_max | Tension maximale MPPT (V) |
| 5 | V_DC_max | Tension DC max tolérable (V) |
| 6 | I_MPPT | Courant max par MPPT (A) |
| 7 | Nb_MPPT | Nombre d'entrées MPPT |
| 8 | Type_reseau | Mono, Tri 3x230 ou Tri 3x400 |
| 9 | Famille | Hybride / Store |
| 10 | V_nom_dc | Tension DC nominale idéale (V) |

### Exemple d’ajout :

```python
("Store5.0Mono", 5000, 10000, 80, 550, 600, 16, 2, "Mono", "Store", 350),
```

---

# 4. Ajouter / modifier une batterie

Les batteries sont listées dans :

```python
batteries = [
    ["Sigen6", 6],
    ["Sigen10", 10],
]
```

Chaque entrée = `[ID, capacité_kWh]`.

---

# 5. Fonctionnement interne du dimensionnement

L’algorithme teste des combinaisons complètes de strings et de MPPT.

## 5.1. Calcul des longueurs de strings

Pour chaque onduleur, l’application vérifie :

- **Voc froid** ≤ `V_DC_max`  
- **Vmp chaud** ∈ `[V_MPP_min, V_MPP_max]`  
- **Courant de string** ≤ `I_MPPT`  
- Un seul string par MPPT (conforme aux fiches Sigen)  
- Utilisation maximale des modules disponibles  
- Tension DC proche de la tension nominale `V_nom_dc`  
- Ratio DC/AC dans un intervalle cohérent (ex. 0.8–2.0)  

L’algorithme :

1. Calcule l’intervalle possible de modules en série (N_min, N_max)  
2. Génère toutes les longueurs de strings admissibles sur chaque MPPT  
3. Explore toutes les combinaisons possibles (0 ou 1 string par MPPT)  
4. Élimine celles qui ne respectent pas les contraintes électriques  
5. Calcule, pour chaque configuration valide :  
   - nombre total de panneaux utilisés  
   - P_DC totale  
   - ratio DC/AC = P_DC / P_AC_nom  
   - tension moyenne des strings vs `V_nom_dc`  
6. Choisit la configuration avec le meilleur score (max panneaux, max MPPT utilisés, tension proche nominale, ratio proche de la cible).

Résultat enregistré :

- `strings`: liste des nombres de panneaux par MPPT (ex. `[13, 13]`)  
- `N_used`: nombre total de panneaux câblés  
- `N_series_main`: longueur de string “typique” / principale  
- `P_dc`: puissance DC totale (W)  
- `ratio_dc_ac`: ratio P_DC / P_AC_nom  

---

# 6. Simulation horaire complète (8760 h)

L’application modélise une année complète heure par heure (8760 points).

## 6.1. Production PV (8760 valeurs)

1. À partir de la puissance installée (kWc), l’appli calcule un profil mensuel type (Belgique).  
2. Ce profil mensuel est redistribué sur des journées types via un profil PV horaire normalisé (lever / midi / coucher du soleil).  
3. On obtient un vecteur `pv_hourly` de longueur 8760.

## 6.2. Consommation horaire

1. L’utilisateur saisit une **consommation annuelle** (kWh).  
2. Il choisit un **profil mensuel**.  
3. Il choisit un **profil horaire**.  
4. Ces profils sont combinés pour générer un vecteur `cons_hourly` (8760 valeurs).

## 6.3. Batterie

La batterie est simulée physiquement :

- SOC persistant (pas remis à zéro artificiellement)  
- Charge si `PV > conso`  
- Décharge si `conso > PV`  
- Puissance max de charge/décharge  
- Rendement charge/décharge  
- Aucune situation où autoconsommation > production  

---

# 7. Utilisation de l’application Streamlit

## 7.1. Choix du panneau

- Sélectionner un modèle de panneau  
- Indiquer le nombre total de panneaux

## 7.2. Réseau & famille d’onduleur

- Choisir : Mono / Tri 3x230 / Tri 3x400  
- Sélectionner : Auto / Store / Hybride

## 7.3. Ratio DC/AC

- Définit la limite pour la sélection automatique de l’onduleur

## 7.4. Batterie

- Activer ou non  
- Choisir la capacité (6 à 50 kWh)

## 7.5. Profils de consommation

- Entrez la consommation annuelle  
- Choisissez un profil mensuel  
- Choisissez un profil horaire  

## 7.6. Températures

- Température minimale : impact sur Voc froid  
- Température maximale : impact sur Vmp chaud  

## 7.7. Choix de l’onduleur

- Auto : l’application propose le modèle optimal  
- Manuel : sélectionner n’importe quel modèle compatible  

Après le choix, l’application recalcule :

- strings  
- tensions  
- P_DC  
- ratio DC/AC  

---

# 8. Résultats affichés

- Puissance DC installée  
- Nombre de panneaux câblés  
- Répartition des strings par MPPT  
- Ratio DC/AC  
- Production annuelle  
- Autoconsommation  
- Taux d’autoconsommation  
- Taux de couverture  

Section spéciale :  
- tension MPPT réelle  
- longueur de string par entrée  
- MPPT inutilisés

---

# 9. Visualisations

## 9.1. Graphique mensuel

- Production PV  
- Consommation  
- Autoconsommation directe + batterie  

## 9.2. Profil horaire — jour moyen

- Production horaire  
- Consommation horaire  
- Autoconsommation horaire  

---

# 10. Export Excel

Le bouton **Générer l’Excel** :

- Exporte le matériel choisi  
- Ajoute le profil complet  
- Ajoute la vérification strings (Voc froid / Vmp chaud / MPPT)  
- Fournit une synthèse prête pour le client


