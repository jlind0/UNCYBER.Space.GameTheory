from math import exp, log
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
with st.expander("About this app", expanded=False):
    st.markdown(
        """
# OODA Attrition – State-of-the-Art Overview  

This Streamlit dashboard lets you **prototype, stress-test, and compare competing air-combat doctrines** by combining:

* **Modern decision-cycle theory** (OODA)  
* **State-of-the-art aggregation functions** drawn from economics, reliability engineering, and cognition research  
* **Interactive Bayesian sliders** for effort allocation and lethality

The result is a *rapid-fire lab* where planners can see how small shifts in doctrine or resource mix ripple through a multi-nation fight.

---

## 1  Function Families & When to Use Them

| Family | Mathematical Form | Intuition | When to assign it to a nation | Key coefficients |
|--------|-------------------|-----------|-------------------------------|------------------|
| **`power`** (generalised mean) | \(E = \bigl( \alpha O^p + \beta R^p + \gamma D^p + \delta A^p \bigr)^{1/p}\) | Smoothly blends inputs; tune *p* to move from max-like to min-like behaviour. | Nations whose *tempo* is limited by their **slowest** O/R/D/A phase (lower *p*) or whose strengths compound (higher *p*). | `alpha…delta` (phase weights), `p` (elasticity) |
| **`ces`** (Constant-Elasticity of Substitution) | Same form as `power`, but parameters are usually called *ρ*. | Widely used in combat modelling; controls how easily effort shifts between phases. | Forces that can fluidly re-task ISR assets or pilots → pick **high substitutability** (*ρ* near 0). Rigid doctrines → *ρ* farther from 0. | `alpha…delta`, `rho` |
| **`cobb`** (Cobb–Douglas) | \(E = k\,O^{\alpha}R^{\beta}D^{\gamma}A^{\delta}\) | Pure multiplicative synergy; if one phase is 0, effectiveness is 0. | Highly integrated doctrines (e.g., USAF “fusion warfare”) where a single phase failure is catastrophic. | `k` (scale), `alpha…delta` |
| **`exp`** (Exponential saturation) | \(E = c\,[1 - e^{-\lambda(O+R+D+A)}]\) | Rapid gains early, diminishing returns later. | Conscript or low-tech forces: initial improvements help a lot, but plateau quickly. | `c` (cap), `lam` (rise rate) |
| **`logit`** (probabilistic trigger) | \(E = \bigl[1+e^{-(\Sigma \alpha_i \ln P_i - \theta)}\bigr]^{-1}\) | Interprets effectiveness as a **probability of seizing the initiative**. | Nations with doctrine centred on *critical decision points* (e.g., Russia’s emphasis on first-salvo advantage). | `alpha…delta`, `theta` (difficulty) |
| **`quad`** (full quadratic) | Weighted quadratic over all pairings. | Captures **synergies and trade-offs** explicitly (e.g., Observe-Orient cross-term). | Research or wargame labs exploring niche interactions; expensive but expressive. | `w1…w10` (weights) |
| **`hybrid`** (bespoke mix) | \(E = k\,O^{\alpha}e^{-\lambda R} + \beta D + \gamma \sqrt{A}\) | Semi-empirical formula merging power, decay, and surprise. | NATO-style composite doctrine: strong ISR surge (O), rapid R hamper, and decisive action spikes (A). | `k, alpha, beta, gamma, delta, lam` |

### Picking Families in Practice
1. **Doctrine survey** Tag each nation’s *training philosophy*—is it attrition-centric, manoeuvre-centric, or probability-of-kill focused?  
2. **Elasticity guess** If phases can substitute one another, favour `power`/`ces`; else use `cobb` or `logit`.  
3. **Complexity budget** For quick what-ifs, start with `cobb` or `power`. Use `quad` only when you have data to justify ten weights.  
4. **Tune coefficients** Start from the default table in *Settings → Coefficients*. Then calibrate against red-/blue-flag sortie data or Monte-Carlo runs.

---

## 2  ORDA Effort Shares and Why They Matter

The sliders labelled **O / R / D** set the *fraction of pilot & C2 effort* devoted to each phase; **A** is auto-filled to ensure \(O+R+D+A=1\).

* **Power-/CES-style families:** Effectiveness rises fastest when effort flows into the **most-weighted** phase; e.g., if `alpha > beta`, boosting *O* gives better marginal returns than *R*.  
* **Cobb–Douglas:** Decreasing any one share below ~0.1 sharply cuts \(E\). Balanced ORDA usually beats extreme specialisation.  
* **Exponential:** Returns plateau—after ~0.3 total effort per phase you get <5 percent gain. Good for modelling diminishing ISR returns.  
* **Logit:** Think *threshold*. Until the weighted log-sum crosses \(\theta\), \(E\) is near 0; once past it, small extra effort lands outsized payoff.  
* **Hybrid:**   Observe fuels the first term; Reduce *R* (Radar deception) to avoid the negative exponent; high *A* boosts the square-root term—great for doctrines that strike after blinding radar.

### Quick-Start Heuristics
| Scenario | ORDA tweak | Expected outcome |
|----------|-----------|------------------|
| “Blind-then-strike” SEAD package | Raise *R* to 0.35, drop *D* | Hurts enemy detect track chain; `hybrid` or `logit` nations benefit most |
| Pilot training surge | Increase *D* (decision) for rookies | `power` family gains where *p* > 1; `cobb` nations still need Observe support |
| Swarm drones | Push *A* ≥ 0.5 | Only safe with `exp` (plateau) or `quad` (positive A² term) to avoid over-commit |

---

## 3  Using the Sliders

1. **Select Nation-level family** in *Sidebar → Aggregation family by nation*.  
2. **Adjust ORDA** inside each cohort expander. Watch the live **Attrition curve** reshape.  
3. **Fine-tune coefficients** under *Phase Weights*—hover for tool-tips.  
4. **Normalise E** with the *Logistic remap* (λ, midpoint) if cross-family scaling is desired.

---

### Further Reading
* Boyd, J. (1995) *Destruction and Creation* – foundational OODA essay.  
* Moffat, J. (2017) *Command and Control in Military Crises* – formal tempo models.  
* Kott, A. et al. (2020) “Game-Theoretic Combat Modelling with Heterogeneous Forces”, *Journal of Defense Modeling & Simulation*.
* OODA Game Theory Paper: [https://multiplex.studio/files/OODA-Game.pdf](https://multiplex.studio/files/OODA-Game.pdf)
* Source Code: [https://github.com/jlind0/UNCYBER.Space.GameTheory/blob/main/AirWar/app.py](https://github.com/jlind0/UNCYBER.Space.GameTheory/blob/main/AirWar/app.py)

Happy experimenting—may your model reveal the hidden corners of air-combat tempo!
        """
    )
# ────────────────────────────────────────────────
EPS = 1e-9
st.set_page_config(page_title="OODA – Six‑Cohort Duel", layout="wide")
st.title("OODA Attrition – NATO & Ukraine vs China & Russia (4 + 5 Gen mix)")

# ───────────────── Aggregation families ──────────────────

def power_mean(O, R, D, A, alpha, beta, gamma, delta, p):
    s = alpha * O ** p + beta * R ** p + gamma * D ** p + delta * A ** p
    return s ** (1 / p) if p != 0 else np.exp(
        (alpha * log(O) + beta * log(R) + gamma * log(D) + delta * log(A))
        / (alpha + beta + gamma + delta)
    )


def CES(O, R, D, A, alpha, beta, gamma, delta, rho):
    s = alpha * O ** rho + beta * R ** rho + gamma * D ** rho + delta * A ** rho
    return s ** (1 / rho)


def cobb(O, R, D, A, k, alpha, beta, gamma, delta):
    return k * O ** alpha * R ** beta * D ** gamma * A ** delta


def exp_sat(O, R, D, A, c, lam):
    return c * (1 - np.exp(-lam * (O + R + D + A)))


def logit(O, R, D, A, alpha, beta, gamma, delta, theta):
    z = (
        alpha * log(O + EPS)
        + beta * log(R + EPS)
        + gamma * log(D + EPS)
        + delta * log(A + EPS)
        - theta
    )
    return 1 / (1 + exp(-z))


def quad(O, R, D, A, w):
    return (
        w["w1"] * O ** 2
        + w["w2"] * R ** 2
        + w["w3"] * D ** 2
        + w["w4"] * A ** 2
        + w["w5"] * O * R
        + w["w6"] * O * D
        + w["w7"] * O * A
        + w["w8"] * R * D
        + w["w9"] * R * A
        + w["w10"] * D * A
    )


def hybrid(O, R, D, A, k, alpha, beta, gamma, delta, lam):
    return k * (O ** alpha) * (np.exp(-lam * R)) + beta * D + gamma * np.sqrt(A)


FAMILY_FUNCS = {
    "power": power_mean,
    "ces": CES,
    "cobb": cobb,
    "exp": exp_sat,
    "logit": logit,
    "quad": quad,
    "hybrid": hybrid,
}
FAMILIES = list(FAMILY_FUNCS.keys())

GENERIC = {
    "power": dict(alpha=0.25, beta=0.25, gamma=0.25, delta=0.25, p=0.8),
    "ces": dict(alpha=0.25, beta=0.25, gamma=0.25, delta=0.25, rho=1.5),
    "cobb": dict(k=0.7, alpha=0.3, beta=0.3, gamma=0.2, delta=0.2),
    "exp": dict(c=1.0, lam=0.05),
    "logit": dict(alpha=0.3, beta=0.3, gamma=0.2, delta=0.2, theta=0.1),
    "quad": dict(
        w1=0.2,
        w2=0.2,
        w3=0.2,
        w4=0.2,
        w5=0.05,
        w6=0.05,
        w7=0.05,
        w8=0.02,
        w9=0.02,
        w10=0.01,
    ),
    "hybrid": dict(k=0.6, alpha=0.3, beta=0.2, gamma=0.2, delta=0.3, lam=0.05),
}
# ───────── Cohort defaults: specific airframes ─────────

DEFAULT_COHORTS = [
    ("NATO",    "F-35A"),
    ("NATO",    "F-16C"),
    ("NATO",    "Eurofighter Typhoon"),
    ("Ukraine", "MiG-29"),
    ("Ukraine", "Su-27"),
    ("Ukraine",  "F-16C"),
    ("China",   "J-20"),
    ("China",   "J-10C"),
    ("Russia",  "Su-57"),
    ("Russia",  "Su-35S"),
]

COHORT_DEFAULTS = {
    "NATO": {
        "F-35A":              {"count": 24, "k": 0.080, "orda": (0.32, 0.28, 0.20), "vuln": 0.65},
        "F-16C":              {"count": 30, "k": 0.055, "orda": (0.30, 0.30, 0.25), "vuln": 1.00},
        "Eurofighter Typhoon":{"count": 12, "k": 0.065, "orda": (0.30, 0.28, 0.22), "vuln": 0.90},
    },
    "Ukraine": {
        "MiG-29":             {"count": 25, "k": 0.035, "orda": (0.30, 0.25, 0.25), "vuln": 1.15},
        "Su-27":              {"count": 25, "k": 0.040, "orda": (0.28, 0.26, 0.26), "vuln": 1.10},
        "F-16C":              {"count": 15, "k": 0.045, "orda": (0.30, 0.30, 0.25), "vuln": 1.05},
    },
    "China": {
        "J-20":               {"count": 18, "k": 0.080, "orda": (0.32, 0.28, 0.20), "vuln": 0.70},
        "J-10C":              {"count": 30, "k": 0.050, "orda": (0.28, 0.30, 0.22), "vuln": 1.00},
    },
    "Russia": {
        "Su-57":              {"count": 25, "k": 0.075, "orda": (0.32, 0.28, 0.20), "vuln": 0.75},
        "Su-35S":             {"count": 30, "k": 0.060, "orda": (0.30, 0.28, 0.22), "vuln": 0.85},
    },
}

# ───────── Generic fall-back for any airframe label not in COHORT_DEFAULTS ─────────
AIRFRAME_DEFAULTS = {
    "F-35A":               {"count": 24, "k": 0.080, "orda": (0.32, 0.28, 0.20), "vuln": 0.65},
    "F-16C":               {"count": 60, "k": 0.045, "orda": (0.30, 0.30, 0.25), "vuln": 1.00},
    "Eurofighter Typhoon": {"count": 40, "k": 0.055, "orda": (0.30, 0.28, 0.22), "vuln": 0.90},
    "MiG-29":              {"count": 28, "k": 0.035, "orda": (0.30, 0.25, 0.25), "vuln": 1.15},
    "Su-27":               {"count": 24, "k": 0.040, "orda": (0.28, 0.26, 0.26), "vuln": 1.10},
    "J-20":                {"count": 20, "k": 0.080, "orda": (0.32, 0.28, 0.20), "vuln": 0.70},
    "J-10C":               {"count": 40, "k": 0.050, "orda": (0.28, 0.30, 0.22), "vuln": 1.00},
    "Su-57":               {"count": 12, "k": 0.075, "orda": (0.32, 0.28, 0.20), "vuln": 0.75},
    "Su-35S":              {"count": 30, "k": 0.060, "orda": (0.30, 0.28, 0.22), "vuln": 0.85},
}
NATION_BASES = {
    "NATO": {
        "Incirlik AB (TUR)":      (37.00, 35.43),   # Adana, Turkey
        "M. Kogălniceanu AB (RO)":(44.36, 28.49),   # Black-Sea coast, Romania
    },
    "Ukraine": {
        "Kulbakino Mykolaiv":     (46.44, 32.45),
        "Odesa Intl":             (46.43, 30.17),
    },
    "Russia": {
        "Belbek AB (Sevastopol)": (44.69, 33.57),
        "Saki AB (Crimea)":       (45.05, 33.59),
    },
    "China": {
        "Khmeimim Det (Syria)":   (35.41, 35.97),
    },
}
# ───────── Default ORDA-family coefficients per nation × airframe ─────────
# Keys must match nation-airframe pairs used in COHORT_DEFAULTS & DEFAULT_COHORTS
DEFAULT_COEFFS = {
    "NATO": {
        "F-35A": dict(alpha=0.32, beta=0.22, gamma=0.18, delta=0.28, k=0.6,  lam=0.05),  # hybrid
        "F-16C": dict(alpha=0.28, beta=0.18, gamma=0.22, delta=0.32, k=0.6,  lam=0.05),
        "Eurofighter Typhoon":
                     dict(alpha=0.30, beta=0.20, gamma=0.22, delta=0.28, k=0.6,  lam=0.05),
    },
    "Ukraine": {
        "MiG-29": dict(alpha=0.30, beta=0.30, gamma=0.20, delta=0.20, p=0.90),          # power
        "Su-27":  dict(alpha=0.28, beta=0.32, gamma=0.18, delta=0.22, p=0.95),
        "F-16C": dict(alpha=0.28, beta=0.32, gamma=0.18, delta=0.22, p=0.95),
    },
    "China": {
        "J-20":  dict(alpha=0.28, beta=0.28, gamma=0.22, delta=0.22, rho=1.40),         # CES
        "J-10C": dict(alpha=0.24, beta=0.24, gamma=0.28, delta=0.24, rho=1.60),
    },
    "Russia": {
        "Su-57":  dict(alpha=0.32, beta=0.32, gamma=0.18, delta=0.18, theta=0.08),      # logit
        "Su-35S": dict(alpha=0.28, beta=0.28, gamma=0.22, delta=0.22, theta=0.12),
    },
}
AIRFRAME_LOADOUTS = {
    "F-35A":  {"AIM-120C": 4, "AIM-9X": 2},
    "F-16C":  {"AIM-120C": 2, "AIM-9M": 2},
    "Eurofighter Typhoon": {"Meteor": 4, "ASRAAM": 2},
    "MiG-29": {"R-27R": 2, "R-73": 2},
    "Su-27":  {"R-27ER": 2, "R-73": 2},
    "J-20":   {"PL-15": 4, "PL-10": 2},
    "J-10C":  {"PL-15": 2, "PL-8": 2},
    "Su-57":  {"R-77M": 4, "R-74M2": 2},
    "Su-35S": {"R-77-1": 4, "R-73": 2},
}
WEAPON_LIBRARY = {
    "AIM-120C": {"range_km": 105, "pk": 0.55},
    "AIM-9X":   {"range_km":  35, "pk": 0.30},
    "AIM-9M":   {"range_km":  30, "pk": 0.25},
    "Meteor":   {"range_km": 150, "pk": 0.60},
    "ASRAAM":   {"range_km":  40, "pk": 0.32},
    "R-27R":    {"range_km":  80, "pk": 0.35},
    "R-27ER":   {"range_km": 110, "pk": 0.40},
    "R-73":     {"range_km":  30, "pk": 0.25},
    "PL-15":    {"range_km": 150, "pk": 0.55},
    "PL-10":    {"range_km":  40, "pk": 0.32},
    "PL-8":     {"range_km":  35, "pk": 0.25},
    "R-77M":    {"range_km": 160, "pk": 0.55},
    "R-77-1":   {"range_km": 110, "pk": 0.45},
    "R-74M2":   {"range_km":  40, "pk": 0.30},
}

# Keep an editable copy in session state
if "weapons" not in st.session_state:
    st.session_state.weapons = {w: stats.copy() for w, stats in WEAPON_LIBRARY.items()}
if 'cohort_params' not in st.session_state:
    st.session_state.cohort_params = {}
if 'cohorts' not in st.session_state:
    st.session_state.cohorts = DEFAULT_COHORTS.copy()
if "cohort_weapons" not in st.session_state:
    st.session_state.cohort_weapons = {}
if "nations" not in st.session_state:
    st.session_state.nations = {
        "NATO":    {"coalition": "Allied", "family": "hybrid"},
        "Ukraine": {"coalition": "Allied", "family": "power"},
        "China":   {"coalition": "Asia",   "family": "ces"},
        "Russia":  {"coalition": "Asia",   "family": "logit"},
    }

if "airframes" not in st.session_state:
    st.session_state.airframes = AIRFRAME_DEFAULTS.copy()
if "airframe_loadouts" not in st.session_state:
    st.session_state.airframe_loadouts = {
        af: d.copy() for af, d in AIRFRAME_LOADOUTS.items()
    }
# editable copy of bases
if "bases" not in st.session_state:
    st.session_state.bases = {n: d.copy() for n, d in NATION_BASES.items()}
# cohort → { base_name: qty }
if "cohort_base_counts" not in st.session_state:
    st.session_state.cohort_base_counts = {}
# 2) Sidebar: add nations
with st.sidebar.expander("➕ Add Nation", expanded=False):
    new_nat = st.text_input("Nation name", key="new_nation_name")
    coal   = st.text_input("Coalition",  key="new_nation_coal")
    fam    = st.selectbox("Family", ["cobb","power","ces","logit"], key="new_nation_fam")
    if st.button("Create Nation"):
        if new_nat and new_nat not in st.session_state.nations:
            st.session_state.nations[new_nat] = {"coalition": coal, "family": fam}
            st.success(f"Added nation “{new_nat}”")
        else:
            st.error("Enter a unique nation name.")

# 3) Sidebar: add airframes
with st.sidebar.expander("➕ Add Airframe", expanded=False):
    new_af     = st.text_input("Airframe label (e.g. “6-Gen”)", key="new_af_name")
    cnt        = st.number_input("Default count", min_value=0, max_value=500, value=10, step=1, key="new_af_cnt")
    k_val      = st.number_input("Default k",    min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="new_af_k")
    vuln_val   = st.number_input("Default vuln", min_value=0.0, max_value=2.0, value=1.0, step=0.05, key="new_af_vuln")
    orda_A     = st.slider("O", 0.0, 1.0, 0.3, 0.01, key="new_af_ordaA")
    orda_R     = st.slider("R", 0.0, 1.0, 0.3, 0.01, key="new_af_ordaR")
    orda_D     = st.slider("D", 0.0, 1.0, 0.25,0.01, key="new_af_ordaD")
    if st.button("Create Airframe"):
        if new_af and new_af not in st.session_state.airframes:
            # normalize ORDA if it doesn't sum to 1
            total = orda_A+orda_R+orda_D
            shares = (orda_A/total, orda_R/total, orda_D/total)
            st.session_state.airframes[new_af] = {
                "count": cnt, "k": k_val, "vuln": vuln_val, "orda": shares
            }
            st.success(f"Added airframe “{new_af}”")
        else:
            st.error("Enter a unique airframe label.")
# ───────── UI helper – shares ─────────
def share_sliders(tag: str, tbl=(0.25, 0.25, 0.25), pfx: str = ""):
    """Expose O-, R-, D-sliders; A is auto-computed as 1-(O+R+D)."""
    # ignore any preset A, accept 3- or 4-tuples
    if len(tbl) < 3:
        tbl = (0.25, 0.25, 0.25)
    O_def, R_def, D_def = tbl[0], tbl[1], tbl[2]

    colO, colR, colD = st.columns(3)
    with colO:
        O = st.slider(f"{pfx}O ({tag})", 0.0, 1.0, O_def, 0.01)
    with colR:
        R = st.slider(f"{pfx}R ({tag})", 0.0, 1.0, R_def, 0.01)
    with colD:
        D = st.slider(f"{pfx}D ({tag})", 0.0, 1.0, D_def, 0.01)

    A = 1.0 - (O + R + D)
    if A < 0:
        st.warning("O + R + D exceed 1; A set to 0.")
        A = 0.0
    st.caption(f"A (auto) = {A:.2f}")
    return O, R, D, A

# ───────── Effectiveness helpers ─────────

def raw_E(shares, fam, params=None):
    """Compute raw effectiveness with optional overridden params."""
    base_params = GENERIC.get(fam, {})
    if params is None:
        params = base_params
    fn = FAMILY_FUNCS[fam]
    if fam == "quad":
        # merge quad weights with overrides
        p = GENERIC["quad"].copy()
        p.update(params)
        return fn(*shares, p)
    return fn(*shares, **params)


def logistic_remap(val: float, lam: float = 5.0, mid: float = 0.5) -> float:
    """Bound *val* to (0,1) using a logistic centred on *mid* with slope *lam*."""
    return 1.0 / (1.0 + exp(-lam * (val - mid)))
if 'cohorts' not in st.session_state:
    st.session_state.cohorts = DEFAULT_COHORTS.copy()
with st.sidebar.expander("🛠 Scenario Editor", expanded=False):
    st.write("Add a new cohort (nation + airframe):")
    new_nation   = st.selectbox("Nation",   list(st.session_state.nations.keys()), key="new_cohort_nation")
    new_airframe = st.selectbox("Airframe", list(st.session_state.airframes.keys()), key="new_cohort_airframe")
    if st.button("Add Cohort"):
        candidate = (new_nation, new_airframe)
        if candidate in st.session_state.cohorts:
            st.warning(f"Cohort {candidate} already exists.")
        else:
            st.session_state.cohorts.append(candidate)
            # seed base allocation: all jets at the first listed base
            bases_for_nat = list(st.session_state.bases.get(new_nation, {}))
            if bases_for_nat:
                st.session_state.cohort_base_counts[f"{new_nation}-{new_airframe}"] = {
                    bases_for_nat[0]: COHORT_DEFAULTS.get(new_nation, {})
                        .get(new_airframe, AIRFRAME_DEFAULTS[new_airframe])["count"]
                }
            st.success(f"Added cohort {candidate}")
            st.rerun()
def _update_family(nation, selectbox_key):
    # 1) Update the nation’s family
    new_family = st.session_state[selectbox_key]
    st.session_state.nations[nation]["family"] = new_family

    # 2) For each cohort of that nation, clear out old sliders and reset params
    for coh_nation, airframe in st.session_state.cohorts:
        if coh_nation != nation:
            continue

        key = f"{coh_nation}-{airframe}"

        # Determine the new default coefficients for this nation+airframe
        default = (
            GENERIC.get(new_family, {})).copy()

        # Remove any stale slider state for this cohort’s ORDA params
        prefix = f"{key}-orda-param-"
        for sess_key in list(st.session_state.keys()):
            if sess_key.startswith(prefix):
                del st.session_state[sess_key]

        # Overwrite the cohort’s params with the new defaults
        st.session_state.cohort_params[key] = default

    # 3) Rerun so all sliders rebuild with the new defaults
    st.rerun()
# ───────────── Sidebar ─────────────
with st.sidebar:
    st.header("Force counts, lethality & families")

        # ————————————— Air-bases by nation —————————————
    with st.expander("Air-bases", expanded=False):
        for nat in sorted(st.session_state.bases):
            st.markdown(f"**{nat}**")
            for bname, (lat, lon) in list(st.session_state.bases[nat].items()):
                col1, col2, col3 = st.columns((4, 2, 2))
                with col1:
                    st.text_input("Name", bname, key=f"bname-{nat}-{bname}", disabled=True)
                with col2:
                    st.number_input("Lat", value=lat, key=f"blat-{nat}-{bname}")
                with col3:
                    st.number_input("Lon", value=lon, key=f"blon-{nat}-{bname}")

            # add a new base for this nation
            new_bn = st.text_input("New base name", key=f"newbase-{nat}")
            # ⬇️ map-picker section
            st.caption("Click map or type numbers ↓")
            col_lat, col_lon = st.columns(2)
            new_lat = col_lat.number_input("Lat°", -90.0, 90.0, key=f"newlat-{nat}")
            new_lon = col_lon.number_input("Lon°", -180.0, 180.0, key=f"newlon-{nat}")

            # show small map centred on Black Sea
            _m = folium.Map(location=[44.5, 34.0], zoom_start=5)
            folium.LatLngPopup().add_to(_m)
            # render & capture click
            map_out = st_folium(_m, height=250, width=400, key=f"map-{nat}")
            if map_out and map_out.get("last_clicked"):
                new_lat = map_out["last_clicked"]["lat"]
                new_lon = map_out["last_clicked"]["lng"]
                # update inputs visually
                st.session_state[f"newlat-{new_bn}-{nat}"] = new_lat
                st.session_state[f"newlon-{new_bn}-{nat}"] = new_lon
            if st.button("Add base", key=f"addbase-{nat}"):
                if new_bn and new_bn not in st.session_state.bases[nat]:
                    st.session_state.bases[nat][new_bn] = (new_lat, new_lon)
                    st.toast(f"Base “{new_bn}” added to {nat}")
                    st.rerun()
    # ————————————— Weapon catalogue (shared stats) —————————————
    with st.expander("Weapon characteristics", expanded=False):
        for w in sorted(st.session_state.weapons):
            with st.expander(w, expanded=False):
                spec = st.session_state.weapons[w]
                r_key = f"{w}-rng"; p_key = f"{w}-pk"
                spec["range_km"] = st.number_input(
                    f"{w} range (km)", 10, 300, spec["range_km"], 1, key=r_key
                )
                spec["pk"] = st.slider(
                    f"{w} Pₖ", 0.05, 0.95, spec["pk"], 0.01, key=p_key
                )

        # ————————————— Airframe default load-outs —————————————
        st.subheader("Airframe default load-outs")
        for af in sorted(st.session_state.airframe_loadouts):
            with st.expander(af, expanded=False):
                load = st.session_state.airframe_loadouts[af]
                # edit existing weapons
                for w in list(load):
                    q_key = f"{af}-{w}-qty"
                    new_q = st.number_input(
                        f"{w} qty", 0, 12, load[w], 1, key=q_key
                    )
                    if new_q == 0:                               # ← deletion path
                        del load[w]                              # 1) remove here
                        # 2) propagate to all cohorts flying this air-frame
                        for ckey, cweps in st.session_state.cohort_weapons.items():
                            _, _, c_af = ckey.partition("-")     # "Nation-Airframe"
                            if c_af == af:
                                cweps.pop(w, None)
                        st.toast(f"{w} removed from {af} and its cohorts")
                        st.rerun()
                    else:
                        load[w] = new_q
                # add another weapon
                add_w = st.selectbox(
                    "Add weapon", ["— select —"] + sorted(st.session_state.weapons),
                    key=f"{af}-addw"
                )
                add_q = st.number_input("Qty", 1, 12, 2, 1, key=f"{af}-addq")
                if st.button("Add", key=f"{af}-addbtn"):
                    if add_w == "— select —":
                        st.warning("Pick a weapon first.")
                    else:
                        # 1) add to the current air-frame
                        load[add_w] = add_q

                        # 2) propagate to _every_ other air-frame that ALREADY carries this weapon
                        for other_af, other_load in st.session_state.airframe_loadouts.items():
                            if other_af == af:
                                continue
                            if add_w in other_load:
                                other_load[add_w] = add_q
                                                # 3) ensure all current cohorts flying those air-frames get the weapon
                        for ckey, cweps in st.session_state.cohort_weapons.items():
                            # cohort key format: "Nation-Airframe"
                            _, _, c_af = ckey.partition("-")
                            if add_w in st.session_state.airframe_loadouts.get(c_af, {}):
                                # add only if not already present (honour manual changes)
                                cweps.setdefault(add_w, add_q)
                        st.toast(f"{add_w} set to qty {add_q} for all applicable air-frames")
                        st.rerun()
        # ————————————— Aggregation families (per nation) —————————————
    st.subheader("Aggregation family by nation")
    for nation in st.session_state.nations:
        # grab current, then let user override
        select_key = f"fam-{nation}"
        curr = st.session_state.nations[nation]["family"]
        new_fam = st.selectbox(
            f"Family for {nation}",
            FAMILIES,
            index=FAMILIES.index(curr),
            key=select_key,
            on_change=_update_family,
            args = (nation, select_key)
        )
        st.session_state.nations[nation]["family"] = new_fam
        

    # —————————— now your existing counts/k/kvals/shares/vulns loops ——————————
    counts = {}
    kvals  = {}
    famsel = {}
    shares = {}
    vulns  = {}

    for nation, airframe in st.session_state.cohorts:
        
        presets = (
            COHORT_DEFAULTS
              .get(nation, {})
              .get(airframe, AIRFRAME_DEFAULTS[airframe])
        )

        key = f"{nation}-{airframe}"
            # clone current default load-out on first appearance
        if key not in st.session_state.cohort_weapons:
            st.session_state.cohort_weapons[key] = (
                st.session_state.airframe_loadouts.get(airframe, {}).copy()
            )
        with st.expander(key, expanded=False):
            if st.button("Delete", key=f"delete-{nation}-{airframe}"):
            # 1) remove the cohort tuple itself
                for ix, (ination, iairframe) in enumerate(st.session_state.cohorts):
                    if(nation == ination and iairframe == airframe):
                        st.session_state.cohorts.pop(ix)

                # 2) remove any widgets/session keys tied to this cohort
                prefix = f"{nation}-{airframe}"
                for k in list(st.session_state.keys()):
                    if k.startswith(prefix):
                        del st.session_state[k]
                st.session_state.cohort_base_counts.pop(key, None)
                st.rerun()
            counts[key] = st.slider(
                f"{nation} {airframe} count", 0, 200,
                presets["count"], 1, key=f"{key}-count"
            )
            kvals[key] = st.slider(
                f"k ({nation} {airframe})", 0.0, 0.2,
                presets["k"], 0.01, key=f"{key}-k"
            )
            shares[key] = share_sliders(
                f"{nation} {airframe}", presets["orda"], pfx=""
            )
            vulns[key] = st.slider(
                f"Vulnerability ({nation} {airframe})", 0.5, 1.5,
                presets["vuln"], 0.05, key=f"{key}-vuln"
            )
                        # ————————————— Base selector —————————————
                        # ————————————— Base allocation —————————————
            nat_bases = list(st.session_state.bases.get(nation, {}))
            if key not in st.session_state.cohort_base_counts:
                # first render → put everything at first base
                st.session_state.cohort_base_counts[key] = {
                    nat_bases[0] if nat_bases else "—": counts[key]
                }

            st.markdown("**Base allocation**")
            total_assigned = 0
            for b in nat_bases:
                alloc_key = f"{key}-base-{b}"
                current   = st.session_state.cohort_base_counts[key].get(b, 0)
                new_qty = st.slider(
                    f"{b}", 0, counts[key], current, 1, key=alloc_key
                )
                st.session_state.cohort_base_counts[key][b] = new_qty
                total_assigned += new_qty

            # warn if allocations don’t match cohort size
            if total_assigned != counts[key]:
                st.warning(f"{total_assigned} / {counts[key]} aircraft assigned")
            # ————————————— Cohort weapon quantities —————————————
            st.markdown("**Weapons**")
            for w in list(st.session_state.cohort_weapons[key]):
                spec = st.session_state.weapons.get(w, {"range_km":"?", "pk":0})
                q_key = f"{key}-{w}-qty"
                qty = st.slider(
                    f"{w} qty", 0, 12,
                    st.session_state.cohort_weapons[key][w], 1, key=q_key
                )
                if qty == 0:
                    del st.session_state.cohort_weapons[key][w]
                else:
                    st.session_state.cohort_weapons[key][w] = qty
                st.caption(f" Range {spec['range_km']} km | Pₖ {spec['pk']:.2f}")
            # **new**: record which family this cohort uses
            # use family set per nation
            famsel[key] = st.session_state.nations[nation]["family"]
        # seed default coefficients for this cohort (nation+airframe)
            default = DEFAULT_COEFFS.get(nation, {}).get(airframe, GENERIC.get(famsel[key], {}))
            st.session_state.cohort_params.setdefault(key, default.copy())
            # dynamic per-cohort ORDA coefficient overrides based on selected family
            fam = famsel[key]
            base_params = GENERIC.get(fam, {})
            # ensure storage exists
            st.session_state.cohort_params.setdefault(key, {})
            for param, val in list(st.session_state.cohort_params[key].items()):
                max_val = max(1.0, val * 2)
                slider_key = f"{key}-orda-param-{param}"
                st.session_state.cohort_params[key][param] = st.slider(
                    f"{param.capitalize()}",
                    0.0,
                    max_val,
                    val,
                    max_val/100,
                    key=slider_key,
                )
        
    st.header("Model parameters")
    horizon = st.slider("Engagement (s)", 30, 600, 30, 10)
    dt = 0.2

    st.subheader("Logistic remap (E → Ê)")
    lam_E = st.slider("λ (slope)", 1.0, 10.0, 1.5, 0.5)
    mid_E = st.slider("mid‑point", 0.1, 1.0, 1.0, 0.05)

    st.subheader("Kill advantage scale σ")
    sigma = st.slider("σ (steepness)", 1.0, 10.0, 3.5, 0.5)

# ───────── Calculate effectiveness ─────────
# Anchor‑ratio scaling: compare every raw E to its family's baseline value
ANCHOR_SHARES = (0.25, 0.25, 0.25, 0.25)
E0 = {fam: raw_E(ANCHOR_SHARES, fam) for fam in FAMILIES}


E_raw = {
    t: raw_E(shares[t], famsel[t], st.session_state.cohort_params.get(t, {}))
    for t in shares
}
E_scaled = {t: E_raw[t] / (E0[famsel[t]] + EPS) for t in E_raw}
E = {t: logistic_remap(E_scaled[t], lam_E, mid_E) for t in E_scaled}


# ───────── Simulation arrays ─────────
steps = int(horizon / dt) + 1
_time = np.linspace(0, horizon, steps)
# create state arrays for each cohort in the dynamic list
state = {
    f"{nation}-{airframe}": np.empty(steps)
    for nation, airframe in st.session_state.cohorts
}

for t in state:
    state[t][0] = counts[t]

side_tags = {}
for nation, airframe in st.session_state.cohorts:
    key = f"{nation}-{airframe}"
    coal = st.session_state.nations[nation]["coalition"]
    side_tags.setdefault(coal, []).append(key)


for i in range(1, steps):
    # carry forward last-tick counts
    for tag in state:
        state[tag][i] = state[tag][i - 1]

    # per-side fire-power  P = Σ (count × k × Ê)
    P = {
        side: sum(state[tag][i] * kvals[tag] * E[tag]
                  for tag in tags)
        for side, tags in side_tags.items()
    }

    # kill budgets this tick (both sides shoot)
    K = {side: sigma * P[side] * dt for side in P}
    sides = list(side_tags.keys())
    # apply losses proportionally to the opposite side
    for side in sides:
        for foe in sides:
            if foe == side:
                continue
            foe_total = sum(
                state[tag][i] * vulns[tag]
            for tag in side_tags[foe]
            ) + EPS
    # total enemy pressure from "foe" on "side"
            attr_rate = sum(
                E[enemy] * state[enemy][i]
                for enemy in side_tags[foe]
            )
            
            if foe_total == 0:
                continue
            for tag in side_tags[foe]:
                effective_K = K[side] * (attr_rate / (sum(state[e][i] for e in side_tags[foe]) + EPS))
                hits = hits = effective_K * (state[tag][i] * vulns[tag]) / foe_total
                loss = hits * vulns[tag]
                state[tag][i] = max(0.0, state[tag][i] - loss)


# ───────── Plot ─────────
fig, ax = plt.subplots()
from itertools import cycle
styles = ["-", "--", "-.", ":", "dashdot", "dotted", (0, (3, 1, 1, 1))]
style_iter = cycle(styles)
 # Plot each cohort series; will re-run automatically when cohorts change
for nation, airframe in st.session_state.cohorts:
    key = f"{nation}-{airframe}"
    sty = next(style_iter)
    ax.plot(_time, state[key], label=key, linestyle=sty, linewidth=2)
ax.set_xlabel("Itterations")
ax.set_ylabel("Aircraft remaining")
ax.set_title("Attrition – (Logistic Δ‑based kills)")
ax.legend()
st.pyplot(fig)

# ───────── Metrics summary ─────────
side1_name = "NATO⁺UKR"
side2_name = "CHN⁺RUS"

 # compute survivors for each side dynamically
metrics = {coal: sum(state[t][-1] for t in tags) for coal, tags in side_tags.items()}
cols = st.columns(len(metrics))
for col, (coal, survivors) in zip(cols, metrics.items()):
    col.metric(f"{coal} survivors", f"{survivors:.1f}")