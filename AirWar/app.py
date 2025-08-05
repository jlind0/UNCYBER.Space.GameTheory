"""
Streamlit MVP – 6-Cohort Duel
=============================
Side 1  ▸ **NATO 5 Gen, NATO 4 Gen, Ukraine 4 Gen**  
Side 2  ▸ **China 5 Gen, China 4 Gen, Russia 4 Gen**

Every cohort now has:
* Force-count slider
* Base lethality **k** slider
* O-R-D share sliders (A auto-fills)
* Drop-down to choose one of the **seven aggregation families**

Run locally
```bash
pip install streamlit numpy matplotlib
streamlit run oodamvp.py
```
Then mix doctrines and numbers to see how the six-way sky fight plays out.
"""

from math import exp, log
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

EPS = 1e-9
st.set_page_config(page_title="OODA – Six-Cohort Duel", layout="wide")
st.title("OODA Attrition – NATO & Ukraine vs China & Russia (4 + 5 gen mix)")

# ───────────────── Aggregation families ──────────────────

def power_mean(O,R,D,A,alpha,beta,gamma,delta,p):
    s = alpha*O**p + beta*R**p + gamma*D**p + delta*A**p
    return s**(1/p) if p != 0 else np.exp((alpha*log(O)+beta*log(R)+gamma*log(D)+delta*log(A))/(alpha+beta+gamma+delta))

def CES(O,R,D,A,alpha,beta,gamma,delta,rho):
    s = alpha*O**rho + beta*R**rho + gamma*D**rho + delta*A**rho
    return s**(1/rho)

def cobb(O,R,D,A,k,alpha,beta,gamma,delta):
    return k*O**alpha*R**beta*D**gamma*A**delta

def exp_sat(O,R,D,A,alpha,beta,gamma,delta):
    return 1-exp(-(alpha*O+beta*R+gamma*D+delta*A))

def logit(O,R,D,A,alpha,beta,gamma,delta,theta):
    z = alpha*log(O+EPS)+beta*log(R+EPS)+gamma*log(D+EPS)+delta*log(A+EPS)-theta
    return 1/(1+exp(-z))

def quad(O,R,D,A,w):
    return (w['w1']*O**2+w['w2']*R**2+w['w3']*D**2+w['w4']*A**2+
            w['w12']*O*R+w['w13']*O*D+w['w14']*O*A+
            w['w23']*R*D+w['w24']*R*A+w['w34']*D*A)

def hybrid(O,R,D,A,alpha,beta,gamma,delta,lam):
    return alpha*O+beta*R+gamma*D+delta*A+lam*(O*R*D*A)**0.25

FAMILY_FUNCS = {
    'power':  power_mean,
    'ces':    CES,
    'cobb':   cobb,
    'exp':    exp_sat,
    'logit':  logit,
    'quad':   quad,
    'hybrid': hybrid,
}
FAMILIES = list(FAMILY_FUNCS.keys())

GENERIC = {
    'power': dict(alpha=0.25,beta=0.25,gamma=0.25,delta=0.25,p=0.8),
    'ces':   dict(alpha=0.25,beta=0.25,gamma=0.25,delta=0.25,rho=1.5),
    'cobb':  dict(k=1,alpha=1.4,beta=1.3,gamma=1.2,delta=1.5),
    'exp':   dict(alpha=1,beta=0.8,gamma=0.6,delta=0.6),
    'logit': dict(alpha=2,beta=1,gamma=1,delta=2,theta=0),
    'quad':  dict(w1=1,w2=1,w3=4,w4=4,w12=1,w13=1,w14=1,w23=1,w24=1,w34=5),
    'hybrid':dict(alpha=0.25,beta=0.25,gamma=0.25,delta=0.25,lam=0.5),
}

COHORTS = [
    ('NATO 5-Gen','Side 1','cobb'),
    ('NATO 4-Gen','Side 1','cobb'),
    ('Ukraine 4-Gen','Side 1','power'),
    ('China 5-Gen','Side 2','ces'),
    ('China 4-Gen','Side 2','exp'),
     ('Russia 5-Gen','Side 2','logit'),
    ('Russia 4-Gen','Side 2','quad'),
]

# UI helper – shares

def share_sliders(lbl,base=(0.25,0.25,0.25)):
    st.subheader(lbl)
    o=st.slider(f"{lbl} O-share",0.0,1.0,base[0],0.05)
    r=st.slider(f"{lbl} R-share",0.0,1.0,base[1],0.05)
    d=st.slider(f"{lbl} D-share",0.0,1.0,base[2],0.05)
    tot=o+r+d
    if tot>=1:
        o,r,d=[x/(tot+EPS)*0.9 for x in (o,r,d)]
        tot=o+r+d
    a=1-tot
    st.text(f"A-share → {a:.2f}")
    return o,r,d,a

# Compute E

def E_val(shares,fam):
    params=GENERIC[fam]
    fn=FAMILY_FUNCS[fam]
    if fam=='quad':
        p=GENERIC['quad'].copy();p.update(params);return fn(*shares,p)
    return fn(*shares,**params)

# ───────────── Sidebar ─────────────
with st.sidebar:
    st.header("Force counts & lethality k")
    counts={}; kvals={}; famsel={}; shares={}
    for tag,side,dfam in COHORTS:
        counts[tag]=st.slider(f"{tag} count",0,200,60,10)
        kvals[tag]=st.slider(f"k {tag}",0.0,0.2,0.05,0.01)
        famsel[tag]=st.selectbox(f"{tag} family",FAMILIES,FAMILIES.index(dfam))
        shares[tag]=share_sliders(tag)

    st.header("Timeline")
    horizon=st.slider("Engagement (s)",30,600,180,10); dt=0.2

# Calculate E for cohorts
E={t:E_val(shares[t],famsel[t]) for t,_,_ in COHORTS}

# Simulation arrays
steps=int(horizon/dt)+1; time=np.linspace(0,horizon,steps)
state={t:np.empty(steps) for t,_,_ in COHORTS}
for t in state: state[t][0]=counts[t]

# Helper to get side sums
side_tags={
    'Side 1':[t for t,s,_ in COHORTS if s=='Side 1'],
    'Side 2':[t for t,s,_ in COHORTS if s=='Side 2'],
}

for i in range(1,steps):
    s1=sum(state[t][i-1] for t in side_tags['Side 1'])
    s2=sum(state[t][i-1] for t in side_tags['Side 2'])
    if s1==0 or s2==0:
        for t in state: state[t][i:]=state[t][i-1]
        break

    kills1=dt*sum(kvals[t]*E[t]*state[t][i-1] for t in side_tags['Side 1'])
    kills2=dt*sum(kvals[t]*E[t]*state[t][i-1] for t in side_tags['Side 2'])

    # distribute onto opponents proportionally
    for side,kills,op in [(side_tags['Side 2'],kills1,s2),(side_tags['Side 1'],kills2,s1)]:
        for t in side:
            frac=state[t][i-1]/op if op>0 else 0
            state[t][i]=max(state[t][i-1]-kills*frac,0)
    # carry unchanged for others processed above
    for t in state:
        if len(state[t])>i and state[t][i]==0 and state[t][i-1]>0:
            state[t][i]=state[t][i-1]

# ───────── Plot ─────────
fig,ax=plt.subplots()
styles=['-','--','-.',':','-.','--','--']
for (tag,_,_),sty in zip(COHORTS,styles):
    ax.plot(time,state[tag],label=tag,linestyle=sty,linewidth=2)
ax.set_xlabel("Time (s)");ax.set_ylabel("Aircraft remaining")
ax.set_title("Attrition – Six-Cohort Duel")
ax.legend()
st.pyplot(fig)

# Metrics summary
s1_end=sum(state[t][-1] for t in side_tags['Side 1'])
s2_end=sum(state[t][-1] for t in side_tags['Side 2'])
col1,col2=st.columns(2)
col1.metric("Side 1 survivors",f"{s1_end:.1f} (NATO⁺UKR)")
col2.metric("Side 2 survivors",f"{s2_end:.1f} (CHN⁺RUS)")

st.caption("Six cohorts, seven aggregation families, endless doctrine permutations.  ✈️🧮 #vibecoding")
