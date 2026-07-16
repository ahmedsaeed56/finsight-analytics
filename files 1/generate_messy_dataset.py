"""
JazzCash-style DFS Dataset v2 — REALISTIC & MESSY
==================================================
Changes vs v1 (per review):
 - monthly_income REPLACED by (a) declared_income_band (KYC self-report, messy,
   missing) and (b) avg_monthly_inflow_pkr (observed wallet inflows — the proxy
   real fintechs actually model on).
 - Realistic MESS injected for a cleaning phase (see MESS MANIFEST below).
Hidden ground truths (recover after cleaning):
  churn <- complaints, failed txns, inactivity | default <- inflow-to-loan ratio,
  score, Balochistan | Eid spikes Apr/Jun-2025 | 4 latent segments | insurance <-
  savers+dependents.
MESS MANIFEST (what your cleaning must handle):
  customers: ~200 exact duplicate rows; region case/whitespace variants
   ("punjab","PUNJAB ","Sindh "); age sentinels -999 & impossible 250; tenure
   nulls (~5%); declared_income_band missing ~12% & label variants ("25-50k",
   "25k-50k"); booleans as mixed Yes/TRUE/1/N; churned_12m as Y/N/blank;
   onboarding_date in 3 formats (2021-03-14, 14/03/2021, Mar 14, 2021).
  loans: ~50 duplicate loan_ids; interest_rate as "24%" strings on some rows;
   3 negative amounts (data-entry); disbursed_date mixed formats.
  transactions: ~1% rows missing (gaps); a few negative txn_counts; month
   written as "2025-04" and "Apr-2025" variants.
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
N = 15_000

regions = ["Punjab","Sindh","KP","Balochistan","Islamabad","AJK-GB"]
region = rng.choice(regions, N, p=[.42,.24,.14,.05,.10,.05])
cities = {"Punjab":["Lahore","Faisalabad","Rawalpindi","Multan","Sargodha"],
"Sindh":["Karachi","Hyderabad","Sukkur"],"KP":["Peshawar","Mardan","Abbottabad"],
"Balochistan":["Quetta","Gwadar"],"Islamabad":["Islamabad"],"AJK-GB":["Muzaffarabad","Gilgit"]}
city=[rng.choice(cities[r]) for r in region]
segment = rng.choice(["payroll","merchant","saver","borrower"], N, p=[.35,.20,.25,.20])
age = np.clip(rng.normal(33,9,N).astype(int),18,70)
tenure = np.clip(rng.exponential(20,N).astype(int),1,96)
inflow = np.round(np.clip(rng.lognormal(10.3,.6,N),8_000,900_000),-2)
inflow = np.where(segment=="merchant", inflow*1.4, inflow).round(-2)
dependents = rng.poisson(1.6,N).clip(0,8)
smartphone = rng.random(N)<0.83
complaints = rng.poisson(0.6,N).clip(0,12)
failed = rng.poisson(1.1,N).clip(0,20)
base={"payroll":14,"merchant":46,"saver":9,"borrower":12}
txn_rate = np.array([base[s] for s in segment])*rng.lognormal(0,.35,N)
has_sav=(segment=="saver")|(rng.random(N)<0.22)
sav_bal=np.where(has_sav,np.round(rng.lognormal(9.2,1.0,N),-2),0)
has_ins=rng.random(N)<np.clip(0.10+0.25*(segment=="saver")+0.03*dependents,0,.85)
score=np.clip(420+1.9*np.minimum(tenure,60)+0.00030*inflow-14*complaints-6*failed
 +22*smartphone+rng.normal(0,28,N),300,850).round().astype(int)
chl=(-2.1+0.34*complaints+0.11*failed-0.045*np.minimum(tenure,48)-0.012*txn_rate
 +0.5*(segment=="saver")-0.6*(segment=="merchant"))
churned=rng.random(N)<1/(1+np.exp(-chl))

def band(v):
    if v<25_000: return "<25k"
    if v<50_000: return "25-50k"
    if v<100_000: return "50-100k"
    if v<250_000: return "100-250k"
    return "250k+"
declared=[band(v*rng.uniform(.7,1.4)) for v in inflow]      # self-report noise

onb = pd.to_datetime("2025-06-30")-pd.to_timedelta(tenure*30+rng.integers(0,30,N),unit="D")
cust = pd.DataFrame({"customer_id":[f"C{100000+i}" for i in range(N)],
 "age":age,"region":region,"city":city,"segment_true":segment,
 "onboarding_date":onb.strftime("%Y-%m-%d"),"wallet_tenure_months":tenure.astype(float),
 "declared_income_band":declared,"avg_monthly_inflow_pkr":inflow,
 "dependents":dependents,"smartphone_user":smartphone.astype(object),
 "avg_monthly_txns":txn_rate.round(1),"complaints_12m":complaints,
 "failed_txns_12m":failed,"has_savings":has_sav.astype(object),
 "savings_balance_pkr":sav_bal,"has_insurance":has_ins.astype(object),
 "credit_score":score,"churned_12m":np.where(churned,"Y","N").astype(object)})

# ---------- MESS: customers ----------
idx=rng.choice(N,150,replace=False); cust.loc[idx,"age"]=-999
cust.loc[rng.choice(N,10,replace=False),"age"]=250
cust.loc[rng.choice(N,int(.05*N),replace=False),"wallet_tenure_months"]=np.nan
cust.loc[rng.choice(N,int(.12*N),replace=False),"declared_income_band"]=np.nan
v=rng.choice(N,int(.06*N),replace=False)
cust.loc[v,"declared_income_band"]=cust.loc[v,"declared_income_band"].str.replace("25-50k","25k-50k")
m=rng.choice(N,int(.15*N),replace=False)
cust.loc[m,"region"]=cust.loc[m,"region"].str.lower()
m2=rng.choice(N,int(.08*N),replace=False)
cust.loc[m2,"region"]=cust.loc[m2,"region"].str.upper()+" "
for col in ["smartphone_user","has_savings","has_insurance"]:
    k=rng.choice(N,int(.3*N),replace=False)
    cust.loc[k,col]=cust.loc[k,col].map({True:"Yes",False:"N"})
    k2=rng.choice(N,int(.2*N),replace=False)
    cust.loc[k2,col]=cust.loc[k2,col].map(lambda x:{True:1,False:0}.get(x,x))
cust.loc[rng.choice(N,int(.02*N),replace=False),"churned_12m"]=""
d1=rng.choice(N,int(.2*N),replace=False)
cust.loc[d1,"onboarding_date"]=pd.to_datetime(cust.loc[d1,"onboarding_date"]).dt.strftime("%d/%m/%Y")
d2=rng.choice(N,int(.1*N),replace=False)
cust.loc[d2,"onboarding_date"]=pd.to_datetime(cust.loc[d2,"onboarding_date"],dayfirst=False,errors="coerce").fillna(pd.Timestamp("2023-01-01")).dt.strftime("%b %d, %Y")
cust=pd.concat([cust,cust.sample(200,random_state=1)],ignore_index=True)   # dup rows

# ---------- loans ----------
p=(0.55*(segment=="borrower")+0.25*(segment=="merchant")+0.1); p/=p.sum()
bi=rng.choice(N,8_000,replace=False,p=p); L=len(bi)
purpose=rng.choice(["nano_loan","merchant_advance","device_finance","emergency"],L,p=[.45,.22,.18,.15])
amount=np.select([purpose=="nano_loan",purpose=="merchant_advance",purpose=="device_finance",purpose=="emergency"],
 [rng.integers(2_000,25_000,L),rng.integers(30_000,400_000,L),rng.integers(15_000,120_000,L),rng.integers(5_000,50_000,L)])
term=rng.choice([1,3,6,12],L,p=[.35,.3,.25,.1])
cb=cust.iloc[bi].reset_index(drop=True)
lti=amount/cb["avg_monthly_inflow_pkr"]
dl=(-3.6+1.5*np.clip(lti,0,4)-0.0045*cb["credit_score"]+0.45*(cb["region"].str.strip().str.title()=="Balochistan")+rng.normal(0,.4,L))
dflt=rng.random(L)<1/(1+np.exp(-dl))
loans=pd.DataFrame({"loan_id":[f"L{500000+i}" for i in range(L)],
 "customer_id":cb["customer_id"],
 "disbursed_date":(pd.to_datetime("2024-07-01")+pd.to_timedelta(rng.integers(0,330,L),unit="D")).strftime("%Y-%m-%d"),
 "purpose":purpose,"amount_pkr":amount.astype(float),"term_months":term,
 "interest_rate_pct":np.round(rng.uniform(18,36,L),1).astype(object),
 "inflow_to_loan_ratio":lti.round(2),"defaulted":dflt})
loans.loc[rng.choice(L,3,replace=False),"amount_pkr"]*=-1
k=rng.choice(L,int(.25*L),replace=False)
loans.loc[k,"interest_rate_pct"]=loans.loc[k,"interest_rate_pct"].astype(str)+"%"
d=rng.choice(L,int(.15*L),replace=False)
loans.loc[d,"disbursed_date"]=pd.to_datetime(loans.loc[d,"disbursed_date"]).dt.strftime("%d/%m/%Y")
loans=pd.concat([loans,loans.sample(50,random_state=2)],ignore_index=True)  # dup loan_ids

# ---------- transactions ----------
months=pd.period_range("2024-07","2025-06",freq="M")
eb={pd.Period("2025-04"):1.45,pd.Period("2025-06"):1.30}
rows=[]
for m in months:
    b=eb.get(m,1.0)
    t=rng.poisson(np.maximum(txn_rate/3,.4)*b)
    v=np.round(t*rng.lognormal(7.6,.5,N)*b,-1)
    lbl = str(m) if rng.random()>0.15 else m.strftime("%b-%Y")   # mixed month formats
    rows.append(pd.DataFrame({"customer_id":cust["customer_id"][:N],"month":lbl,
                              "txn_count":t,"txn_value_pkr":v}))
tx=pd.concat(rows,ignore_index=True)
tx=tx.drop(rng.choice(len(tx),int(.01*len(tx)),replace=False)).reset_index(drop=True)
tx.loc[rng.choice(len(tx),25,replace=False),"txn_count"]*=-1

cust.to_csv("customers_raw.csv",index=False)
loans.to_csv("loans_raw.csv",index=False)
tx.to_csv("transactions_raw.csv",index=False)
print("customers:",cust.shape,"loans:",loans.shape,"tx:",tx.shape,
 "| default:",round(dflt.mean(),3),"| churn:",round((cust['churned_12m']=='Y').mean(),3))
