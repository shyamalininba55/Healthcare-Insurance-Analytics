"""Generate reproducible synthetic insurance and feedback datasets."""
import numpy as np, pandas as pd
from pathlib import Path
rng=np.random.default_rng(42); n=1400
age=rng.integers(18,66,n); sex=rng.choice(["female","male"],n)
bmi=np.clip(rng.normal(30.6,6.1,n),16,53); children=rng.integers(0,6,n)
smoker=rng.choice(["yes","no"],n,p=[.205,.795])
region=rng.choice(["northeast","northwest","southeast","southwest"],n)
policy=rng.choice(["Basic","Standard","Premium"],n,p=[.35,.45,.20])
ded=rng.choice([500,1000,1500,2000,3000],n)
coverage=np.where(policy=="Basic",50000,np.where(policy=="Standard",100000,200000))+rng.integers(-5000,5001,n)
network=rng.choice(["HMO","PPO","EPO"],n)
claims=np.maximum(0,rng.poisson(1.2,n)+(age>55).astype(int))
charges=np.maximum(800,950+age*245+np.maximum(bmi-25,0)*170+(smoker=="yes")*19000+children*260+(policy=="Premium")*800+claims*1100+rng.normal(0,1800,n))
cat=pd.cut(charges,[0,7000,18000,np.inf],labels=["Low","Medium","High"]).astype(str)
out=Path("data"); out.mkdir(exist_ok=True)
pd.DataFrame({"customer_id":[f"C{i+1:04d}" for i in range(n)],"age":age,"sex":sex,"bmi":bmi.round(1),"children":children,"smoker":smoker,"region":region,"policy_type":policy,"deductible":ded,"coverage_limit":coverage,"network_type":network,"claim_count":claims,"charges":charges.round(2),"claim_cost_category":cat}).to_csv(out/"insurance_customers.csv",index=False)
pos=["Claim was settled quickly and the support team was helpful.","Very satisfied with the policy coverage and easy claim process.","Customer service answered my questions clearly.","Cashless hospital service was smooth and convenient.","Premium is reasonable and the benefits are excellent."]
neg=["My claim was delayed and customer support did not respond.","Policy exclusions were unclear and the claim was rejected.","Premium increased and the service was disappointing.","Hospital network information was confusing.","Reimbursement took too long and communication was poor."]
neu=["I received my policy document by email.","The renewal date is next month.","I contacted the branch to update my address.","The policy includes hospital coverage details.","I submitted the required claim documents."]
rows=[]
for i in range(1000):
 s=rng.choice(["Positive","Negative","Neutral"],p=[.42,.38,.20]); pool={"Positive":pos,"Negative":neg,"Neutral":neu}[s]
 rows.append((f"F{i+1:04d}",f"C{rng.integers(1,n+1):04d}",rng.choice(pool),s))
pd.DataFrame(rows,columns=["feedback_id","customer_id","feedback_text","sentiment"]).to_csv(out/"customer_feedback.csv",index=False)
print("Generated datasets in data/")
