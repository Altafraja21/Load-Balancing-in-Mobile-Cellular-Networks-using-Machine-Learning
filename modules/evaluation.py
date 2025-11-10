import numpy as np
def jain_index(l): l=np.array(l,float); return 1.0 if l.sum()==0 else (l.sum()**2)/(len(l)*(l**2).sum())
def throughput_proxy(l,c): l,c=np.array(l),np.array(c); s=np.minimum(l,c); return s.sum(),(s/c).mean()
def compute_loads(users,bs): return users["Connected_BS"].value_counts().reindex(bs["BS_ID"]).fillna(0).astype(int).values
