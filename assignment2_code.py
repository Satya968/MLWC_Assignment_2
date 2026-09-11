import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")
os.makedirs(OUT, exist_ok=True)
SNR = [0,5,10,15,20,25,30]
L, N, KDB = 6, 1000, 9
EPS = 1e-12
rng = np.random.RandomState(67)

# ---------------- Q1: USE EXISTING LOS/NLOS DATASET ----------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "los_nlos_dataset.csv")

def q1_dataset():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Could not find los_nlos_dataset.csv in:\n{SCRIPT_DIR}\n\n"
            "Put the already-generated los_nlos_dataset.csv in the same "
            "folder as assignment2_solution.py."
        )
    return pd.read_csv(INPUT_CSV)

def q1_features(df):
    # Feature definitions follow the supplied Part (b) reference code.
    real_cols = [f"h_real_{i}" for i in range(L)]
    imag_cols = [f"h_imag_{i}" for i in range(L)]
    tau_cols = [f"tau_{i}" for i in range(L)]

    real_array = df[real_cols].to_numpy(dtype=float)
    imag_array = df[imag_cols].to_numpy(dtype=float)
    tau_array = df[tau_cols].to_numpy(dtype=float)

    # Received power per tap.
    P = real_array**2 + imag_array**2

    # 1) Kurtosis and 2) skewness of received POWER across taps.
    kurtosis = stats.kurtosis(P, axis=1, fisher=True, bias=True)
    skewness = stats.skew(P, axis=1, bias=True)

    # 3) Rising time: time from 10% to 90% cumulative received energy.
    order = np.argsort(tau_array, axis=1)
    tau_sorted = np.take_along_axis(tau_array, order, axis=1)
    P_sorted = np.take_along_axis(P, order, axis=1)
    cum_energy = np.cumsum(P_sorted, axis=1)
    cum_norm = cum_energy / cum_energy[:, -1:]

    def interp_threshold(thresh):
        out = np.empty(len(df))
        for i in range(len(df)):
            out[i] = np.interp(thresh, cum_norm[i], tau_sorted[i])
        return out

    t10 = interp_threshold(0.10)
    t90 = interp_threshold(0.90)
    rising_time = t90 - t10

    # 4) RMS delay spread.
    total_energy = P.sum(axis=1)
    mean_tau = np.sum(P * tau_array, axis=1) / total_energy
    rms_delay_spread = np.sqrt(
        np.sum(P * (tau_array - mean_tau[:, None])**2, axis=1)
        / total_energy
    )

    # 5) Rician K-factor = dominant-tap power / scattered power.
    P_dominant = P.max(axis=1)
    P_scattered = P.sum(axis=1) - P_dominant
    with np.errstate(divide="ignore", invalid="ignore"):
        rician_k = np.where(P_scattered > 0, P_dominant / P_scattered, np.inf)

    return np.column_stack([
        kurtosis, skewness, rising_time, rms_delay_spread, rician_k
    ])

def svm(x,y,tr,te):
    sc=StandardScaler().fit(x[tr])
    m=SVC(C=1,kernel='rbf').fit(sc.transform(x[tr]),y[tr])
    return 100*np.mean(m.predict(sc.transform(x[te]))==y[te]),sc,m

print("Q1: loading existing los_nlos_dataset.csv...")
d=q1_dataset(); d.to_csv(os.path.join(OUT,'los_nlos_dataset.csv'),index=False)
X=q1_features(d)
names=['Kurtosis','Skewness','Rising time (ns)','RMS delay spread (ns)','Rician K-factor']
for i,nm in enumerate(names): d[nm]=X[:,i]
d.to_csv(os.path.join(OUT,'los_nlos_dataset_with_features.csv'),index=False)
d[['snr_db','label']+names].to_csv(os.path.join(OUT,'q1_features.csv'),index=False)
y=d.label.to_numpy(); snr=d.snr_db.to_numpy()

res=[]; splits={}
for s in SNR:
    ix=np.where(snr==s)[0]; tr,te=train_test_split(ix,test_size=.2,stratify=y[ix],random_state=42); splits[s]=(tr,te)
    for j,nm in enumerate(names):
        acc,_,_=svm(X[:,j:j+1],y,tr,te); res.append([s,nm,acc])
    acc,_,_=svm(X,y,tr,te); res.append([s,'All five features',acc])
r1=pd.DataFrame(res,columns=['SNR (dB)','Classifier','Accuracy (%)'])
r1.to_csv(os.path.join(OUT,'q1_svm_accuracy.csv'),index=False)
r1.to_csv(os.path.join(OUT,'svm_feature_analysis_results.csv'),index=False)

p=r1.pivot(index='SNR (dB)',columns='Classifier',values='Accuracy (%)')

# Q1(b) figure: presentation matched to the requested reference layout.
plot_info=[
    ('Kurtosis','SVM-1: Kurtosis','o'),
    ('Skewness','SVM-2: Skewness','s'),
    ('Rising time (ns)','SVM-3: Rise time','^'),
    ('RMS delay spread (ns)','SVM-4: RMS delay spread','D'),
    ('Rician K-factor','SVM-5: Rician K-factor','v'),
    ('All five features','SVM-6: All features','*')
]
plt.figure(figsize=(13,8))
for col,label,marker in plot_info:
    plt.plot(
        p.index, p[col], marker=marker, linewidth=2.5, markersize=8,
        label=label
    )
plt.xlabel('SNR (dB)',fontsize=14)
plt.ylabel('Classification Accuracy (%)',fontsize=14)
plt.title('LOS/NLOS Classification Accuracy vs. SNR (per feature vs. combined)',fontsize=18)
plt.xticks(SNR,fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(59.5,101.5)
plt.grid(True,alpha=.30)
plt.legend(loc='lower right',fontsize=12,framealpha=.90)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'accuracy_vs_snr.png'),dpi=180)
plt.close()

tr25,te25=splits[25]; a25,sc25,m25=svm(X,y,tr25,te25)
c=[]
for s in SNR:
    tr,te=splits[s]; acc,_,_=svm(X,y,tr,te)
    acc25=100*np.mean(m25.predict(sc25.transform(X[te]))==y[te])
    c.append([s,acc,acc25])
c=pd.DataFrame(c,columns=['SNR (dB)','Train = Test SNR Accuracy (%)','Train at 25 dB Accuracy (%)'])
c.to_csv(os.path.join(OUT,'q1c_training_strategy.csv'),index=False)
c.to_csv(os.path.join(OUT,'part_c_train25_vs_matched_results.csv'),index=False)

plt.figure(figsize=(8,5))
plt.plot(c.iloc[:,0],c.iloc[:,1],marker='o',label='Train = Test SNR')
plt.plot(c.iloc[:,0],c.iloc[:,2],marker='s',label='Train at 25 dB')
plt.xlabel('Test SNR (dB)'); plt.ylabel('Accuracy (%)'); plt.title('Q1(c): Training Strategy')
plt.xticks(SNR); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT,'part_c_train25_vs_matched.png'),dpi=180); plt.close()

# ---------------- Q2: 16-QAM + K-MEANS ----------------
rng=np.random.RandomState(67)
lev=np.array([-3,-1,1,3])/np.sqrt(10)
const=np.array([(i,q) for i in lev for q in lev])
rows=[]
for s in SNR:
    sd=np.sqrt(1/(2*10**(s/10)))
    for sid,(i,q) in enumerate(const):
        z=i+1j*q+sd*(rng.randn(200)+1j*rng.randn(200))
        rows += [[s,sid,v.real,v.imag] for v in z]
q2=pd.DataFrame(rows,columns=['snr_db','symbol_id','xI','xQ'])
q2['r']=np.hypot(q2.xI,q2.xQ); q2['theta']=np.arctan2(q2.xQ,q2.xI)
q2.to_csv(os.path.join(OUT,'qam16_awgn_dataset.csv'),index=False)

def purity(y,l):
    return sum(np.bincount(y[l==c],minlength=16).max() for c in np.unique(l))/len(y)

d25=q2[q2.snr_db==25].reset_index(drop=True); A=d25[['xI','xQ']].to_numpy()
ks=range(2,21); ine=[]; sil=[]
for k in ks:
    m=KMeans(k,init='k-means++',n_init=10,random_state=42); z=m.fit_predict(A)
    ine.append(m.inertia_); sil.append(silhouette_score(A,z))
kdf=pd.DataFrame({'K':list(ks),'Inertia':ine,'Silhouette coefficient':sil})
kdf.to_csv(os.path.join(OUT,'q2_kmeans_k_analysis.csv'),index=False)

plt.figure(figsize=(7,5)); plt.plot(kdf.K,kdf.Inertia,marker='o')
plt.xlabel('K'); plt.ylabel('Inertia'); plt.title('Inertia vs K'); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT,'q2b_inertia_vs_k.png'),dpi=180); plt.close()

plt.figure(figsize=(7,5)); plt.plot(kdf.K,kdf['Silhouette coefficient'],marker='o')
plt.xlabel('K'); plt.ylabel('Silhouette coefficient'); plt.title('Silhouette vs K'); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT,'q2b_silhouette_vs_k.png'),dpi=180); plt.close()

fig,ax=plt.subplots(1,2,figsize=(10,4.5))
ax[0].plot(kdf.K,kdf.Inertia,marker='o'); ax[0].set_xlabel('K'); ax[0].set_ylabel('Inertia'); ax[0].set_title('Inertia vs K')
ax[1].plot(kdf.K,kdf['Silhouette coefficient'],marker='o'); ax[1].set_xlabel('K'); ax[1].set_ylabel('Silhouette'); ax[1].set_title('Silhouette vs K')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'inertia_silhouette_vs_k.png'),dpi=180); plt.close(fig)

m=KMeans(16,init='k-means++',n_init=10,random_state=42); lab=m.fit_predict(A)
plt.figure(figsize=(6,6)); plt.scatter(A[:,0],A[:,1],c=lab,s=7,alpha=.35)
plt.scatter(m.cluster_centers_[:,0],m.cluster_centers_[:,1],marker='X',s=120,edgecolor='black',label='Centroids')
plt.xlabel('$x_I$'); plt.ylabel('$x_Q$'); plt.title('K=16 Cartesian K-means at 25 dB'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT,'kmeans16_cartesian_scatter.png'),dpi=180); plt.close()

y2=d25.symbol_id.to_numpy(); fs=[('Cartesian (xI, xQ)',d25[['xI','xQ']].to_numpy()),
 ('Polar (r, theta)',d25[['r','theta']].to_numpy())]
from sklearn.preprocessing import StandardScaler
fs.append(('Combined standardized (r, xI, xQ, theta)',StandardScaler().fit_transform(d25[['r','xI','xQ','theta']])))
pr=[]
for nm,z in fs:
    m=KMeans(16,init='k-means++',n_init=10,random_state=42); lab=m.fit_predict(z)
    pr.append([nm,100*purity(y2,lab)])
pr=pd.DataFrame(pr,columns=['Feature set','Purity (%)'])
pr.to_csv(os.path.join(OUT,'q2_feature_set_purity.csv'),index=False)

# ---------------- Q2(c) ----------------
fixed=KMeans(16,init='k-means++',n_init=10,random_state=42).fit(A).cluster_centers_
ans=[]
for s in SNR:
    q=q2[q2.snr_db==s]; z=q[['xI','xQ']].to_numpy(); yy=q.symbol_id.to_numpy()
    am=KMeans(16,init='k-means++',n_init=10,random_state=42); al=am.fit_predict(z)
    dist=((z[:,None,:]-fixed[None,:,:])**2).sum(2); fl=np.argmin(dist,1)
    ans.append([s,100*purity(yy,al),100*purity(yy,fl)])
ans=pd.DataFrame(ans,columns=['SNR (dB)','Adaptive K-Means Purity (%)','Fixed Template Purity (%)'])
ans.to_csv(os.path.join(OUT,'q2c_strategy_purity.csv'),index=False)
ans.to_csv(os.path.join(OUT,'adaptive_vs_fixed_purity.csv'),index=False)

plt.figure(figsize=(8,5))
plt.plot(ans.iloc[:,0],ans.iloc[:,1],marker='o',label='Adaptive K-Means')
plt.plot(ans.iloc[:,0],ans.iloc[:,2],marker='s',label='Fixed Template')
plt.xlabel('SNR (dB)'); plt.ylabel('Cluster Purity (%)'); plt.title('Adaptive vs Fixed Template')
plt.xticks(SNR); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT,'adaptive_vs_fixed_purity.png'),dpi=180); plt.close()

print("\\nDone. Files saved in:")
print(OUT)
