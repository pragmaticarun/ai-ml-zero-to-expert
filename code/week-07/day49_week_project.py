"""Day 49 project: segment 440 wholesale customers (UCI) with log transform + scaling, k-means/GMM, profile the segments."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from common import load_wholesale
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
df = load_wholesale(); spend = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
print(df[spend].describe().loc[["mean", "50%", "max"]].round(0).to_string())
print("skew raw:", df[spend].skew().round(2).to_dict())
Z = StandardScaler().fit_transform(np.log1p(df[spend])); Zr = StandardScaler().fit_transform(df[spend])
print("k  silhouette(log)  silhouette(raw)")
for k in range(2, 8):
    print(f"{k}  {silhouette_score(Z, KMeans(k, n_init=20, random_state=0).fit_predict(Z)):.4f}           {silhouette_score(Zr, KMeans(k, n_init=20, random_state=0).fit_predict(Zr)):.4f}")
km = KMeans(2, n_init=20, random_state=0).fit(Z); df["seg"] = km.labels_
print("segment profile (median spend) and counts:"); print(df.groupby("seg")[spend].median().round(0).assign(n=df.seg.value_counts().sort_index()).to_string())
print("Channel (1=Horeca, 2=Retail) by segment:"); print(pd.crosstab(df.seg, df.Channel).to_string())
print(f"ARI segment vs Channel (never used in fitting): {adjusted_rand_score(df.Channel, df.seg):.4f}")
seeds = [KMeans(2, n_init=1, random_state=s).fit_predict(Z) for s in range(10)]; print("stability: min ARI between 10 single-init runs:", round(min(adjusted_rand_score(seeds[0], s) for s in seeds), 4))
print("GMM BIC by components:", {c: round(GaussianMixture(c, n_init=5, random_state=0).fit(Z).bic(Z), 1) for c in range(1, 6)})
p = PCA(2).fit(Z); print("PCA 2D explained variance ratio:", p.explained_variance_ratio_.round(4).tolist())
