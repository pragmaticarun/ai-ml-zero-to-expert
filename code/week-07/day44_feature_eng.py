"""Day 44: feature engineering on hourly bike demand: hour as number vs one-hot vs cyclic, interactions; text counts."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from common import load_bike
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
df = load_bike(); y = df.pop("count").astype(float); X = df
for c in ["workingday", "holiday"]: X[c] = (X[c].astype(str) == "True").astype(int)
for c in ["year", "month", "hour", "weekday"]: X[c] = X[c].astype(int)
print("rows", len(X), "columns", list(X.columns)); print("hourly mean demand (0..23):", y.groupby(X.hour).mean().round(0).astype(int).tolist())
num = ["temp", "feel_temp", "humidity", "windspeed", "year", "workingday", "holiday"]; cat = ["season", "weather"]
cyc = FunctionTransformer(lambda h: np.c_[np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)], feature_names_out="one-to-one")
def lin(hour):
    tr = [("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]
    if hour == "numeric": tr.append(("h", "passthrough", ["hour"]))
    if hour == "cyclic": tr.append(("h", make_pipeline(FunctionTransformer(lambda d: d.to_numpy(float)), cyc), ["hour"]))
    if hour in ("onehot", "onehot x workingday"): tr.append(("h", OneHotEncoder(), ["hour"]))
    ct = ColumnTransformer(tr)
    if hour == "onehot x workingday":
        inter = FunctionTransformer(lambda d: np.c_[pd.get_dummies(d.hour).to_numpy(float) * d.workingday.to_numpy()[:, None]])
        ct = ColumnTransformer(tr + [("hxw", inter, ["hour", "workingday"])])
    return make_pipeline(ct, RidgeCV(alphas=np.logspace(-3, 3, 7)))
cv = TimeSeriesSplit(5); mae = lambda m: -cross_val_score(m, X, y, cv=cv, scoring="neg_mean_absolute_error").mean()
print(f"baseline predict-mean CV MAE: {np.mean([np.abs(y.iloc[te] - y.iloc[tr].mean()).mean() for tr, te in cv.split(X)]):.1f}")
for h in ["none", "numeric", "cyclic", "onehot", "onehot x workingday"]: print(f"ridge, hour={h:20s} CV MAE {mae(lin(h)):.1f}")
print(f"HistGB raw features (cats native)  CV MAE {mae(HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=0)):.1f}")
docs = ["the battery drains fast", "battery life is great", "screen cracked, battery fine", "great screen great price"]
cv1 = CountVectorizer(); B = cv1.fit_transform(docs); print("vocab:", list(cv1.get_feature_names_out())); print("counts row0:", B.toarray()[0].tolist())
tf = TfidfVectorizer().fit(docs); print("idf battery vs cracked:", round(float(tf.idf_[tf.vocabulary_["battery"]]), 3), round(float(tf.idf_[tf.vocabulary_["cracked"]]), 3))
