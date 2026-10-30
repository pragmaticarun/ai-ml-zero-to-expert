"""Day 46: leakage demo (select+scale outside vs inside CV) and a ColumnTransformer pipeline with imputation on adult."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common import load_adult
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as sel
from sklearn.impute import SimpleImputer
rng = np.random.default_rng(46); Xn = rng.normal(size=(200, 5000)); yn = rng.integers(0, 2, 200)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
Xs = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)
print(f"pure noise, select 20 of 5000 features on ALL rows then CV: acc {cross_val_score(LogisticRegression(max_iter=1000), Xs, yn, cv=cv).mean():.3f}")
print(f"pure noise, selection inside pipeline:              acc {cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000)), Xn, yn, cv=cv).mean():.3f}")
X, y = load_adult(); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), sel(dtype_include="number")),
                         ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(handle_unknown="ignore")), sel(dtype_include="category"))])
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])
gs = GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1, 10]}, cv=cv, scoring="roc_auc").fit(Xtr, ytr)
print("CV ROC-AUC by C:", dict(zip(["0.01", "0.1", "1", "10"], gs.cv_results_["mean_test_score"].round(4).tolist())))
print(f"best {gs.best_params_}; test ROC-AUC (once) {gs.score(Xte, yte):.4f}")
names = gs.best_estimator_.named_steps["pre"].get_feature_names_out(); print("n features after preprocessing:", len(names), " e.g.", list(names[:3]), list(names[-2:]))
print("unseen category handled? predict on a row with occupation='Astronaut':", end=" ")
r = Xte.iloc[:1].copy(); r["occupation"] = r["occupation"].cat.add_categories(["Astronaut"]); r.loc[r.index[0], "occupation"] = "Astronaut"
print(gs.predict_proba(r)[0, 1].round(4))
