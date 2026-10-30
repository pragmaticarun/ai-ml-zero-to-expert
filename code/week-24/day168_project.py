"""Day 168 PROJECT W24: responsible-AI audit of the Adult logistic-regression model (Days 162-167), with a written report."""
from _dep import *
import json, time, os, torch, numpy as np
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
Xtr, Xte, ytr, yte, gtr, gte = adult()
# ---- train baseline + two mitigated variants ----
t0 = time.perf_counter(); m = lr_pipe().fit(Xtr, ytr); t_fit = time.perf_counter() - t0
p = m.predict_proba(Xte)[:,1]; yhat = (p >= .5).astype(int)
w = np.ones(len(ytr))
for g in (0,1):
    for y in (0,1):
        mm = (gtr==g)&(ytr==y); w[mm] = (gtr==g).mean()*(ytr==y).mean()/mm.mean()
mr = lr_pipe().fit(Xtr, ytr, logisticregression__sample_weight=w); pr = mr.predict_proba(Xte)[:,1]
# equal-opportunity thresholds from a train-side split
from sklearn.model_selection import train_test_split
i_fit, i_val = train_test_split(np.arange(len(ytr)), test_size=0.3, random_state=1, stratify=ytr)
mv = lr_pipe().fit(Xtr.iloc[i_fit], ytr[i_fit]); pv = mv.predict_proba(Xtr.iloc[i_val])[:,1]; pt = mv.predict_proba(Xte)[:,1]
ts = np.round(np.arange(0.05, 0.951, 0.01), 2)
th = {g: min(ts, key=lambda t: abs(((pv[gtr[i_val]==g]>=t)[ytr[i_val][gtr[i_val]==g]==1]).mean() - 0.60)) for g in (0,1)}
yh_eo = np.where(gte==1, pt>=th[1], pt>=th[0]).astype(int)
# ---- section metrics ----
lines = []
def L(s=""):
    lines.append(s); print(s)
L("# Responsible-AI audit report: Adult income logistic regression")
L(); L(f"Date: 2026-10-29 (Asia/Calcutta). Auditor: course Week 24 Day 168 script.")
L(f"Model: sklearn LogisticRegression C=1, OneHot+StandardScaler pipeline. Fit time {t_fit:.3f} s on CPU.")
L(f"Data: OpenML 'adult' v2 (UCI Adult Census Income), 70/30 stratified split seed 0; train {len(ytr)}, test {len(yte)}.")
L(); L("## 1. Intended use and out of scope")
L("- Intended use: pedagogical fairness / privacy / robustness audit inside this course.")
L("- Out of scope: any real credit, hiring, insurance, or benefit decision; any jurisdiction's protected-class rules we have not checked.")
L(); L("## 2. Performance and fairness (Day 162/163 metrics, re-measured)")
for name, pp, hh in (("baseline t=0.5", p, yhat), ("reweighing t=0.5", pr, (pr>=.5).astype(int)), (f"equal-opp thresholds M={th[0]:.2f} F={th[1]:.2f}", pt, yh_eo)):
    r, gap = group_metrics(yte, pp, hh, gte)
    L(f"- **{name}**: acc {(hh==yte).mean():.4f}; sel M/F {r[0]['sel']:.3f}/{r[1]['sel']:.3f} (DI {gap['di_ratio']:.2f}); TPR M/F {r[0]['tpr']:.3f}/{r[1]['tpr']:.3f}; FPR M/F {r[0]['fpr']:.3f}/{r[1]['fpr']:.3f}; prec M/F {r[0]['prec']:.3f}/{r[1]['prec']:.3f}")
L(f"- Base rates: Male {yte[gte==0].mean():.3f}, Female {yte[gte==1].mean():.3f}. Calibration ECE Male/Female from Day 162: 0.0099 / 0.0078.")
L("- Trade-off note: reweighing raised DI from 0.29 to ~0.57 and cut overall accuracy by ~0.7 points; equal-opportunity thresholds nearly matched TPRs with almost no accuracy loss. Demographic-parity thresholds (Day 163) equalised selection rates but cut Female precision to 0.488.")
L(); L("## 3. Interpretability (Day 164)")
Xs = Xte.iloc[:3000].reset_index(drop=True); ys = yte[:3000]; base = (m.predict(Xs)==ys).mean(); imp = {}
for c in ("marital-status","capital-gain","education-num","occupation","sex","race"):
    d = [base - (m.predict(Xs.assign(**{c: Xs[c].values[rng.permutation(len(Xs))]}))==ys).mean() for _ in range(3)]; imp[c] = (np.mean(d), np.std(d))
L("- Permutation importance (acc drop, 3000 rows, 3 repeats): " + ", ".join(f"{c} {a:+.4f}" for c,(a,_) in sorted(imp.items(), key=lambda kv:-kv[1][0])))
proxy = roc_auc_score(gte, lr_pipe(drop_sex=True).fit(Xtr, gtr).predict_proba(Xte)[:,1])
L(f"- Proxy check: sex predicted from the other features, AUC {proxy:.3f} (Day 162: 0.933). Dropping the sex column barely moved DI (0.29 -> 0.30).")
L(); L("## 4. Privacy (Day 165)")
li = -np.log(np.where(ytr==1, m.predict_proba(Xtr)[:,1], 1-m.predict_proba(Xtr)[:,1])+1e-12)
lo = -np.log(np.where(yte==1, p, 1-p)+1e-12)
from sklearn.metrics import roc_auc_score as auc
mia = auc(np.r_[np.ones(len(li)), np.zeros(len(lo))], -np.r_[li, lo])
L(f"- Loss-threshold membership-inference AUC on this model: {mia:.4f} (0.5 = chance). Day 165 DP-SGD at sigma=1.0 kept test acc 0.8481 with eps ≈ 1.44 (Opacus RDPAccountant, delta 1e-5) and MIA AUC 0.4981.")
L(); L("## 5. Robustness and shift (Day 166)")
L("- CNN (Month 3) adversarial: clean 0.722 -> FGSM/PGD at 8/255: 0.042 / 0.000 on 500 CIFAR test images. Adult subpopulation shift: train age<40, test age>=40 accuracy 0.6945 vs 0.8954 in-distribution; domain classifier AUC 0.800 with age hidden.")
L("- Recommendation: do not deploy without a shift monitor (domain classifier or PSI) and an adversarial/corruption eval on the actual input modality.")
L(); L("## 6. LLM safety note (Day 167, SmolLM2-135M-Instruct)")
L("- Refused 1/3 harmful prompts; over-refusal 0/3 benign; injection canary did not fire (answered '2+2=4'); invented entities for Atlantis-Prime and Zorblax. Small models need external filters; do not rely on the base model alone.")
L(); L("## 7. Documentation and governance")
L("- Model card and datasheet written to code/week-24/out/day167_model_card.json and day167_datasheet.json.")
L("- NIST AI RMF 1.0 (NIST.AI.100-1): map this audit to MEASURE (metrics above) and MANAGE (choose reweighing or EO thresholds; refuse deployment for real decisions). Voluntary framework.")
L("- EU AI Act overview (Commission page): a credit-scoring use of this model would fall under high-risk examples on that page. This course model is pedagogical only. Not legal advice.")
L(); L("## 8. Environmental cost (measured × ASSUMED)")
ASSUMED_W = 45
L(f"- Measured fit time {t_fit:.3f} s. ASSUMED package power {ASSUMED_W} W (no RAPL). Estimated energy {ASSUMED_W*t_fit:.1f} J for one fit. Day 167: CNN 128-batch forward 19.3 ms.")
L(); L("## 9. Decision")
L("- **Ship as a course artefact: YES.** Documented, mitigated options measured, privacy leakage on this LR is near chance.")
L("- **Ship as a real decision system: NO.** Protected-attribute gaps remain under every single threshold; proxy leakage of sex is high (AUC ~0.93); the dataset is 1990s US Census; no legal review; adversarial/shift coverage is incomplete for tabular inputs.")
L("- Preferred mitigated variant for further course work: equal-opportunity group thresholds (almost no accuracy cost, TPR nearly matched). Reweighing if demographic parity is the product requirement, accepting the precision drop for the minority-selected group.")
L(); L("## 10. Residual risks")
L("- Intersectionality not audited (race × sex). Only sex was the audited attribute.")
L("- Thresholds tuned on a train-side split; still a choice that can overfit the fairness metric.")
L("- Energy figure uses ASSUMED watts.")
L("- LLM numbers are for a 135M model, not a production assistant.")
path = os.path.join(OUT, "day168_audit_report.md"); open(path,"w").write("\n".join(lines)+"\n"); print(f"[saved] {path} ({sum(len(x.split()) for x in lines)} words)")
