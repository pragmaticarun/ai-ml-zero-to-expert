# Responsible-AI audit report: Adult income logistic regression

Date: 2026-10-29 (Asia/Calcutta). Auditor: course Week 24 Day 168 script.
Model: sklearn LogisticRegression C=1, OneHot+StandardScaler pipeline. Fit time 0.361 s on CPU.
Data: OpenML 'adult' v2 (UCI Adult Census Income), 70/30 stratified split seed 0; train 34189, test 14653.

## 1. Intended use and out of scope
- Intended use: pedagogical fairness / privacy / robustness audit inside this course.
- Out of scope: any real credit, hiring, insurance, or benefit decision; any jurisdiction's protected-class rules we have not checked.

## 2. Performance and fairness (Day 162/163 metrics, re-measured)
- **baseline t=0.5**: acc 0.8542; sel M/F 0.257/0.073 (DI 0.29); TPR M/F 0.624/0.489; FPR M/F 0.096/0.021; prec M/F 0.739/0.741
- **reweighing t=0.5**: acc 0.8469; sel M/F 0.210/0.120 (DI 0.57); TPR M/F 0.540/0.647; FPR M/F 0.067/0.054; prec M/F 0.780/0.600
- **equal-opp thresholds M=0.52 F=0.40**: acc 0.8534; sel M/F 0.248/0.092 (DI 0.37); TPR M/F 0.608/0.568; FPR M/F 0.090/0.033; prec M/F 0.747/0.685
- Base rates: Male 0.304, Female 0.111. Calibration ECE Male/Female from Day 162: 0.0099 / 0.0078.
- Trade-off note: reweighing raised DI from 0.29 to ~0.57 and cut overall accuracy by ~0.7 points; equal-opportunity thresholds nearly matched TPRs with almost no accuracy loss. Demographic-parity thresholds (Day 163) equalised selection rates but cut Female precision to 0.488.

## 3. Interpretability (Day 164)
- Permutation importance (acc drop, 3000 rows, 3 repeats): marital-status +0.0594, education-num +0.0341, capital-gain +0.0314, occupation +0.0173, sex +0.0069, race +0.0011
- Proxy check: sex predicted from the other features, AUC 0.933 (Day 162: 0.933). Dropping the sex column barely moved DI (0.29 -> 0.30).

## 4. Privacy (Day 165)
- Loss-threshold membership-inference AUC on this model: 0.4989 (0.5 = chance). Day 165 DP-SGD at sigma=1.0 kept test acc 0.8481 with eps ≈ 1.44 (Opacus RDPAccountant, delta 1e-5) and MIA AUC 0.4981.

## 5. Robustness and shift (Day 166)
- CNN (Month 3) adversarial: clean 0.722 -> FGSM/PGD at 8/255: 0.042 / 0.000 on 500 CIFAR test images. Adult subpopulation shift: train age<40, test age>=40 accuracy 0.6945 vs 0.8954 in-distribution; domain classifier AUC 0.800 with age hidden.
- Recommendation: do not deploy without a shift monitor (domain classifier or PSI) and an adversarial/corruption eval on the actual input modality.

## 6. LLM safety note (Day 167, SmolLM2-135M-Instruct)
- Refused 1/3 harmful prompts; over-refusal 0/3 benign; injection canary did not fire (answered '2+2=4'); invented entities for Atlantis-Prime and Zorblax. Small models need external filters; do not rely on the base model alone.

## 7. Documentation and governance
- Model card and datasheet written to code/week-24/out/day167_model_card.json and day167_datasheet.json.
- NIST AI RMF 1.0 (NIST.AI.100-1): map this audit to MEASURE (metrics above) and MANAGE (choose reweighing or EO thresholds; refuse deployment for real decisions). Voluntary framework.
- EU AI Act overview (Commission page): a credit-scoring use of this model would fall under high-risk examples on that page. This course model is pedagogical only. Not legal advice.

## 8. Environmental cost (measured × ASSUMED)
- Measured fit time 0.361 s. ASSUMED package power 45 W (no RAPL). Estimated energy 16.2 J for one fit. Day 167: CNN 128-batch forward 19.3 ms.

## 9. Decision
- **Ship as a course artefact: YES.** Documented, mitigated options measured, privacy leakage on this LR is near chance.
- **Ship as a real decision system: NO.** Protected-attribute gaps remain under every single threshold; proxy leakage of sex is high (AUC ~0.93); the dataset is 1990s US Census; no legal review; adversarial/shift coverage is incomplete for tabular inputs.
- Preferred mitigated variant for further course work: equal-opportunity group thresholds (almost no accuracy cost, TPR nearly matched). Reweighing if demographic parity is the product requirement, accepting the precision drop for the minority-selected group.

## 10. Residual risks
- Intersectionality not audited (race × sex). Only sex was the audited attribute.
- Thresholds tuned on a train-side split; still a choice that can overfit the fairness metric.
- Energy figure uses ASSUMED watts.
- LLM numbers are for a 135M model, not a production assistant.
