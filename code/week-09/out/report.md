# Month-2 capstone report: adult income (>50K) classifier

## CV comparison (train split only)
```
        model  roc_auc  roc_auc_std  pr_auc  log_loss  accuracy  fit_s
hist_gb_tuned   0.9280       0.0030  0.8278    0.2778    0.8723 0.5364
      hist_gb   0.9277       0.0030  0.8272    0.2782    0.8720 0.2893
random_forest   0.9166       0.0022  0.8044    0.3003    0.8637 0.7991
       logreg   0.9068       0.0032  0.7661    0.3180    0.8512 0.2489
```

## Frozen model, test set (evaluated once)
- model: hist_gb_tuned
- threshold: 0.3778
- test_roc_auc: 0.9291
- test_pr_auc: 0.83
- precision@thr: 0.7026
- recall@thr: 0.7567
- accuracy@0.5: 0.8727
- brier: 0.0877
- bootstrap 95% ROC-AUC interval: [0.9237, 0.9331]

## Slices
### sex
```
             n  pos_rate  pred_pos_rate  recall  precision
g                                                         
Female  4064.0     0.110          0.103   0.661      0.708
Male    8147.0     0.304          0.335   0.774      0.702
```
### race
```
                          n  pos_rate  pred_pos_rate  recall  precision
g                                                                      
Amer-Indian-Eskimo    113.0     0.115          0.133   0.538      0.467
Asian-Pac-Islander    371.0     0.267          0.280   0.697      0.663
Black                1175.0     0.109          0.113   0.672      0.647
Other                 103.0     0.146          0.117   0.667      0.833
White               10449.0     0.255          0.276   0.765      0.707
```