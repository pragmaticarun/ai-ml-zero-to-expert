# Month-3 capstone report: CIFAR-10-small classifier (CPU)

Data: OpenML 40926 (CIFAR-10-small), 15,000 train / 2,500 val / 2,500 test, fixed seed 0. Model: small ResNet, 308,650 params.
Environment: Python 3.13.5, torch 2.15.0+cpu, CPU only.

## Result
Validation acc 0.7656; test acc 0.7736 (test set used once, after all choices were made on validation).

## Training curve (Day 86)
| epoch | train loss | val acc |
|---|---|---|
| 1 | 1.9333 | 0.3408 |
| 2 | 1.5512 | 0.3396 |
| 3 | 1.3334 | 0.3236 |
| 4 | 1.2028 | 0.4800 |
| 5 | 1.1059 | 0.5348 |
| 6 | 1.0131 | 0.5472 |
| 7 | 0.9546 | 0.6152 |
| 8 | 0.8939 | 0.3728 |
| 9 | 0.8408 | 0.6436 |
| 10 | 0.7906 | 0.6288 |
| 11 | 0.7399 | 0.6924 |
| 12 | 0.6916 | 0.6888 |
| 13 | 0.6410 | 0.7300 |
| 14 | 0.5926 | 0.7624 |
| 15 | 0.5579 | 0.7656 |

## Ablations (Day 87, 8 epochs each)
| config | val acc | delta vs full |
|---|---|---|
| full model | 0.7044 | +0.0000 |
| no augmentation | 0.7112 | +0.0068 |
| no residual connections | 0.7108 | +0.0064 |
| no batch norm (lr 0.01) | 0.3536 | -0.3508 |
| half width (16) | 0.6612 | -0.0432 |
| no weight decay | 0.7032 | -0.0012 |
| label smoothing 0.1 | 0.7024 | -0.0020 |
| 50% of training data | 0.6180 | -0.0864 |

## Error analysis
See the Day 88 console output, out/day88_confident_errors.png and out/day88_review_sheet.csv (tag the 60 errors by hand and add the tag counts here).

## Limitations
- Single seed per ablation; differences under about 0.01 are within seed noise (see Day 85).
- 32x32 images and CPU budget; not comparable with published full-CIFAR-10 results.
