# Evaluation: linear_probe on `random`

## Headline

| Metric | Value |
|---|---:|
| Test clips | 706 |
| Top-1 accuracy | 0.868 |
| Top-5 accuracy | 0.993 |
| Macro F1 | 0.862 |
| Micro F1 | 0.868 |
| ECE (10 bins) | 0.057 |
| Fairness gap (best - worst signer top-1) | 0.334 |
| HIGH - LOW capture-quality top-1 gap | 0.213 |
| Handedness top-1 gap | 0.065 |

## Per-Signer Fairness

| Signer | Quality | Handedness | N | Top-1 | Top-5 | Abstain | Mean confidence |
|---|---|---|---:|---:|---:|---:|---:|
| S01 | HIGH | right-dominant | 48 | 0.896 | 1.000 | 0.104 | 0.865 |
| S02 | HIGH | right-dominant | 56 | 0.911 | 1.000 | 0.161 | 0.840 |
| S03 | HIGH | right-dominant | 53 | 0.887 | 1.000 | 0.151 | 0.825 |
| S04 | HIGH | left-dominant | 42 | 0.738 | 0.976 | 0.190 | 0.799 |
| S05 | HIGH | right-dominant | 55 | 0.909 | 1.000 | 0.200 | 0.804 |
| S06 | HIGH | right-dominant | 50 | 0.880 | 1.000 | 0.200 | 0.799 |
| S07 | HIGH | right-dominant | 58 | 0.966 | 1.000 | 0.052 | 0.877 |
| S08 | HIGH | right-dominant | 53 | 0.925 | 1.000 | 0.170 | 0.803 |
| S09 | HIGH | right-dominant | 41 | 0.976 | 1.000 | 0.122 | 0.864 |
| S10 | HIGH | right-dominant | 56 | 0.911 | 0.982 | 0.179 | 0.822 |
| S11 | LOW | right-dominant | 53 | 0.642 | 0.981 | 0.340 | 0.728 |
| S12 | LOW | right-dominant | 56 | 0.732 | 0.964 | 0.286 | 0.766 |
| S13 | HIGH | left-dominant | 37 | 0.892 | 1.000 | 0.243 | 0.769 |
| S14 | HIGH | right-dominant | 48 | 0.896 | 1.000 | 0.250 | 0.794 |

## Capture-Quality Tiers

| Tier | Signers | N | Top-1 |
|---|---|---:|---:|
| HIGH | S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S13, S14 | 597 | 0.901 |
| LOW | S11, S12 | 109 | 0.688 |

## Calibration

| Confidence bin | Count | Avg confidence | Accuracy | Gap |
|---|---:|---:|---:|---:|
| 0.0-0.1 | 0 | 0.000 | 0.000 | 0.000 |
| 0.1-0.2 | 0 | 0.000 | 0.000 | 0.000 |
| 0.2-0.3 | 7 | 0.260 | 0.429 | 0.168 |
| 0.3-0.4 | 35 | 0.363 | 0.514 | 0.151 |
| 0.4-0.5 | 34 | 0.445 | 0.500 | 0.055 |
| 0.5-0.6 | 57 | 0.556 | 0.649 | 0.093 |
| 0.6-0.7 | 49 | 0.647 | 0.673 | 0.026 |
| 0.7-0.8 | 68 | 0.750 | 0.882 | 0.132 |
| 0.8-0.9 | 101 | 0.858 | 0.950 | 0.093 |
| 0.9-1.0 | 355 | 0.964 | 0.983 | 0.019 |

## Top-20 Confused Gloss Pairs

| True | Predicted | Count |
|---|---|---:|
| HELP | PLEASE | 8 |
| SAD | BAD | 6 |
| DRINK | PLEASE | 5 |
| HOME | PLEASE | 3 |
| BIG | BAD | 3 |
| NAME | WHERE | 2 |
| COME | MOTHER | 2 |
| COME | BROTHER | 2 |
| WRITE | EVENING | 2 |
| TALK | LOVE | 2 |
| FEEL | SEE | 2 |
| ANGRY | SEE | 2 |
| ANGRY | AGAIN | 2 |
| ANGRY | BAD | 2 |
| BIG | SCHOOL | 2 |
| HOT | WATER | 2 |
| COLD | WORK | 2 |
| DAY | SEE | 2 |
| NIGHT | WATER | 2 |
| HELLO | MOTHER | 1 |

## Per-Class Metrics

| Gloss | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| HELLO | 1.000 | 0.833 | 0.909 | 12 |
| THANK-YOU | 1.000 | 1.000 | 1.000 | 12 |
| HELP | 0.800 | 0.333 | 0.471 | 12 |
| WATER | 0.688 | 1.000 | 0.815 | 11 |
| NAME | 1.000 | 0.833 | 0.909 | 12 |
| LEARN | 0.923 | 1.000 | 0.960 | 12 |
| YES | 1.000 | 0.909 | 0.952 | 11 |
| NO | 0.750 | 1.000 | 0.857 | 12 |
| PLEASE | 0.419 | 1.000 | 0.591 | 13 |
| SORRY | 0.769 | 0.909 | 0.833 | 11 |
| MORE | 1.000 | 1.000 | 1.000 | 11 |
| FINISH | 1.000 | 1.000 | 1.000 | 11 |
| EAT | 1.000 | 0.750 | 0.857 | 12 |
| DRINK | 1.000 | 0.545 | 0.706 | 11 |
| GO | 0.923 | 1.000 | 0.960 | 12 |
| COME | 1.000 | 0.545 | 0.706 | 11 |
| WANT | 1.000 | 1.000 | 1.000 | 11 |
| NEED | 1.000 | 1.000 | 1.000 | 11 |
| FRIEND | 1.000 | 0.909 | 0.952 | 11 |
| FAMILY | 0.857 | 1.000 | 0.923 | 12 |
| MOTHER | 0.706 | 1.000 | 0.828 | 12 |
| FATHER | 1.000 | 0.923 | 0.960 | 13 |
| BROTHER | 0.857 | 1.000 | 0.923 | 12 |
| SISTER | 1.000 | 1.000 | 1.000 | 12 |
| HOME | 1.000 | 0.583 | 0.737 | 12 |
| SCHOOL | 0.727 | 0.727 | 0.727 | 11 |
| WORK | 0.733 | 0.917 | 0.815 | 12 |
| PLAY | 0.923 | 1.000 | 0.960 | 12 |
| READ | 0.833 | 0.833 | 0.833 | 12 |
| WRITE | 1.000 | 0.583 | 0.737 | 12 |
| TALK | 0.889 | 0.667 | 0.762 | 12 |
| SEE | 0.625 | 0.833 | 0.714 | 12 |
| LOOK | 0.857 | 1.000 | 0.923 | 12 |
| WAIT | 0.917 | 1.000 | 0.957 | 11 |
| STOP | 1.000 | 1.000 | 1.000 | 12 |
| AGAIN | 0.812 | 1.000 | 0.897 | 13 |
| UNDERSTAND | 0.923 | 1.000 | 0.960 | 12 |
| KNOW | 0.917 | 0.917 | 0.917 | 12 |
| THINK | 1.000 | 1.000 | 1.000 | 11 |
| FEEL | 0.909 | 0.769 | 0.833 | 13 |
| HAPPY | 1.000 | 1.000 | 1.000 | 12 |
| SAD | 1.000 | 0.091 | 0.167 | 11 |
| ANGRY | 1.000 | 0.417 | 0.588 | 12 |
| LOVE | 0.846 | 1.000 | 0.917 | 11 |
| LIKE | 1.000 | 0.818 | 0.900 | 11 |
| GOOD | 0.923 | 1.000 | 0.960 | 12 |
| BAD | 0.458 | 1.000 | 0.629 | 11 |
| BIG | 1.000 | 0.462 | 0.632 | 13 |
| SMALL | 1.000 | 1.000 | 1.000 | 12 |
| HOT | 1.000 | 0.833 | 0.909 | 12 |
| COLD | 0.889 | 0.727 | 0.800 | 11 |
| DAY | 1.000 | 0.769 | 0.870 | 13 |
| NIGHT | 1.000 | 0.833 | 0.909 | 12 |
| TIME | 0.923 | 1.000 | 0.960 | 12 |
| NOW | 0.900 | 0.818 | 0.857 | 11 |
| LATER | 0.923 | 1.000 | 0.960 | 12 |
| MORNING | 1.000 | 1.000 | 1.000 | 12 |
| EVENING | 0.857 | 1.000 | 0.923 | 12 |
| WHERE | 0.786 | 1.000 | 0.880 | 11 |
| WHO | 1.000 | 1.000 | 1.000 | 12 |
