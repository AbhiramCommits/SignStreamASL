# Evaluation: pose_transformer on `random`

## Headline

| Metric | Value |
|---|---:|
| Test clips | 706 |
| Top-1 accuracy | 0.837 |
| Top-5 accuracy | 0.997 |
| Macro F1 | 0.833 |
| Micro F1 | 0.837 |
| ECE (10 bins) | 0.213 |
| Fairness gap (best - worst signer top-1) | 0.336 |
| HIGH - LOW capture-quality top-1 gap | -0.041 |
| Handedness top-1 gap | 0.258 |

## Per-Signer Fairness

| Signer | Quality | Handedness | N | Top-1 | Top-5 | Abstain | Mean confidence |
|---|---|---|---:|---:|---:|---:|---:|
| S01 | HIGH | right-dominant | 48 | 0.896 | 1.000 | 0.354 | 0.664 |
| S02 | HIGH | right-dominant | 56 | 0.875 | 1.000 | 0.464 | 0.629 |
| S03 | HIGH | right-dominant | 53 | 0.811 | 1.000 | 0.396 | 0.642 |
| S04 | HIGH | left-dominant | 42 | 0.619 | 1.000 | 0.762 | 0.473 |
| S05 | HIGH | right-dominant | 55 | 0.873 | 1.000 | 0.491 | 0.620 |
| S06 | HIGH | right-dominant | 50 | 0.840 | 1.000 | 0.340 | 0.632 |
| S07 | HIGH | right-dominant | 58 | 0.931 | 1.000 | 0.379 | 0.656 |
| S08 | HIGH | right-dominant | 53 | 0.774 | 1.000 | 0.396 | 0.665 |
| S09 | HIGH | right-dominant | 41 | 0.902 | 1.000 | 0.317 | 0.673 |
| S10 | HIGH | right-dominant | 56 | 0.893 | 1.000 | 0.357 | 0.664 |
| S11 | LOW | right-dominant | 53 | 0.868 | 1.000 | 0.415 | 0.645 |
| S12 | LOW | right-dominant | 56 | 0.875 | 1.000 | 0.393 | 0.662 |
| S13 | HIGH | left-dominant | 37 | 0.595 | 0.946 | 0.757 | 0.494 |
| S14 | HIGH | right-dominant | 48 | 0.854 | 1.000 | 0.417 | 0.614 |

## Capture-Quality Tiers

| Tier | Signers | N | Top-1 |
|---|---|---:|---:|
| HIGH | S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S13, S14 | 597 | 0.831 |
| LOW | S11, S12 | 109 | 0.872 |

## Calibration

| Confidence bin | Count | Avg confidence | Accuracy | Gap |
|---|---:|---:|---:|---:|
| 0.0-0.1 | 0 | 0.000 | 0.000 | 0.000 |
| 0.1-0.2 | 0 | 0.000 | 0.000 | 0.000 |
| 0.2-0.3 | 10 | 0.267 | 0.100 | 0.167 |
| 0.3-0.4 | 63 | 0.359 | 0.540 | 0.181 |
| 0.4-0.5 | 109 | 0.457 | 0.606 | 0.148 |
| 0.5-0.6 | 126 | 0.550 | 0.849 | 0.299 |
| 0.6-0.7 | 129 | 0.652 | 0.891 | 0.240 |
| 0.7-0.8 | 140 | 0.753 | 0.993 | 0.240 |
| 0.8-0.9 | 113 | 0.842 | 1.000 | 0.158 |
| 0.9-1.0 | 16 | 0.914 | 1.000 | 0.086 |

## Top-20 Confused Gloss Pairs

| True | Predicted | Count |
|---|---|---:|
| FATHER | KNOW | 8 |
| LATER | GO | 8 |
| NEED | ANGRY | 7 |
| FAMILY | HOME | 7 |
| TIME | HAPPY | 7 |
| COME | AGAIN | 6 |
| WRITE | MORNING | 6 |
| LEARN | THANK-YOU | 5 |
| THANK-YOU | LEARN | 3 |
| LOOK | FRIEND | 3 |
| FEEL | GO | 3 |
| NAME | WHERE | 2 |
| YES | WHERE | 2 |
| DRINK | WANT | 2 |
| MOTHER | FINISH | 2 |
| HOME | FAMILY | 2 |
| SCHOOL | NOW | 2 |
| TALK | MOTHER | 2 |
| WAIT | MORE | 2 |
| FEEL | LATER | 2 |

## Per-Class Metrics

| Gloss | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| HELLO | 0.857 | 1.000 | 0.923 | 12 |
| THANK-YOU | 0.643 | 0.750 | 0.692 | 12 |
| HELP | 1.000 | 1.000 | 1.000 | 12 |
| WATER | 0.909 | 0.909 | 0.909 | 11 |
| NAME | 1.000 | 0.833 | 0.909 | 12 |
| LEARN | 0.700 | 0.583 | 0.636 | 12 |
| YES | 0.900 | 0.818 | 0.857 | 11 |
| NO | 1.000 | 1.000 | 1.000 | 12 |
| PLEASE | 1.000 | 0.846 | 0.917 | 13 |
| SORRY | 0.846 | 1.000 | 0.917 | 11 |
| MORE | 0.846 | 1.000 | 0.917 | 11 |
| FINISH | 0.833 | 0.909 | 0.870 | 11 |
| EAT | 0.923 | 1.000 | 0.960 | 12 |
| DRINK | 1.000 | 0.727 | 0.842 | 11 |
| GO | 0.478 | 0.917 | 0.629 | 12 |
| COME | 1.000 | 0.455 | 0.625 | 11 |
| WANT | 0.786 | 1.000 | 0.880 | 11 |
| NEED | 0.750 | 0.273 | 0.400 | 11 |
| FRIEND | 0.786 | 1.000 | 0.880 | 11 |
| FAMILY | 0.714 | 0.417 | 0.526 | 12 |
| MOTHER | 0.750 | 0.750 | 0.750 | 12 |
| FATHER | 0.833 | 0.385 | 0.526 | 13 |
| BROTHER | 0.900 | 0.750 | 0.818 | 12 |
| SISTER | 1.000 | 0.917 | 0.957 | 12 |
| HOME | 0.588 | 0.833 | 0.690 | 12 |
| SCHOOL | 0.900 | 0.818 | 0.857 | 11 |
| WORK | 1.000 | 0.917 | 0.957 | 12 |
| PLAY | 1.000 | 0.917 | 0.957 | 12 |
| READ | 1.000 | 0.917 | 0.957 | 12 |
| WRITE | 1.000 | 0.500 | 0.667 | 12 |
| TALK | 0.909 | 0.833 | 0.870 | 12 |
| SEE | 1.000 | 1.000 | 1.000 | 12 |
| LOOK | 1.000 | 0.750 | 0.857 | 12 |
| WAIT | 1.000 | 0.818 | 0.900 | 11 |
| STOP | 0.800 | 1.000 | 0.889 | 12 |
| AGAIN | 0.632 | 0.923 | 0.750 | 13 |
| UNDERSTAND | 0.923 | 1.000 | 0.960 | 12 |
| KNOW | 0.550 | 0.917 | 0.688 | 12 |
| THINK | 1.000 | 1.000 | 1.000 | 11 |
| FEEL | 1.000 | 0.615 | 0.762 | 13 |
| HAPPY | 0.632 | 1.000 | 0.774 | 12 |
| SAD | 0.900 | 0.818 | 0.857 | 11 |
| ANGRY | 0.556 | 0.833 | 0.667 | 12 |
| LOVE | 1.000 | 0.818 | 0.900 | 11 |
| LIKE | 0.833 | 0.909 | 0.870 | 11 |
| GOOD | 1.000 | 1.000 | 1.000 | 12 |
| BAD | 1.000 | 1.000 | 1.000 | 11 |
| BIG | 0.929 | 1.000 | 0.963 | 13 |
| SMALL | 1.000 | 0.917 | 0.957 | 12 |
| HOT | 1.000 | 0.833 | 0.909 | 12 |
| COLD | 0.889 | 0.727 | 0.800 | 11 |
| DAY | 1.000 | 0.923 | 0.960 | 13 |
| NIGHT | 1.000 | 1.000 | 1.000 | 12 |
| TIME | 1.000 | 0.333 | 0.500 | 12 |
| NOW | 0.733 | 1.000 | 0.846 | 11 |
| LATER | 0.444 | 0.333 | 0.381 | 12 |
| MORNING | 0.647 | 0.917 | 0.759 | 12 |
| EVENING | 0.917 | 0.917 | 0.917 | 12 |
| WHERE | 0.688 | 1.000 | 0.815 | 11 |
| WHO | 1.000 | 1.000 | 1.000 | 12 |
