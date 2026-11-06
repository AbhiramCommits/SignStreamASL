# Evaluation: bigru on `random`

## Headline

| Metric | Value |
|---|---:|
| Test clips | 706 |
| Top-1 accuracy | 0.816 |
| Top-5 accuracy | 0.984 |
| Macro F1 | 0.812 |
| Micro F1 | 0.816 |
| ECE (10 bins) | 0.300 |
| Fairness gap (best - worst signer top-1) | 0.513 |
| HIGH - LOW capture-quality top-1 gap | -0.098 |
| Handedness top-1 gap | 0.349 |

## Per-Signer Fairness

| Signer | Quality | Handedness | N | Top-1 | Top-5 | Abstain | Mean confidence |
|---|---|---|---:|---:|---:|---:|---:|
| S01 | HIGH | right-dominant | 48 | 0.917 | 1.000 | 0.625 | 0.548 |
| S02 | HIGH | right-dominant | 56 | 0.857 | 1.000 | 0.661 | 0.536 |
| S03 | HIGH | right-dominant | 53 | 0.887 | 1.000 | 0.585 | 0.565 |
| S04 | HIGH | left-dominant | 42 | 0.452 | 0.857 | 0.929 | 0.349 |
| S05 | HIGH | right-dominant | 55 | 0.873 | 1.000 | 0.709 | 0.528 |
| S06 | HIGH | right-dominant | 50 | 0.820 | 1.000 | 0.700 | 0.504 |
| S07 | HIGH | right-dominant | 58 | 0.966 | 1.000 | 0.517 | 0.589 |
| S08 | HIGH | right-dominant | 53 | 0.774 | 1.000 | 0.698 | 0.488 |
| S09 | HIGH | right-dominant | 41 | 0.805 | 1.000 | 0.537 | 0.570 |
| S10 | HIGH | right-dominant | 56 | 0.839 | 1.000 | 0.625 | 0.541 |
| S11 | LOW | right-dominant | 53 | 0.887 | 1.000 | 0.717 | 0.522 |
| S12 | LOW | right-dominant | 56 | 0.911 | 1.000 | 0.571 | 0.564 |
| S13 | HIGH | left-dominant | 37 | 0.568 | 0.865 | 1.000 | 0.301 |
| S14 | HIGH | right-dominant | 48 | 0.688 | 1.000 | 0.708 | 0.520 |

## Capture-Quality Tiers

| Tier | Signers | N | Top-1 |
|---|---|---:|---:|
| HIGH | S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S13, S14 | 597 | 0.801 |
| LOW | S11, S12 | 109 | 0.899 |

## Calibration

| Confidence bin | Count | Avg confidence | Accuracy | Gap |
|---|---:|---:|---:|---:|
| 0.0-0.1 | 0 | 0.000 | 0.000 | 0.000 |
| 0.1-0.2 | 12 | 0.173 | 0.333 | 0.160 |
| 0.2-0.3 | 65 | 0.256 | 0.431 | 0.175 |
| 0.3-0.4 | 113 | 0.356 | 0.681 | 0.326 |
| 0.4-0.5 | 148 | 0.448 | 0.730 | 0.282 |
| 0.5-0.6 | 138 | 0.547 | 0.957 | 0.409 |
| 0.6-0.7 | 109 | 0.648 | 0.982 | 0.334 |
| 0.7-0.8 | 88 | 0.743 | 0.989 | 0.246 |
| 0.8-0.9 | 33 | 0.831 | 1.000 | 0.169 |
| 0.9-1.0 | 0 | 0.000 | 0.000 | 0.000 |

## Top-20 Confused Gloss Pairs

| True | Predicted | Count |
|---|---|---:|
| COME | AGAIN | 6 |
| THANK-YOU | LEARN | 5 |
| YES | WHERE | 5 |
| TALK | MOTHER | 5 |
| LOVE | FATHER | 5 |
| LATER | GO | 5 |
| NEED | ANGRY | 4 |
| LOOK | FRIEND | 4 |
| TIME | STOP | 4 |
| MOTHER | FINISH | 3 |
| WRITE | MORNING | 3 |
| FEEL | SEE | 3 |
| SAD | ANGRY | 3 |
| NAME | WHERE | 2 |
| YES | NAME | 2 |
| PLEASE | FAMILY | 2 |
| SORRY | SMALL | 2 |
| DRINK | PLEASE | 2 |
| DRINK | FAMILY | 2 |
| GO | LATER | 2 |

## Per-Class Metrics

| Gloss | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| HELLO | 1.000 | 1.000 | 1.000 | 12 |
| THANK-YOU | 0.857 | 0.500 | 0.632 | 12 |
| HELP | 1.000 | 0.833 | 0.909 | 12 |
| WATER | 0.900 | 0.818 | 0.857 | 11 |
| NAME | 0.769 | 0.833 | 0.800 | 12 |
| LEARN | 0.611 | 0.917 | 0.733 | 12 |
| YES | 1.000 | 0.364 | 0.533 | 11 |
| NO | 0.857 | 1.000 | 0.923 | 12 |
| PLEASE | 0.727 | 0.615 | 0.667 | 13 |
| SORRY | 0.889 | 0.727 | 0.800 | 11 |
| MORE | 1.000 | 0.909 | 0.952 | 11 |
| FINISH | 0.786 | 1.000 | 0.880 | 11 |
| EAT | 0.857 | 1.000 | 0.923 | 12 |
| DRINK | 1.000 | 0.455 | 0.625 | 11 |
| GO | 0.667 | 0.833 | 0.741 | 12 |
| COME | 1.000 | 0.455 | 0.625 | 11 |
| WANT | 0.833 | 0.909 | 0.870 | 11 |
| NEED | 0.571 | 0.364 | 0.444 | 11 |
| FRIEND | 0.688 | 1.000 | 0.815 | 11 |
| FAMILY | 0.588 | 0.833 | 0.690 | 12 |
| MOTHER | 0.643 | 0.750 | 0.692 | 12 |
| FATHER | 0.625 | 0.769 | 0.690 | 13 |
| BROTHER | 1.000 | 0.917 | 0.957 | 12 |
| SISTER | 1.000 | 0.917 | 0.957 | 12 |
| HOME | 0.769 | 0.833 | 0.800 | 12 |
| SCHOOL | 0.818 | 0.818 | 0.818 | 11 |
| WORK | 1.000 | 0.917 | 0.957 | 12 |
| PLAY | 1.000 | 0.917 | 0.957 | 12 |
| READ | 1.000 | 1.000 | 1.000 | 12 |
| WRITE | 0.778 | 0.583 | 0.667 | 12 |
| TALK | 1.000 | 0.500 | 0.667 | 12 |
| SEE | 0.800 | 1.000 | 0.889 | 12 |
| LOOK | 1.000 | 0.667 | 0.800 | 12 |
| WAIT | 1.000 | 1.000 | 1.000 | 11 |
| STOP | 0.632 | 1.000 | 0.774 | 12 |
| AGAIN | 0.591 | 1.000 | 0.743 | 13 |
| UNDERSTAND | 1.000 | 1.000 | 1.000 | 12 |
| KNOW | 0.786 | 0.917 | 0.846 | 12 |
| THINK | 0.917 | 1.000 | 0.957 | 11 |
| FEEL | 1.000 | 0.692 | 0.818 | 13 |
| HAPPY | 1.000 | 0.917 | 0.957 | 12 |
| SAD | 1.000 | 0.636 | 0.778 | 11 |
| ANGRY | 0.533 | 0.667 | 0.593 | 12 |
| LOVE | 0.750 | 0.545 | 0.632 | 11 |
| LIKE | 1.000 | 0.818 | 0.900 | 11 |
| GOOD | 0.800 | 1.000 | 0.889 | 12 |
| BAD | 0.846 | 1.000 | 0.917 | 11 |
| BIG | 1.000 | 0.846 | 0.917 | 13 |
| SMALL | 0.800 | 1.000 | 0.889 | 12 |
| HOT | 0.909 | 0.833 | 0.870 | 12 |
| COLD | 0.875 | 0.636 | 0.737 | 11 |
| DAY | 1.000 | 0.846 | 0.917 | 13 |
| NIGHT | 0.923 | 1.000 | 0.960 | 12 |
| TIME | 1.000 | 0.583 | 0.737 | 12 |
| NOW | 0.769 | 0.909 | 0.833 | 11 |
| LATER | 0.417 | 0.417 | 0.417 | 12 |
| MORNING | 0.750 | 0.750 | 0.750 | 12 |
| EVENING | 1.000 | 1.000 | 1.000 | 12 |
| WHERE | 0.524 | 1.000 | 0.688 | 11 |
| WHO | 0.917 | 0.917 | 0.917 | 12 |
