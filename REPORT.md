# Report

## Bump Plot

### Settings

All three runs below are **pure-background** Graph-AE training:

- **No 3% S/B contamination**
- Window replacement (keep the 80k / 20k counts by swapping in unused
  out-of-window background):
  - Exclude 3600–4000: replaced **6,251** train-bkg + **1,540** val-bkg;
  - Exclude 3000–4300: replaced **28,518** train-bkg + **7,100** val-bkg.
- Model: EdgeGraphAE / unique-6 / 30 subjets / log edge features,
  `hidden_dim=64`, `latent_dim=2`, no BatchNorm, 50 epochs, seed 123.

### Metrics summary

| Setting | Train/val mJJ exclusion | LHCO AUC / MaxSIC | BB1 AUC / MaxSIC |
|---------|-------------------------|------------------:|-----------------:|
| No mJJ exclude | none | 0.9098 / 2.311 | 0.8891 / 2.484 |
| Exclude 3600–4000 | `3600 ≤ mJJ < 4000` GeV | 0.9103 / 2.319 | 0.8910 / 2.404 |
| Exclude 3000–4300 | `3000 ≤ mJJ < 4300` GeV | 0.9129 / 2.357 | 0.8931 / 2.561 |

For each exclusion setting, in-window train/val background events are replaced
as listed under Settings; training remains pure background.

### BB1 top anomaly-score selections

BB1 $m_{JJ}$ distributions of events in the top anomaly-score fraction
(1% → 0.01%), comparing the three pure-background trainings above
(`last.pt`; 100 GeV bins, Gaussian-smoothed).

Figure: [`figs/bb1_threeway_top1_to_0p01_row.png`](figs/bb1_threeway_top1_to_0p01_row.png)

<img src="figs/bb1_threeway_top1_to_0p01_row.png" alt="BB1 three-way top-score row" width="100%" />

#### Exclude `3600 ≤ mJJ < 4000`

Baseline vs window-excluded training; shaded band marks the excluded train window.

Figure: [`figs/bb1_exclude3600_4000_top1_to_0p01_row.png`](figs/bb1_exclude3600_4000_top1_to_0p01_row.png)

<img src="figs/bb1_exclude3600_4000_top1_to_0p01_row.png" alt="BB1 exclude 3600-4000 top-score row" width="100%" />

#### Exclude `3000 ≤ mJJ < 4300`

Baseline vs window-excluded training; shaded band marks the excluded train window.

Figure: [`figs/bb1_exclude3000_4300_top1_to_0p01_row.png`](figs/bb1_exclude3000_4300_top1_to_0p01_row.png)

<img src="figs/bb1_exclude3000_4300_top1_to_0p01_row.png" alt="BB1 exclude 3000-4300 top-score row" width="100%" />

