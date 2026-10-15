# SDM–ATII: what we tried and learned

We implemented [SDM–ATII from the ICML 2025 paper](https://proceedings.mlr.press/v267/cai25k.html),
then adapted its generator, critic and anomaly score to jet graphs.

## 1. Experiment settings

- **Initial exploration:** `Nsig=0`, with 60,000 background training events and 60,000
  background validation events in the SR (`3.3 < mJJ < 3.7 TeV`). The paper
  assumes normal-only training; here, normal graphs are background jets.
- **Evaluation:** one background-trained model tested on LHCO, BB1 and all four
  KS signals. LHCO and each KS test contain 340,000 background and 20,000 signal
  events in the SR.
- **Protocol:** seed 42, 50 epochs, evaluation every 5 epochs; report epoch 50.
- **Metrics:** AUC / maxSIC, using the same KS background-statistics cut as the
  [results overview](results.md).

Sections 2–6 describe background-only training. Section 7 repeats the retained
versions with `Nsig=1000`, training separately for each signal.

These benchmarks guided the exploration; the results below are development
comparisons, not an untouched final holdout.

## 2. Start with paper-based ATII

- **Input:** 30-subjet graphs, node `ln(pT)` and binary Unique-6 adjacency.
- **Generator:** GIN Gaussian encoders sample node latents, then reconstruct
  node features and a continuous adjacency matrix.
- **Critic:** a separate GIN with Global Sum Pooling scores real and generated
  graphs. Alternate generator and critic updates.
- **Training:** RMSprop at `0.001`, batch 128 events; discrepancy and KL weights
  both `1`, with critic parameters clipped to `[-0.01, 0.01]`.
- **Event score:** `-D(jet0) - D(jet1)`; higher means more anomalous.

Implementation choices: three 16-channel GIN layers, concatenated to 48 channels;
10-dimensional node-wise Gaussian latents; node decoder `10 → 16 → 1` and
critic head `48 → 16 → 1`. Node squared error, adjacency BCE and both Gaussian
KL terms are summed within each jet, then averaged over jets. Padding and
diagonal pairs are excluded. These architecture and reduction choices are ours.

The result is strongly signal-dependent: LHCO AUC is **0.6734**, while Z′ and
YHH reach **0.9431** and **0.9833**. The critic stops changing in later epochs;
a diagnostic update is completely removed by weight clipping. Generator
gradients are also dominated by KL rather than the adversarial term.

Changing to a stabilized GIN, NetGe or EdgeConv scalar critic does not solve
this: LHCO AUC is **0.5226, 0.3947 and 0.3562**, respectively.

## 3. Replace the scalar critic with reconstruction error

The **Energy** adaptation keeps alternating generator/critic training, but
uses an autoencoder's reconstruction error as the critic. This follows the
[EBGAN energy idea](https://arxiv.org/abs/1609.03126), rather than the original
ATII critic.

- Use **NetGe max-message + symmetric linear/bilinear edge decoder**, with a
  two-dimensional node latent. Generator and critic have separate parameters.
- Standardize node `ln(pT)` and physical-edge `ln(theta)`, `ln(kT)`, `ln(z)`
  using training-set means and standard deviations.
- Keep Unique-6 connections fixed; generate node and edge features, not adjacency.
- Define energy `E(G)` as node MSE + physical-edge MSE. Average errors within
  each jet, then average over jets for training.
- Use Gaussian node latents in the generator, with log standard deviation
  bounded to `[-3, 1]`; reduce the KL weight to `0.01`.

The retained training recipe is:

```text
Generator: E(generated) + node/edge discrepancy + 0.01 × KL
Critic:    E(real) + 0.1 × max(0, 0.2 − E(generated))
```

Use Adam at `0.0002`, betas `(0, 0.9)`, cosine decay to `0.000002`, batch
256 events and gradient clipping at `1`. This replaces critic weight clipping.

With critic-energy scoring, LHCO reaches **0.9227 / 2.655**, versus
**0.9210 / 2.578** for the matched AE-only control. However, all four KS signals
are slightly worse than AE-only: adversarial training alone is not a consistent gain.

## 4. Change what we score

On the same 30-subjet Energy checkpoint, we compared three jet scores:

1. **Critic energy:** reconstruction error of the input graph, `E(G)`.
2. **Generator score:** node/edge discrepancy + `0.01 × KL`.
3. **Full generator score:** generator score + `E(reconstructed graph)`.

Generator scores use the posterior mean, without random sampling. Sum the
two jet scores to obtain the event score.

The generator score gives LHCO **0.9271 / 2.781** and BB1 SR **0.8740 / 2.640**.
Adding the reconstructed graph's energy changes the ranking: Z′ AUC rises
**0.8925 → 0.9417** and YHH **0.8594 → 0.9458**, but LHCO falls to **0.9140**
and BB1 SR to **0.8411**. All three scores use the same trained weights.

## 5. Increase resolution and add local edge information

- **Energy 60:** raise the maximum from 30 to 60 subjets. Keep the architecture,
  training recipe and full generator score fixed; refit feature scaling on
  the 60-subjet training graphs.
- **Energy 60 + critic context:** average the physical-edge features incident
  on each node, project them to 32 channels and add them to the critic's node
  embedding before GELU. This adds **96 parameters**. Max-message aggregation,
  generator architecture, losses and score stay unchanged.

The added local information improves all five LHCO/KS AUCs and maxSICs over
Energy 60, but lowers BB1 SR AUC.

## 6. Final results: background-only training

All Energy rows below use the **full generator score**.

| Model | LHCO | WR | YY′ | Z′ | YHH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original ATII, 30 | 0.6734 / 1.053 | 0.6094 / 1.035 | 0.7125 / 1.134 | 0.9431 / 2.806 | 0.9833 / 6.872 |
| Energy 30 | 0.9140 / 2.395 | 0.8583 / 1.760 | 0.8750 / 1.994 | 0.9417 / 2.690 | 0.9458 / 2.809 |
| Energy 60 | 0.9397 / 3.772 | 0.9073 / 2.500 | 0.9053 / 2.672 | 0.9246 / 2.405 | 0.9112 / 2.177 |
| Energy 60 + critic context | 0.9473 / 4.927 | 0.9178 / 3.444 | 0.9238 / 3.689 | 0.9385 / 2.723 | 0.9419 / 2.731 |

BB1 full range contains 999,166 background and 834 signal events. Its SR
contains 120,279 background and 223 signal events; apply the SR cut to the
same event scores.

| Model | BB1 full range | BB1 SR |
| --- | ---: | ---: |
| Original ATII, 30 | 0.7376 / 1.251 | 0.6284 / 1.064 |
| Energy 30 | 0.9133 / 3.688 | 0.8411 / 1.895 |
| Energy 60 | 0.9164 / 3.894 | 0.8740 / 2.742 |
| Energy 60 + critic context | 0.9120 / 3.974 | 0.8605 / 2.729 |

## What we learned

- The original ATII critic performs poorly on several signals but remains
  especially strong on Z′ and YHH.
- Reconstruction-based critics are much more useful here than the tested
  scalar critics. Their improvement cannot be attributed to adversarial training alone.
- Generator scoring and higher subjet resolution give substantial gains,
  but neither improves every signal.
- With background-only training, local edge information helps all five LHCO/KS
  signals but lowers BB1 SR AUC. The mixed-training repeat below is less uniform.

## 7. Repeat with Nsig=1000

All **20 models** completed 50 epochs: four versions × five training signals.

- Train separately for LHCO, WR, YY′, Z′ and YHH, seed 42.
- Use each signal's SR mixture, retaining its injected signals. Truth labels
  do not enter the loss or feature-scaling fit.
- Keep each version's architecture, optimizer, loss and score unchanged.
- Evaluate the corresponding signal every 5 epochs; report epoch 50 and
  additionally evaluate BB1 full range and SR.

`Nsig=1000` specifies injection before mass cuts and splitting. Each model uses
60,000 SR training and 60,000 SR validation events. The training subsets contain
381, 398, 319, 396 and 366 signals for LHCO, WR, YY′, Z′ and YHH, respectively.

| Model | LHCO | WR | YY′ | Z′ | YHH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original ATII, 30 | 0.6783 / 1.057 | 0.6277 / 1.044 | 0.7150 / 1.141 | 0.9438 / 2.832 | 0.9835 / 6.901 |
| Energy 30 | 0.9202 / 2.525 | 0.8668 / 1.841 | 0.8758 / 2.001 | 0.9437 / 2.730 | 0.9381 / 2.617 |
| Energy 60 | 0.9388 / 3.732 | 0.9043 / 2.423 | 0.9050 / 2.634 | 0.9277 / 2.467 | 0.9301 / 2.455 |
| Energy 60 + critic context | 0.9474 / 4.715 | 0.9133 / 2.841 | 0.9286 / 3.739 | 0.9147 / 2.310 | 0.9060 / 2.105 |

BB1 uses the **LHCO-trained checkpoint** for each version:

| Model | BB1 full range | BB1 SR |
| --- | ---: | ---: |
| Original ATII, 30 | 0.7386 / 1.255 | 0.6291 / 1.065 |
| Energy 30 | 0.9141 / 3.808 | 0.8430 / 2.045 |
| Energy 60 | 0.9163 / 3.935 | 0.8720 / 2.649 |
| Energy 60 + critic context | 0.9122 / 3.854 | 0.8656 / 2.646 |

Both 60-subjet adaptations exceed 0.90 AUC on all five LHCO/KS signals.
Critic context improves LHCO, WR and YY′, but lowers Z′ and YHH compared with
Energy 60. Keep both as a trade-off, not a universal improvement. Original ATII
still gives the strongest YHH result in this comparison; BB1 SR remains below 0.90.

![ATII adaptations, baseline, NetGe, GLADC and KS: AUC and maxSIC](../assets/atii.png)

## 8. Run the retained versions

The [training script](../scripts/atii.sh) uses the existing train and inference
entry points. Choose `original`, `energy` or `context`; `SUBJETS` selects 30 or 60.

For 60 subjets, build the graphs once in a separate directory:

```bash
python -m scripts.build_graph --max-subjets 60 --output-dir data/graphs60
```

Both resolutions use pT-centred coordinates. Training reads the subjet count
from the graph file.

Train and evaluate one signal:

```bash
VARIANT=original SIGNAL=LHCO DEVICE=cuda:0 bash scripts/atii.sh
VARIANT=energy SUBJETS=30 SIGNAL=LHCO DEVICE=cuda:0 bash scripts/atii.sh
VARIANT=energy SUBJETS=60 SIGNAL=LHCO DEVICE=cuda:0 bash scripts/atii.sh
VARIANT=context SUBJETS=60 SIGNAL=LHCO DEVICE=cuda:0 bash scripts/atii.sh
```

To evaluate a saved model, use the existing inference entry point:

```bash
python -m scripts.inference --checkpoint "$CHECKPOINT" \
  --split-file data/splits/CWoLa_LHCO_nsig1000.npz \
  --output "$OUTPUT" --device cuda:0
```

It uses the checkpoint's graph directory and score. Exported research weights
are named `inference.pt`; new training runs save `last.pt`.
