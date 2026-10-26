"""Physics-informed AE on the official Kitchen Sink CWoLa splits (LHCO, Nsig=1000, runs 0-4).

Run r rebuilds the Kitchen Sink pool and test signals with its seeds (signal shuffle 1934 + r;
mixture shuffle s_0 = 1001, s_r = s_(r-1) + r + 1000), retrains the AE on the SR events of a
stratified half of that pool, and trains the readout with the run-r member seeds. The 340k test
background events are the same for every run. As in the original study, every run reuses the input
normalization and target standardization fitted on the LHCO (run 0) training SR events, and each
training batch is processed in sorted event order.

    python -m scripts.ks_runs --runs 0 1 2 3 4 --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.physics_ae import lhco_training, load_packed, pack, readout_inputs, split_path, statistics, train
from src.metrics import score_metrics
from src.readout import ensemble_scores
from src.splits import EVENT_DTYPE, SR_EVENT_CAP, TEST_SIGNAL_CAP, signal_region


def mixture_seed(run):
    seed = 1001
    for step in range(1, run + 1):
        seed += step + 1000
    return seed


def samples(run, n_signal=1000, feature_dir=Path('data/features/kitchensink')):
    """CWoLa pool (Kitchen Sink order) and test signals of run r, as LHCO table events."""
    frame = pd.read_hdf(feature_dir/'LHCO'/'baseline.h5')
    events = np.zeros(len(frame), dtype=EVENT_DTYPE)
    events['source'], events['row'] = 0, np.arange(len(frame))
    events['mjj'], events['truth'], events['weak_label'] = frame['mjj'], frame['label'], -1
    background, signal = events[events['truth'] == 0], events[events['truth'] == 1].copy()
    np.random.RandomState(1934 + run).shuffle(signal)
    mixture = np.concatenate([background, signal[:n_signal]])
    rng = np.random.RandomState(mixture_seed(run))
    rng.shuffle(mixture)
    in_sr = signal_region(mixture)
    sr = mixture[in_sr][:SR_EVENT_CAP].copy()
    sideband = mixture[~in_sr & (mixture['mjj'] > 3.1) & (mixture['mjj'] < 3.9)].copy()
    sr['weak_label'], sideband['weak_label'] = 1, 0
    pool = np.concatenate([sr, sideband])
    np.random.RandomState(rng.randint(300)).shuffle(pool)
    held_out = signal[n_signal:]
    return pool, held_out[signal_region(held_out)][:TEST_SIGNAL_CAP]


def prepare(run, split_dir, packed_dir):
    """Write the run split (shared test background from the LHCO split) and pack its graphs."""
    pool, test_signal = samples(run)
    with np.load(split_path('LHCO')) as split:
        background = split['test'][split['test']['source'] == 1]
        files = split['source_files'].copy()
    split_file = split_dir/f'run{run}.npz'
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez(split_file, pool=pool, test=np.concatenate([background, test_signal]), source_files=files)
    pack(split_file, packed_dir/f'run{run}', parts=('pool', 'test'))
    return split_file


def run_one(run, *, stats, device, seed, split_dir, packed_dir, run_dir):
    split_file = split_dir/f'run{run}.npz'
    if not (packed_dir/f'run{run}'/'test.pt').exists():
        split_file = prepare(run, split_dir, packed_dir)
    data, events = load_packed(packed_dir/f'run{run}', 'pool')
    half, other = train_test_split(np.arange(len(events)), test_size=.5, stratify=events['weak_label'],
                                   random_state=mixture_seed(run))
    sr = events['weak_label'] == 1
    model = train(data, events, np.sort(half[sr[half]]), np.sort(other[sr[other]]), run_dir/f'run{run}',
                  seed=seed, device=device, stats=stats, sort_batches=True).eval()
    with np.load(split_file) as split:
        paths = [Path('data/graphs')/name for name in split['source_files']]
    test_data, test_events = load_packed(packed_dir/f'run{run}', 'test')
    scores = ensemble_scores(readout_inputs(model, data, events, paths, device), events['weak_label'],
                             readout_inputs(model, test_data, test_events, paths, device), run=run,
                             model_dir=run_dir/f'run{run}'/'readout')
    metrics = score_metrics(scores, test_events['truth'])
    np.save(run_dir/f'run{run}'/'test_scores.npy', scores)
    (run_dir/f'run{run}'/'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print('run', run, 'AUC %.4f maxSIC %.2f maxSIC(KS) %.2f' % (metrics['auc'], metrics['max_sic'], metrics['max_sic_ks']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--split-dir', type=Path, default=Path('data/splits/ks_runs'))
    parser.add_argument('--packed-dir', type=Path, default=Path('data/packed/ks_runs'))
    parser.add_argument('--run-dir', type=Path, default=Path('runs/physics_ae/ks_runs'))
    parser.add_argument('--lhco-packed-dir', type=Path, default=Path('data/packed'))
    args = parser.parse_args()
    data, events, train_rows, _ = lhco_training(args.lhco_packed_dir)
    stats = statistics(data, events, train_rows, args.device)
    del data
    for run in args.runs:
        run_one(run, stats=stats, device=args.device, seed=args.seed, split_dir=args.split_dir,
                packed_dir=args.packed_dir, run_dir=args.run_dir)


if __name__ == '__main__':
    main()
