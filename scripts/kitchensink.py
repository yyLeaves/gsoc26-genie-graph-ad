"""Train and evaluate the single-seed KS classifier on prepared native-table splits."""

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from src.metrics import score_metrics


SEED = 42
SIGNALS = ('LHCO', 'XtoWRto3W', 'XtoYYprime', 'ZpToTpTp', 'YtoHHto4T')
# Source codes in the prepared events.npz files, not graph-file row identities.
SOURCES = ('LHCO', 'extra_qcd', *SIGNALS[1:])
FEATURE_SETS = ('baseline', 'subjettiness', 'efp', 'combined')


def load_features(feature_dir, feature):
    """Load author-built columns in their saved order; EFP reuses the Combined cache."""
    if feature in ('baseline', 'subjettiness'):
        frame = pd.read_hdf(feature_dir / 'LHCO' / f'{feature}.h5', start=0, stop=1)
        columns = list(frame.columns[1:-1])
    else:
        columns = json.loads((feature_dir / 'LHCO/combined.json').read_text())['columns']
        if feature == 'efp':
            columns = [name for name in columns if name in ('mj1', 'mdiff') or name.startswith('EFP')]

    banks = {}
    for source in SOURCES:
        folder = feature_dir / source
        if feature in ('baseline', 'subjettiness'):
            frame = pd.read_hdf(folder / f'{feature}.h5')
            banks[source] = frame[columns].to_numpy(dtype=np.float32)
        else:
            metadata = json.loads((folder / 'combined.json').read_text())
            matrix = np.load(folder / 'combined.npy', mmap_mode='r')
            indices = [metadata['columns'].index(name) for name in columns]
            banks[source] = matrix[:, indices]
    return columns, banks


def assemble(events, banks):
    """Join feature rows in the exact order of the selected event identities."""
    matrix = np.empty((len(events), next(iter(banks.values())).shape[1]), dtype=np.float32)
    for code, source in enumerate(SOURCES):
        selected = events['source'] == code
        if selected.any():
            matrix[selected] = banks[source][events['row'][selected]]
    return matrix


def train_model(folder, feature, signal, n_signal, columns, banks, events):
    folder.mkdir(parents=True)
    pool = events['pool']
    matrix = assemble(pool, banks)
    target = pool['weak_label']
    positive_weight = len(target) / target.sum() - 1
    weight = np.where(target == 1, positive_weight, 1.)

    # Match the existing KS run: sklearn splits the pool internally, not a second
    # split of events['train']. Truth labels and mJJ are not classifier inputs.
    model = HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=31, validation_fraction=.5, random_state=SEED,
    )
    start = time.monotonic()
    model.fit(matrix, target, sample_weight=weight)
    training_seconds = time.monotonic() - start
    joblib.dump(model, folder / 'classifier.joblib')
    np.savez_compressed(folder / 'loss.npz',
                        training=-model.train_score_, validation=-model.validation_score_)
    config = dict(
        mode='CWoLa', seed=SEED, signal='background_only' if n_signal == 0 else signal,
        n_signal=n_signal, feature_set=feature, columns=columns,
        classifier=model.get_params(), sklearn=sklearn.__version__,
        boosting_iterations=model.n_iter_, training_seconds=training_seconds,
        train_events=len(events['train']), validation_events=len(events['validation']),
        positive_class_weight=float(positive_weight),
        target='SR=1, SB=0; truth is never supplied to fit.',
        validation='Internal stratified 50/50 split; early stopping on weighted weak-label log loss.',
        score='predict_proba(X)[:,1]',
        differences_from_paper='Single model at seed42; no ensembles or repeated runs.',
    )
    (folder / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(feature, signal, 'Nsig', n_signal, 'trained', model.n_iter_, 'rounds', flush=True)
    return model, config


def evaluate(folder, checkpoint, events_file, model, config, banks, events):
    folder.mkdir(parents=True, exist_ok=True)
    test = events['test']
    scores = model.predict_proba(assemble(test, banks))[:, 1]
    # Reuse the saved metadata instead of duplicating it for every feature set.
    with (folder / 'scores.npz').open('xb') as file:
        np.savez_compressed(file, score=scores)
    result = score_metrics(scores, test['truth'])
    result.update(
        signal=events_file.parent.parent.name, nsig_injected=config['n_signal'],
        feature_set=config['feature_set'], seed=SEED,
        boosting_iterations=config['boosting_iterations'], training_seconds=config['training_seconds'],
        checkpoint=str(checkpoint.resolve()), events=str(events_file.resolve()),
        source_names=SOURCES, mass_unit='TeV', score_order='Exactly events["test"] order',
    )
    with (folder / 'metrics.json').open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(config['feature_set'], result['signal'], 'Nsig', config['n_signal'],
          f'AUC {result["auc"]:.5f} maxSIC(KS) {result["max_sic_ks"]:.3f}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-sets', nargs='+', choices=FEATURE_SETS, default=list(FEATURE_SETS))
    parser.add_argument('--signals', nargs='+', choices=SIGNALS, default=list(SIGNALS))
    parser.add_argument('--n-signal', nargs='+', type=int, choices=range(0, 1001, 50), default=[1000])
    parser.add_argument('--feature-dir', type=Path, default=Path('data/features/kitchensink'))
    parser.add_argument('--split-dir', type=Path,
                        default=Path('runs/kitchensink_cwola_nsig_scan/seed42/data'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/kitchensink_cwola/seed42'))
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()

    with threadpool_limits(limits=args.threads):
        for feature in args.feature_sets:
            columns, banks = load_features(args.feature_dir, feature)
            zero_model = None
            for n_signal in args.n_signal:
                for signal in args.signals:
                    events_file = args.split_dir / signal / f'nsig{n_signal:04d}/events.npz'
                    with np.load(events_file) as loaded:
                        events = dict(loaded)
                    model_name = 'background_only' if n_signal == 0 else signal
                    model_dir = args.output_dir / feature / model_name / f'nsig{n_signal:04d}'
                    result_dir = args.output_dir / feature / signal / f'nsig{n_signal:04d}'

                    # At Nsig=0 every signal uses the same background-only pool.
                    if n_signal == 0 and zero_model is not None:
                        model, config = zero_model, zero_config
                    else:
                        model, config = train_model(model_dir, feature, signal, n_signal,
                                                    columns, banks, events)
                        if n_signal == 0:
                            zero_model, zero_config = model, config
                    evaluate(result_dir, model_dir / 'classifier.joblib', events_file,
                             model, config, banks, events)
                    del events
            del banks, model, zero_model
            gc.collect()


if __name__ == '__main__':
    main()
