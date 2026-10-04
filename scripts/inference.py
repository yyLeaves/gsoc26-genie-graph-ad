"""Score a split with a model checkpoint or a frozen calibrated model pair."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.calibration import apply_calibration
from src.checkpoint import load_checkpoint
from src.dataset import EventDataset
from src.models import create_model
from src.metrics import score_metrics
from src.scoring import score_loader, score_pair_loader, scoring_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--calibration', type=Path, help='frozen model pair from scripts.calibrate')
    parser.add_argument('--split-file', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    parser.add_argument('--graph-dir', type=Path, default=Path('data/graphs'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128, help='events per batch')
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0, cuda:1, ...')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--score', help='score to store as total; defaults to the checkpoint setting')
    args = parser.parse_args()

    dataset = EventDataset(args.split_file, split=args.split, graph_dir=args.graph_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    variant_scores = ()
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        settings = checkpoint['settings']
        model_name = settings.get('model', 'edgeconv')
        default_score, available = scoring_options(model_name, variant=settings.get('netge_variant', 'original'))
        score = args.score or settings.get('score', default_score)
        if score not in available:
            parser.error(f'--score for {model_name} must be one of: {", ".join(available)}')
        model = create_model(settings)
        model.load_state_dict(checkpoint['model'])
        model.to(args.device)
        edge_weight = checkpoint['settings']['edge_weight']
        epoch = checkpoint['epoch']
        print(f'Scoring {len(dataset):,} {args.split} events at epoch {epoch}', flush=True)
        result = score_loader(model, loader, args.device, edge_weight=edge_weight, score=score)
        if model_name in ('gladc', 'netge', 'sdm_nat'):
            variant_scores = available
        provenance = dict(checkpoint=str(args.checkpoint.resolve()), edge_weight=edge_weight, epoch=epoch)
        if model_name == 'sdm_nat':
            provenance.update(model='sdm_nat', implementation='Paper-based ICML 2025 SDM-NAT; no author code',
                              sdm_layer_norm=settings.get('sdm_layer_norm', True),
                              input_features='ln(pT) nodes and binary adjacency; no physical edge features',
                              score_definition='Sum over two jets of sigmoid(-normal_logit)')
    else:
        if args.score not in (None, 'total'):
            parser.error('Calibrated gated scoring uses --score total')
        score = 'total'
        bundle = torch.load(args.calibration, map_location='cpu', weights_only=True)
        models = {}
        for name in ('baseline', 'regularized'):
            # Existing EdgeConv pairs predate per-model settings.
            model = create_model(bundle.get(name+'_settings', {}))
            model.load_state_dict(bundle[name+'_model'])
            models[name] = model.to(args.device)
        settings = bundle['settings']
        print(f'Gated scoring: {len(dataset):,} {args.split} events', flush=True)
        result = score_pair_loader(models['baseline'], models['regularized'], loader, args.device,
                                   baseline_edge_weight=settings['baseline_edge_weight'],
                                   reg_edge_weight=settings['reg_edge_weight'],
                                   baseline_score=settings.get('baseline_score'),
                                   reg_score=settings.get('reg_score'))
        result.update(apply_calibration(result, bundle['calibration']))
        provenance = dict(calibration_file=str(args.calibration.resolve()))
        for name in ('baseline_epoch', 'reg_epoch'):
            provenance[name] = settings[name]
    result.update(provenance)
    metrics = score_metrics(result['total'], result['truth'])
    score_variants = {name: score_metrics(result[name], result['truth']) for name in variant_scores}
    with np.load(args.split_file) as split:
        result.update(source_files=split['source_files'], mass_unit=split['mass_unit'])
    result.update(split_file=str(args.split_file.resolve()), split=args.split, score=score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output.with_suffix('.json')
    if args.output.exists() or metrics_path.exists():
        raise FileExistsError(f'Score output already exists: {args.output} or {metrics_path}')
    with args.output.open('xb') as file:
        np.savez_compressed(file, **result)
    with metrics_path.open('x') as file:
        json.dump(dict(split=args.split, split_file=str(args.split_file.resolve()),
                       score=score, **provenance, metrics=metrics, score_variants=score_variants),
                  file, indent=2, allow_nan=False)
    print(f'Saved {len(result["row"]):,} event scores to {args.output}', flush=True)
    print(f'AUC={metrics["auc"]}  maxSIC={metrics["max_sic"]}  KS maxSIC={metrics["max_sic_ks"]}', flush=True)


if __name__ == '__main__':
    main()
