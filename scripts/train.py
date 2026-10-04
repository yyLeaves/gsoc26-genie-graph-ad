"""Train a jet model on SR/reference; save the last epoch and periodic metrics."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from src.checkpoint import save_checkpoint
from src.dataset import EventDataset
from src.metrics import score_metrics
from src.models import create_model
from src.models.netge import NETGE_VARIANTS
from src.scoring import score_loader, scoring_options
from src.training import fit_netge_scaling, train_epoch, validate


def _arguments():
    # Unspecified training options come from the checkpoint when resuming.
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument('--resume', type=Path, default=None, help='continue an interrupted last.pt run')
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--graph-dir', type=Path)
    parser.add_argument('--sample', choices=['SR', 'reference'],
                        help='reference = CWoLa sideband or IAD extra QCD')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--model', choices=['edgeconv', 'gladc', 'netge', 'sdm_nat'])
    parser.add_argument('--netge-variant', choices=NETGE_VARIANTS)
    parser.add_argument('--objective', choices=['reconstruction', 'cycle', 'contrast', 'full', 'nat'],
                        help='GLADC/NetGe-Jet ablation; original NetGe uses full, SDM-NAT uses nat')
    parser.add_argument('--score', help='primary anomaly score; default depends on model/variant')
    parser.add_argument('--epochs', type=int, help='total epochs, default 50')
    parser.add_argument('--batch-size', type=int, help='events per batch, two jets each; defaults follow the model recipe')
    parser.add_argument('--lr', type=float, help='fixed learning rate, or peak for OneCycleLR')
    parser.add_argument('--edge-weight', type=float)
    parser.add_argument('--reg-weight', type=float,
                        help='EB3 regularization weight; 0 = baseline, 1 = previous EB3 setting')
    parser.add_argument('--sdm-discrepancy-weight', type=float, help='SDM-NAT lambda, default 1')
    parser.add_argument('--sdm-kl-weight', type=float, help='SDM-NAT gamma, default 1')
    parser.add_argument('--sdm-layer-norm', action=argparse.BooleanOptionalAction,
                        help='per-node LayerNorm in SDM-NAT GIN layers, enabled by default')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--eval-interval', type=int, help='test AUC/SIC interval, default 5; also evaluate final epoch')
    parser.add_argument('--device', help='cpu or cuda:0, cuda:1, ...; default cpu for new runs')
    parser.add_argument('--num-workers', type=int)
    options = vars(parser.parse_args())
    checkpoint = None
    if options['resume']:
        checkpoint = torch.load(options['resume'], map_location='cpu', weights_only=True)
        required = {'model', 'settings', 'epoch', 'optimizer', 'scheduler', 'rng', 'history'}
        if not required.issubset(checkpoint):
            parser.error('This checkpoint lacks complete resume state; use it for inference, not exact resume')
        saved = checkpoint['settings'].copy()
        saved.setdefault('objective', 'reconstruction')
        saved.setdefault('score', 'total')
        if saved.get('model') == 'netge':
            saved.setdefault('netge_variant', 'original')
        if saved.get('model') == 'sdm_nat':
            saved.setdefault('sdm_layer_norm', True)
        for name, value in options.items():
            if name in ('resume', 'device', 'num_workers'):
                continue
            value = str(value.resolve()) if isinstance(value, Path) else value
            if value != saved.get(name):
                parser.error(f'--{name.replace("_", "-")} cannot change on resume (saved: {saved.get(name)})')
        values = {**saved, 'resume': options['resume'], 'output_dir': options['resume'].resolve().parent,
                  'device': options.get('device', saved['device']),
                  'num_workers': options.get('num_workers', saved['num_workers'])}
    else:
        for name in ('split_file', 'sample', 'output_dir'):
            if name not in options:
                parser.error(f'--{name.replace("_", "-")} is required for a new run')
        model = options.get('model', 'edgeconv')
        netge = model == 'netge'
        sdm = model == 'sdm_nat'
        variant = options.get('netge_variant', 'original')
        default_score, _ = scoring_options(model, variant=variant)
        values = dict(graph_dir=Path('data/graphs'), epochs=50,
                      batch_size=150 if netge else 256, lr=0.0001 if netge else 0.003,
                      edge_weight=1., reg_weight=0., seed=123, eval_interval=5, device='cpu', num_workers=0,
                      model=model, objective='reconstruction' if model == 'edgeconv' else 'full', score=default_score)
        if netge:
            values['netge_variant'] = variant
            if variant in ('jet', 'fraction'):
                values.update(objective='reconstruction', seed=42)
            if variant == 'fraction':
                values.update(batch_size=256, lr=0.003)
        if sdm:
            values.update(batch_size=128, lr=0.001, objective='nat',
                          sdm_discrepancy_weight=1., sdm_kl_weight=1., sdm_layer_norm=True)
        values.update(options)
    for name in ('split_file', 'graph_dir', 'output_dir'):
        values[name] = Path(values[name])
    args = argparse.Namespace(**values)
    if args.epochs < 1 or args.eval_interval < 1:
        parser.error('--epochs and --eval-interval must be at least 1')
    if args.reg_weight < 0:
        parser.error('--reg-weight must be nonnegative')
    baseline = args.model in ('edgeconv', 'EdgeConvAE')
    if args.reg_weight and not baseline:
        parser.error('--reg-weight applies only to EdgeConv EB3')
    if baseline and args.objective != 'reconstruction':
        parser.error('Use --model gladc for cycle/contrast objectives')
    if args.model == 'netge':
        allowed = dict(original=('full',),
                       jet=('reconstruction', 'cycle', 'contrast', 'full'), fraction=('reconstruction',))
        if args.objective not in allowed[args.netge_variant] or args.edge_weight != 1:
            parser.error(f'NetGe {args.netge_variant} uses unit weights and objectives: {allowed[args.netge_variant]}')
    elif hasattr(args, 'netge_variant'):
        parser.error('--netge-variant applies only to NetGe')
    if args.model == 'gladc' and args.objective == 'nat':
        parser.error('Use --model sdm_nat for the NAT objective')
    if args.model == 'sdm_nat':
        if args.objective != 'nat' or args.edge_weight != 1:
            parser.error('SDM-NAT uses --objective nat and does not use --edge-weight')
        weights = (args.sdm_discrepancy_weight, args.sdm_kl_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            parser.error('SDM-NAT loss weights must be finite and nonnegative')
    elif any(name.startswith('sdm_') for name in vars(args)):
        parser.error('--sdm-* options apply only to sdm_nat')
    _, available = scoring_options(args.model, variant=getattr(args, 'netge_variant', 'original'))
    if args.score not in available:
        parser.error(f'--score for {args.model} must be one of: {", ".join(available)}')
    return args, checkpoint


def _write_history(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args, checkpoint = _arguments()
    if checkpoint is not None and checkpoint['epoch'] >= args.epochs:
        for name, rows in checkpoint['history'].items():
            if rows:
                _write_history(args.output_dir / f'{name}.csv', rows)
        print(f'Run already completed epoch {checkpoint["epoch"]}', flush=True)
        return

    datasets = []
    weak_label = {'SR': 1, 'reference': 0}[args.sample]
    for part in ('train', 'validation'):
        dataset = EventDataset(args.split_file, split=part, graph_dir=args.graph_dir)
        indices = np.flatnonzero(dataset.events['weak_label'] == weak_label)
        if len(indices) == 0:
            raise ValueError(f'No {args.sample} events in {part}')
        datasets.append(Subset(dataset, indices))
    training_loader = DataLoader(
        datasets[0], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        datasets[1], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        EventDataset(args.split_file, split='test', graph_dir=args.graph_dir),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        # Monitoring must not consume the training RNG stream.
        generator=torch.Generator().manual_seed(args.seed),
    )

    torch.manual_seed(args.seed)
    model = create_model(vars(args)).to(args.device)
    variant = getattr(args, 'netge_variant', 'original')
    if checkpoint is None and args.model == 'netge' and variant == 'fraction':
        scaling_loader = DataLoader(datasets[0], batch_size=256, shuffle=False,
                                    num_workers=args.num_workers,
                                    generator=torch.Generator().manual_seed(args.seed))
        fit_netge_scaling(model, scaling_loader)
    plain_adam = (args.model == 'netge' and variant != 'fraction') or args.model == 'sdm_nat'
    total_steps = args.epochs * len(training_loader)
    if plain_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler, schedule_settings, grad_clip_norm = None, None, None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        # Match the baseline's warm-up rule, including short runs.
        pct_start = min(max(max(2, int(0.02 * total_steps)) / total_steps, 0.01), 0.9)
        schedule = dict(max_lr=args.lr, total_steps=total_steps, pct_start=pct_start,
                        anneal_strategy='linear', div_factor=5.0, final_div_factor=3.0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **schedule)
        schedule_settings, grad_clip_norm = dict(name='OneCycleLR', **schedule), 1.0
    history = {'losses': [], 'metrics': []}
    start_epoch = 1
    if checkpoint is None:
        settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                    for key, value in vars(args).items() if key != 'resume'}
        settings.update(optimizer='Adam' if plain_adam else 'AdamW', weight_decay=0. if plain_adam else 0.01,
                        grad_clip_norm=grad_clip_norm, scheduler=schedule_settings,
                        regularization='eb3' if args.reg_weight else 'none',
                        n_train=len(datasets[0]), n_validation=len(datasets[1]))
        args.output_dir.mkdir(parents=True)
        with (args.output_dir / 'settings.json').open('w') as file:
            json.dump(settings, file, indent=2)
    else:
        settings = checkpoint['settings']
        if scheduler is not None and total_steps != checkpoint['scheduler']['total_steps']:
            raise ValueError('Training loader length changed; cannot resume the saved OneCycleLR schedule')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        history = checkpoint['history']
        if args.model in ('edgeconv', 'EdgeConvAE') and not args.reg_weight:
            # Older baseline logs recorded the objective only as total.
            for row in history['losses']:
                row.setdefault('train_objective', row['train_total'])
                row.setdefault('validation_objective', row['validation_total'])
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng']['torch'])
        training_loader.generator.set_state(checkpoint['rng']['loader'])
        if checkpoint['rng']['cuda'] is not None and torch.device(args.device).type == 'cuda':
            torch.cuda.set_rng_state(checkpoint['rng']['cuda'], device=args.device)
        for name, rows in history.items():
            if rows:
                _write_history(args.output_dir / f'{name}.csv', rows)
        print(f'Resuming after epoch {checkpoint["epoch"]}', flush=True)

    print(f'{args.sample}: {len(datasets[0]):,} training / {len(datasets[1]):,} validation events', flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        training = train_epoch(model, training_loader, optimizer, args.device,
                               edge_weight=args.edge_weight, reg_weight=args.reg_weight,
                               scheduler=scheduler, objective=args.objective, grad_clip_norm=grad_clip_norm)
        validation = validate(model, validation_loader, args.device,
                              edge_weight=args.edge_weight, reg_weight=args.reg_weight, objective=args.objective)
        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
        row.update({f'train_{name}': value for name, value in training.items()})
        row.update({f'validation_{name}': value for name, value in validation.items()})
        history['losses'].append(row)
        message = (f'Epoch {epoch}/{args.epochs}: train_objective={training["objective"]:.6g} '
                   f'validation_objective={validation["objective"]:.6g}')
        if args.reg_weight:
            message += f' reg={training["reg"]:.6g}'
        evaluate_now = epoch % args.eval_interval == 0 or epoch == args.epochs
        if evaluate_now:
            scored = score_loader(model, test_loader, args.device, edge_weight=args.edge_weight, score=args.score)
            metrics = score_metrics(scored['total'], scored['truth'])
            if args.model in ('edgeconv', 'EdgeConvAE'):
                history['metrics'].append(dict(epoch=epoch, split='test', **metrics))
            else:
                for name in scoring_options(args.model, variant=variant)[1]:
                    values = metrics if name == 'total' else score_metrics(scored[name], scored['truth'])
                    history['metrics'].append(dict(epoch=epoch, split='test', score=name, **values))
            message += (f' AUC={metrics["auc"]} maxSIC={metrics["max_sic"]}'
                        f' KS maxSIC={metrics["max_sic_ks"]}')

        rng = dict(torch=torch.get_rng_state(), loader=training_loader.generator.get_state(),
                   cuda=torch.cuda.get_rng_state(args.device) if torch.device(args.device).type == 'cuda' else None)
        state = {
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'epoch': epoch, 'settings': settings,
            'rng': rng, 'history': history,
        }
        save_checkpoint(args.output_dir / 'last.pt', state)
        if evaluate_now and args.model in ('gladc', 'netge', 'sdm_nat'):
            save_checkpoint(args.output_dir / f'epoch{epoch:02d}.pt', state)
        for name, rows in history.items():
            if rows:
                _write_history(args.output_dir / f'{name}.csv', rows)
        print(message, flush=True)


if __name__ == '__main__':
    main()
