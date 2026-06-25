import argparse
import time
from itertools import chain

from src.data import shards
from src.data.build_graph import build_graph_shards
from src.data.events import EventReader
from src.data.extractor import JetConfig, JetExtractor
from src.data.preprocess import make_metadata


def jets_in_range(reader, extractor, start, stop, chunk=20000):
    """Per-jet Data objects from events [start, stop)."""
    for s in range(start, stop, chunk):
        labels, data = reader.read(s, min(s + chunk, stop))
        for event_id, (label, jets) in enumerate(
                zip(labels, extractor.cluster_chunk(data)), start=s):
            yield from extractor.event_jets(jets, int(label), event_id)


def main(args):
    reader = EventReader(args.h5_path)
    extractor = JetExtractor(JetConfig(min_jet_pt=args.min_jet_pt,
                                       n_subjets=args.n_subjets))
    bkg_stop = min(args.bkg_events, reader.n_events)
    sig_stop = min(args.sig_start + args.sig_events, reader.n_events)
    sig_events = max(0, sig_stop - args.sig_start)
    n_events = bkg_stop + sig_events
    print(f"Input  : {args.h5_path}  ({reader.n_events:,} events)")
    print(f"Subset : {bkg_stop:,} bkg [0:] + {sig_events:,} sig "
          f"[{args.sig_start:,}:]   min_jet_pt={args.min_jet_pt} "
          f"n_subjets={args.n_subjets}")

    t0 = time.time()
    pc_dir = f"{args.output_dir}_jets"
    jets = chain(
        jets_in_range(reader, extractor, 0, bkg_stop),
        jets_in_range(reader, extractor, args.sig_start, sig_stop),
    )
    stats = shards.write_shards(jets, pc_dir, shard_size=8192)
    labels = stats["labels"]
    print(f"Jets   : {stats['n']:,}  (signal={labels.sum():,}  "
          f"bkg={(labels == 0).sum():,})  ({time.time()-t0:.0f}s)")

    shards.save_metadata(make_metadata(stats, extractor.cfg, n_events), pc_dir)
    build_graph_shards(pc_dir, args.output_dir, strategy=args.strategy,
                       k=args.k, edge_feats=args.edge_features,
                       normalize_edge_pt=args.normalize_edge_pt)
    print(f"Done   : {args.output_dir}/  ({time.time()-t0:.0f}s total)")


def parse_args():
    p = argparse.ArgumentParser(description="Build a bkg+signal LHCO graph subset")
    p.add_argument("--h5_path", required=True)
    p.add_argument("--output_dir", required=True, help="graph-shard dir to write")
    p.add_argument("--bkg_events", type=int, default=100000)
    p.add_argument("--sig_start", type=int, default=1000000,
                   help="first signal event index (LHCO: signal starts at 1e6)")
    p.add_argument("--sig_events", type=int, default=40000)
    p.add_argument("--min_jet_pt", type=float, default=1200.0)
    p.add_argument("--n_subjets", type=int, default=30)
    p.add_argument("--strategy", default="unique",
                   choices=["knn", "laman", "unique", "fully_connected"])
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--edge_features", default="log",
                   choices=["none", "linear", "log"])
    p.add_argument("--raw_edge_pt", action="store_false",
                   dest="normalize_edge_pt",
                   help="Compute edge k_T from raw input pt instead of "
                        "pt/Σpt_jet.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
