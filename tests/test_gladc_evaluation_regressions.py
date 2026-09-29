"""Exercise evaluation entry points without loading research datasets."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.eval.scoring import EventScores


@pytest.mark.parametrize("mode", [None, "pooled", "projected", "unit_projected"])
@pytest.mark.parametrize("score_mode", ["cycle", "cycle_global", "reconstruction"])
def test_labeled_eval_uses_and_records_checkpoint_cycle_mode(
        tmp_path, monkeypatch, mode, score_mode):
    from scripts import eval_labeled_dataset as script

    checkpoint = tmp_path / "last.pt"
    training = {} if mode is None else {"cycle_global_mode": mode}
    torch.save({"run_config": {"training": training}}, checkpoint)
    class Dataset(list):
        meta = {}

    dataset = Dataset([None] * 8)
    monkeypatch.setattr(script, "load_any", lambda *a: (object(), object()))
    monkeypatch.setattr(script, "JetDataset", lambda *a, **k: dataset)
    monkeypatch.setattr(script, "ensure_dataset_matches", lambda *a: None)
    expected_mode = mode or "pooled"

    def score(*args, **kwargs):
        assert kwargs["score_mode"] == score_mode
        assert kwargs.get("cycle_global_mode") == expected_mode
        return EventScores(np.array([0., 2., 1., 3.]),
                           np.array([0, 0, 1, 1]), np.arange(4))

    monkeypatch.setattr(script, "score_events", score)
    output = tmp_path / "eval"
    script.main(SimpleNamespace(
        cpu=True, output_dir=str(output), checkpoint=str(checkpoint),
        data_dir="graphs", cache_shards=1, split_manifest=None, split="all",
        event_score_agg="sum", anomaly_score=score_mode, batch_size=8))
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["cycle_global_mode"] == expected_mode


@pytest.fixture
def suite(tmp_path, monkeypatch):
    from scripts import eval_cycle_suite as script

    checkpoint = tmp_path / "last.pt"
    torch.save({"run_config": {"training": {"cycle_global_mode": "projected"}}},
               checkpoint)
    split = tmp_path / "split.npz"
    np.savez(split, val_bkg_events=[0, 1], test_bkg_events=[2, 3],
             test_sig_events=[4, 5])
    args = SimpleNamespace(
        cpu=True, checkpoint=str(checkpoint), output_dir=str(tmp_path / "eval"),
        cycle_global_mode=None, bkg_dir="background", split_manifest=str(split),
        cache_shards=1, batch_size=8, n_signal=2, signal=["signal=signal_graphs"],
        bb1_dir="bb1", only_bb1=False, skip_bb1=True)
    monkeypatch.setattr(script, "load_cycle_model", lambda *a: (object(), object()))
    monkeypatch.setattr(script, "JetDataset", lambda *a, **k: [None] * 4)
    monkeypatch.setattr(script, "ensure_dataset_matches", lambda *a: None)
    monkeypatch.setattr(script, "indices_for_event_ids",
                        lambda ds, label, ids: (np.arange(4), ids))
    monkeypatch.setattr(script, "event_indices",
                        lambda ds, label, n: (np.arange(2 * n), np.arange(n)))

    def score(model, ds, indices, device, batch_size, name, **kwargs):
        n = len(indices) // 2
        labels = (np.array([0, 1]) if name == "BB1" else
                  np.full(n, int(name not in {"calibration", "background"})))
        return {component: EventScores(np.linspace(0.2, 0.8, n), labels, np.arange(n))
                for component in script.COMPONENTS}

    monkeypatch.setattr(script, "score_components", score)
    return script, args


@pytest.mark.parametrize("selection,expected", [
    ("full", {"LHCO", "signal", "BB1"}),
    ("only_bb1", {"BB1"}),
    ("skip_bb1", {"LHCO", "signal"}),
])
def test_suite_summary_contains_only_this_invocations_results(suite, selection, expected):
    script, args = suite
    args.skip_bb1 = False
    script.main(args)
    output = script.Path(args.output_dir)
    original = {name: (output / name / "metrics.json").read_bytes()
                for name in {"LHCO", "signal", "BB1"} - expected}
    args.only_bb1 = selection == "only_bb1"
    args.skip_bb1 = selection == "skip_bb1"
    script.main(args)
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["datasets"]) == expected
    for name, contents in original.items():
        assert (output / name / "metrics.json").read_bytes() == contents
    assert "evaluation" not in summary
    assert not (output / "evaluation.json").exists()


def test_suite_does_not_carry_old_signal_settings_into_a_new_summary(suite):
    script, args = suite
    args.signal = ["A=graphs_a", "B=graphs_b"]
    script.main(args)
    output = script.Path(args.output_dir)
    old_b = (output / "B" / "metrics.json").read_bytes()
    args.signal = ["A=different_graphs_a"]
    args.n_signal = 1
    script.main(args)
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["datasets"]) == {"LHCO", "A"}
    assert summary["datasets"]["A"]["reconstruction"]["n_signal"] == 1
    assert (output / "B" / "metrics.json").read_bytes() == old_b


@pytest.mark.parametrize("change", ["checkpoint_path", "checkpoint_contents",
                                   "cycle_mode", "split_repacked", "background"])
def test_suite_can_evaluate_new_inputs_without_adopting_old_results(suite, change):
    script, args = suite
    script.main(args)
    output = script.Path(args.output_dir)
    old_lhco = (output / "LHCO" / "metrics.json").read_bytes()
    if change.startswith("checkpoint"):
        if change == "checkpoint_path":
            args.checkpoint = str(output.parent / "other.pt")
        torch.save({"run_config": {"training": {"cycle_global_mode": "projected"}},
                    "different_weights": torch.ones(1)}, args.checkpoint)
    elif change == "cycle_mode":
        args.cycle_global_mode = "pooled"
    elif change == "split_repacked":
        with np.load(args.split_manifest) as split:
            arrays = {key: split[key] for key in split.files}
        np.savez_compressed(args.split_manifest, **arrays)
    else:
        args.bkg_dir = "different_background"
    args.only_bb1, args.skip_bb1 = True, False
    script.main(args)
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["datasets"]) == {"BB1"}
    assert summary["checkpoint"] == args.checkpoint
    assert (output / "LHCO" / "metrics.json").read_bytes() == old_lhco


def test_suite_ignores_old_metrics_without_requiring_a_provenance_file(suite):
    script, args = suite
    output = script.Path(args.output_dir)
    (output / "LHCO").mkdir(parents=True)
    (output / "LHCO" / "metrics.json").write_text('old incomplete output')
    args.only_bb1, args.skip_bb1 = True, False
    script.main(args)
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["datasets"]) == {"BB1"}
    assert (output / "LHCO" / "metrics.json").read_text() == 'old incomplete output'
    assert not (output / "evaluation.json").exists()
