import numpy as np
import pytest
import torch

from src.checkpoint import (load_checkpoint, require_training_state,
                            save_checkpoint)
from src.models import ModelSpec, create_model, load_model
from src.training.artifacts import (capture_training_state,
                                    restore_training_checkpoint)


def test_model_checkpoint_is_self_contained(tmp_path):
    spec = ModelSpec(type="node_graph", in_dim=5, use_bn=False)
    model = create_model(spec)
    path = tmp_path / "model.pt"

    save_checkpoint(
        path,
        model_spec=spec.to_dict(),
        model_state=model.state_dict(),
    )

    loaded = load_model(path, "cpu")
    assert type(loaded) is type(model)
    assert not (tmp_path / "config.json").exists()


def test_contrastive_projection_checkpoint_is_self_contained(tmp_path):
    spec = ModelSpec(
        type="edge_graph", in_dim=1, edge_dim=3, use_bn=False,
        contrastive_projection_dim=16,
        contrastive_source="eb2",
    )
    model = create_model(spec)
    path = tmp_path / "contrastive.pt"
    save_checkpoint(
        path,
        model_spec=spec.to_dict(),
        model_state=model.state_dict(),
    )

    loaded = load_model(path, "cpu")
    assert loaded.contrastive_projection_dim == 16
    assert loaded.contrastive_source == "eb2"
    assert all(
        torch.equal(loaded.state_dict()[name], value)
        for name, value in model.state_dict().items()
    )


def test_checkpoint_rejects_unversioned_state_dict(tmp_path):
    path = tmp_path / "raw_state.pt"
    torch.save({"weight": torch.ones(1)}, path)

    with pytest.raises(ValueError, match="unsupported format"):
        load_checkpoint(path)


def test_training_checkpoint_restores_optimizer_scheduler_and_rng(tmp_path):
    spec = ModelSpec(type="node_graph", in_dim=5, hidden_dim=8, latent_dim=2,
                     use_bn=False)
    model = create_model(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.5, total_iters=3
    )

    # Populate AdamW's moments so optimizer restoration is meaningful.
    optimizer.zero_grad()
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    scheduler.step()

    rng = np.random.default_rng(17)
    rng.random(4)
    signature = {"exact_context": True}
    history = [{"epoch": 1, "val_loss": 0.25}]
    state = capture_training_state(
        epoch=1,
        optimizer=optimizer,
        scheduler=scheduler,
        best_val=0.25,
        best_epoch=1,
        patience=0,
        history=history,
        monitor_best=None,
        rng=rng,
        elapsed_time_s=2.0,
    )
    saved_parameters = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    expected_numpy = rng.random()
    expected_torch = torch.rand(1)

    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model_spec=spec.to_dict(),
        model_state=model.state_dict(),
        run_config={"resume_signature": signature},
        training_state=state,
    )

    for parameter in model.parameters():
        parameter.data.zero_()
    rng.random(10)
    torch.rand(10)

    payload = load_checkpoint(path)
    restored = restore_training_checkpoint(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        rng=rng,
        expected_signature=signature,
    )

    assert require_training_state(payload)["epoch"] == 1
    assert restored["history"] == history
    assert scheduler.last_epoch == state["scheduler_state"]["last_epoch"]
    assert all(
        torch.equal(model.state_dict()[name], expected)
        for name, expected in saved_parameters.items()
    )
    assert rng.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)


def test_resume_rejects_a_different_context(tmp_path):
    spec = ModelSpec(type="node_graph", in_dim=5, use_bn=False)
    model = create_model(spec)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    rng = np.random.default_rng(0)
    state = capture_training_state(
        epoch=1,
        optimizer=optimizer,
        scheduler=scheduler,
        best_val=1.0,
        best_epoch=1,
        patience=0,
        history=[{"epoch": 1}],
        monitor_best=None,
        rng=rng,
        elapsed_time_s=1.0,
    )
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model_spec=spec.to_dict(),
        model_state=model.state_dict(),
        run_config={"resume_signature": {"dataset": "A"}},
        training_state=state,
    )

    with pytest.raises(ValueError, match="does not match"):
        restore_training_checkpoint(
            load_checkpoint(path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rng=rng,
            expected_signature={"dataset": "B"},
        )


def test_resume_does_not_infer_missing_training_settings():
    # The old backfill would accept an undocumented topo_norm=False and then
    # proceed to training-state loading. Reject before restoring any state.
    saved = {"training": {"epochs": 50}}
    expected = {"training": dict(
        epochs=50, topo_reg="none", lambda_topo=0.0, topo_norm=False,
        unique_k=6, cycle_weight=0.0, cycle_component="total",
        contrast_weight=0.0, perturb_scale=1.0, contrast_temperature=0.2,
        contrast_source="latent", relation_objective="none", relation_weight=0.0,
        relation_node_mask=0.15, relation_edge_mask=0.10,
        relation_temperature=0.2, relation_moment_weight=0.1,
        anomaly_score="reconstruction",
    )}
    with pytest.raises(ValueError, match="does not match"):
        restore_training_checkpoint(
            {"run_config": {"resume_signature": saved}},
            model=None, optimizer=None, scheduler=None, rng=None,
            expected_signature=expected,
        )
