"""Event-wise gated scoring for strong-attraction and baseline models."""

import numpy as np
import torch

DEFAULT_GATE_TAU = 0.5
DEFAULT_GATE_TEMPERATURE = 0.25


def _sum_by_jet(values, batch_index, num_jets):
    return values.new_zeros((num_jets,) + values.shape[1:]).index_add_(
        0, batch_index, values)


def jet_structure_features(batch_index, num_jets, pos, pt):
    """Return normalized participation ratio and pT-weighted RMS radius."""
    pt = pt.reshape(-1).detach().clamp_min(0.0)
    count = _sum_by_jet(torch.ones_like(pt), batch_index, num_jets)
    fraction = pt / _sum_by_jet(pt, batch_index, num_jets).clamp_min(1e-12)[
        batch_index]
    sum_fraction_squared = _sum_by_jet(
        fraction.square(), batch_index, num_jets).clamp_min(1e-12)
    participation_ratio = 1.0 / (count.clamp_min(1.0) * sum_fraction_squared)
    radius_squared = pos.detach().square().sum(dim=-1)
    rms_radius = _sum_by_jet(
        fraction * radius_squared, batch_index, num_jets).clamp_min(0.0).sqrt()
    return participation_ratio, rms_radius


def fit_structure_calibration(pr, radius):
    """Fit median/IQR gate calibration on the selected calibration sample."""
    calibration = {}
    for name, values in (("pr", pr), ("radius", radius)):
        q25, center, q75 = np.quantile(
            np.asarray(values, dtype=np.float64), [0.25, 0.5, 0.75])
        if q75 <= q25:
            raise ValueError(f"{name} calibration IQR must be positive")
        calibration[f"{name}_center"] = float(center)
        calibration[f"{name}_scale"] = float(q75 - q25)
    return calibration


def structure_gate(
    pr, radius, *, pr_center, pr_scale, radius_center, radius_scale,
    tau=DEFAULT_GATE_TAU, temperature=DEFAULT_GATE_TEMPERATURE,
):
    """Map calibrated jet-structure coordinates to strong-model weights."""
    chi = 0.5 * (
        (pr - torch.as_tensor(pr_center, device=pr.device))
        / torch.as_tensor(pr_scale, device=pr.device)
        + (radius - torch.as_tensor(radius_center, device=pr.device))
        / torch.as_tensor(radius_scale, device=pr.device)
    )
    return torch.sigmoid(
        (torch.as_tensor(tau, device=pr.device) - chi)
        / torch.as_tensor(temperature, device=pr.device)
    ).detach()


def compactness_gate(batch_index, num_jets, pos, pt, **parameters):
    """Return the strong-attraction weight for each jet."""
    pr, radius = jet_structure_features(batch_index, num_jets, pos, pt)
    return structure_gate(pr, radius, **parameters)


def _percentile(reference, scores):
    reference = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    if reference.size == 0:
        raise ValueError("calibration reference must be non-empty")
    return np.searchsorted(reference, scores, side="right") / reference.size


def gated_event_score(strong, baseline, gate, strong_reference,
                      baseline_reference):
    """Percentile-calibrate both event scores and apply the input-only gate."""
    strong = _percentile(strong_reference, strong)
    baseline = _percentile(baseline_reference, baseline)
    gate = np.asarray(gate, dtype=np.float64)
    if strong.shape != baseline.shape or strong.shape != gate.shape:
        raise ValueError("strong, baseline, and gate scores must align")
    return gate * strong + (1.0 - gate) * baseline
