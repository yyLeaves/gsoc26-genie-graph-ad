import awkward as ak
import numpy as np

__all__ = ["Jet", "p4_components", "to_pseudojets", "sum_p4", "dijet_mass",
           "relative_coords"]

Jet = tuple[np.ndarray, np.ndarray, np.ndarray]  # (pT, η, φ)


def p4_components(pt, eta, phi):
    """(px, py, pz, E) for massless constituents from (pT, η, φ)."""
    return (pt * np.cos(phi), pt * np.sin(phi),
            pt * np.sinh(eta), pt * np.cosh(eta))


def to_pseudojets(pt, eta, phi) -> ak.Array:
    """Convert (pT, η, φ) arrays to an awkward 4-vector array for fastjet."""
    px, py, pz, E = p4_components(pt, eta, phi)
    return ak.Array({"px": px, "py": py, "pz": pz, "E": E})


def sum_p4(jet: Jet) -> tuple[float, float, float, float]:
    """Sum 4-momentum of a jet's massless constituents."""
    pt, eta, phi = jet
    return ((pt * np.cos(phi)).sum(), (pt * np.sin(phi)).sum(),
            (pt * np.sinh(eta)).sum(), (pt * np.cosh(eta)).sum())


def dijet_mass(jet1: Jet, jet2: Jet) -> float:
    """Invariant mass of two jets from their constituent arrays."""
    px1, py1, pz1, E1 = sum_p4(jet1)
    px2, py2, pz2, E2 = sum_p4(jet2)
    return float(
        np.sqrt(
            max((E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 -
                (pz1 + pz2)**2, 0)))


def relative_coords(pt, eta, phi) -> tuple[np.ndarray, np.ndarray]:
    """(Δη, Δφ) of each particle vs the pT-weighted jet axis.

    φ averaged circularly (arctan2 of summed unit vectors); naive mean breaks
    at the ±π boundary.
    """
    eta0 = np.average(eta, weights=pt)
    phi0 = np.arctan2((pt * np.sin(phi)).sum(), (pt * np.cos(phi)).sum())
    return eta - eta0, (phi - phi0 + np.pi) % (2 * np.pi) - np.pi
