from .edgeconv import EdgeConvAE
from .gladc import GLADC
from .netge import NetGe, NetGeJet, build_netge
from .sdm import SDMNAT

__all__ = ['EdgeConvAE', 'GLADC', 'NetGe', 'NetGeJet', 'SDMNAT', 'create_model']


def create_model(settings):
    name = settings.get('model', 'edgeconv')
    if name in ('edgeconv', 'EdgeConvAE'):
        return EdgeConvAE()
    if name == 'gladc':
        return GLADC()
    if name == 'netge':
        return build_netge(settings.get('netge_variant', 'original'))
    if name == 'sdm_nat':
        return SDMNAT(discrepancy_weight=settings.get('sdm_discrepancy_weight', 1.),
                      kl_weight=settings.get('sdm_kl_weight', 1.),
                      layer_norm=settings.get('sdm_layer_norm', True))
    raise ValueError(f'Unknown model: {name}')
