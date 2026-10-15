from .edgeconv import EdgeConvAE
from .gladc import GLADC
from .netge import NetGe, NetGeJet, NetGeMaxBilinear, build_netge
from .sdm import SDMATII
from .sdm_energy import SDMEnergy

__all__ = ['EdgeConvAE', 'GLADC', 'NetGe', 'NetGeJet', 'NetGeMaxBilinear', 'SDMATII', 'SDMEnergy', 'create_model']


def create_model(settings):
    name = settings['model']
    if name == 'edgeconv':
        return EdgeConvAE()
    if name == 'gladc':
        return GLADC(settings.get('gladc_variant', 'original'))
    if name == 'netge':
        return build_netge(settings.get('netge_variant', 'original'),
                           objective=settings.get('objective', 'reconstruction'))
    if name == 'sdm_atii':
        return SDMATII(discrepancy_weight=settings.get('sdm_discrepancy_weight', 1.),
                       kl_weight=settings.get('sdm_kl_weight', 1.),
                       critic_clip=settings.get('sdm_critic_clip', .01))
    if name == 'sdm_energy':
        return SDMEnergy(settings.get('sdm_backbone', 'netge'),
                         discrepancy_weight=settings.get('sdm_discrepancy_weight', 1.),
                         kl_weight=settings.get('sdm_kl_weight', .01),
                         margin=settings.get('sdm_margin', .2),
                         negative_weight=settings.get('sdm_negative_weight', .1),
                         critic_context=settings.get('sdm_critic_context', False),
                         max_subjets=settings.get('max_subjets', 30))
    raise ValueError(f'Unknown model: {name}')
