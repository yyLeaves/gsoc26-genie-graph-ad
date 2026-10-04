"""NetGe variants: original, jet and relative-pT.

Original formulas are retained for comparison (see GLADC_LICENSE). Jet variants
keep Unique-6 topology and reconstruct physical features. build_netge selects
the implementation without wrapping modules or changing checkpoint keys.
"""

import torch
from torch import nn
from torch.func import functional_call
from torch_geometric.utils import scatter, to_dense_adj, to_dense_batch

from .edgeconv import _DecoderBlock
from ..losses import contrastive_loss


NETGE_VARIANTS = ('original', 'jet', 'fraction')


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gc1 = nn.Linear(1, 256, bias=False)
        self.gc4 = nn.Linear(256, 128, bias=False)
        self.proj_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 128))
        self.leaky_relu = nn.LeakyReLU(0.5)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, adjacency):
        hidden = self.dropout(self.leaky_relu(self.gc1(adjacency @ x)))
        z = self.gc4(adjacency @ hidden)
        return z, self.proj_head(z.max(dim=1).values)


class _AttributeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gc1 = nn.Linear(128, 256, bias=False)
        self.gc4 = nn.Linear(256, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.5)
        self.dropout = nn.Dropout(0.1)

    def forward(self, z, adjacency):
        hidden = self.dropout(self.leaky_relu(self.gc1(adjacency @ z)))
        return self.gc4(adjacency @ hidden)


class _StructureDecoder(nn.Module):
    def forward(self, z):
        return torch.sigmoid(z @ z.transpose(1, 2))


class NetGe(nn.Module):
    """Sum-aggregation NetGe: ln(pT) nodes and binary adjacency.

    The archived formulas use dense 30-node padding, including padded positions
    in global max pooling, structure reconstruction and latent-cycle losses.
    The encoder is shared by the input and reconstructed graphs.
    """

    def __init__(self):
        super().__init__()
        self.variant = 'original'
        self.shared_encoder = _Encoder()
        self.attr_decoder = _AttributeDecoder()
        self.struct_decoder = _StructureDecoder()

    @torch.no_grad()
    def _perturbed_graph(self, inputs):
        """Use noisy encoder weights without changing the model or its projection head."""
        parameters = {}
        for name, parameter in self.shared_encoder.named_parameters():
            value = parameter.detach()
            if name.startswith('proj_head.'):
                parameters[name] = value
            else:
                noise = torch.normal(0, torch.ones_like(value) * value.std())
                parameters[name] = value + noise
        _, graph = functional_call(self.shared_encoder, parameters, inputs)
        return graph

    def reconstruct(self, batch, *, contrast=False):
        """Reconstruct PyG jet graphs and re-encode them with the same encoder.

        B is the number of jets; N=30 includes padding. Output fields:
          inputs: (x, adjacency, None, mask), shaped (B,N,1), (B,N,N),
                  no edge features, and (B,N) respectively.
          node / adjacency: reconstructions shaped (B,N,1) / (B,N,N).
          latent / reencoded: original / reconstructed node vectors (B,N,128).
          graph / reencoded_graph: corresponding graph vectors (B,128).
          noisy_graph: perturbed-encoder graph vectors (B,128), only with contrast=True.
        """
        x, mask = to_dense_batch(batch.x, batch.batch, max_num_nodes=30)
        adjacency = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=30).gt(0).to(x)
        inputs = (x, adjacency)
        z, graph = self.shared_encoder(*inputs)
        result = dict(inputs=(x, adjacency, None, mask), latent=z, graph=graph)
        # Match the old RNG order: clean encode, perturb + noisy encode, decode + cycle.
        if contrast:
            result['noisy_graph'] = self._perturbed_graph(inputs)
        xhat = self.attr_decoder(z, adjacency)
        ahat = self.struct_decoder(z)
        zhat, ghat = self.shared_encoder(xhat, ahat)
        result.update(node=xhat, adjacency=ahat, reencoded=zhat, reencoded_graph=ghat)
        return result


def pair_mask(mask):
    diagonal = torch.eye(mask.shape[1], device=mask.device, dtype=torch.bool)
    return mask[:, :, None] & mask[:, None, :] & ~diagonal


class _JetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(1, 32), nn.GELU())
        self.self1 = nn.Linear(32, 64)
        self.message1 = nn.Sequential(nn.Linear(67, 64), nn.GELU(), nn.Linear(64, 64))
        self.self2 = nn.Linear(64, latent_dim)
        self.message2 = nn.Sequential(nn.Linear(131, 64), nn.GELU(), nn.Linear(64, latent_dim))
        self.proj_head = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 128))

    def forward(self, x, edge_index, edge_attr, mask):
        root, neighbor = edge_index
        nodes = self.embedding(x.flatten(0, 1))
        messages = self.message1(torch.cat([nodes[root], nodes[neighbor]-nodes[root], edge_attr], -1))
        hidden = torch.nn.functional.gelu(
            self.self1(nodes) + scatter(messages, root, dim=0, dim_size=len(nodes), reduce='mean')
        )
        messages = self.message2(torch.cat([hidden[root], hidden[neighbor]-hidden[root], edge_attr], -1))
        z = self.self2(hidden) + scatter(messages, root, dim=0, dim_size=len(nodes), reduce='mean')
        z = z.reshape(*mask.shape, -1) * mask[..., None]
        graph = self.proj_head(z.masked_fill(~mask[..., None], -torch.inf).max(1).values)
        return z, graph


class NetGeJet(nn.Module):
    """Inputs: x (B,N,1), A (B,N,N), mask (B,N), physical edges (B,N,N,3).

    Outputs follow NetGe's named fields, except edge is (E,3) on present edges.
    Both views use the same encoder and same sparse edge_index.
    """

    fixed_topology = True
    variant = 'jet'

    def __init__(self, latent_dim=2):
        super().__init__()
        self.shared_encoder = _JetEncoder(latent_dim)
        self.node_decoder = nn.ModuleList([
            _DecoderBlock(latent_dim, 32, nn.ReLU()),
            _DecoderBlock(32, 1, nn.Identity()),
        ])
        self.edge_decoder = nn.Sequential(nn.Linear(2*latent_dim, 64), nn.ReLU(),
                                          nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
        self.edge_direct = nn.Linear(2*latent_dim, 3)

    def reconstruct(self, batch, *, contrast=False):
        """Convert a PyG jet batch to the dense inputs used by all jet variants."""
        x, mask = to_dense_batch(batch.x, batch.batch, max_num_nodes=30)
        adjacency = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=30).gt(0).to(x)
        edges = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr, max_num_nodes=30)
        return self(x, adjacency, mask, edges, contrast=contrast)

    def forward(self, x, adjacency, mask, edges, *, contrast=False):
        graph, root, neighbor = adjacency.nonzero(as_tuple=True)
        edge_index = torch.stack([graph * x.shape[1] + root, graph * x.shape[1] + neighbor])
        edge_attr = edges[graph, root, neighbor]
        inputs = (x, edge_index, edge_attr, mask)
        z, g = self.shared_encoder(*inputs)
        output = dict(inputs=(x, adjacency, edges, mask), latent=z, graph=g,
                      pair_mask=pair_mask(mask), adjacency=adjacency, edge_graph=graph,
                      edge_target=edge_attr)
        if contrast:
            with torch.no_grad():
                parameters = {}
                for name, parameter in self.shared_encoder.named_parameters():
                    value = parameter.detach()
                    parameters[name] = (value if name.startswith('proj_head.')
                                        else value + torch.randn_like(value) * value.std())
                _, output['noisy_graph'] = functional_call(self.shared_encoder, parameters, inputs)
        flat_z = z.flatten(0, 1)
        endpoints = torch.cat([torch.minimum(flat_z[edge_index[0]], flat_z[edge_index[1]]),
                               torch.maximum(flat_z[edge_index[0]], flat_z[edge_index[1]])], -1)
        ehat = self.edge_decoder(endpoints) + self.edge_direct(endpoints)
        xhat = flat_z
        for block in self.node_decoder:
            xhat = block(xhat, edge_index)
        xhat = xhat.reshape_as(x) * mask[..., None]
        zhat, ghat = self.shared_encoder(xhat, edge_index, ehat, mask)
        output.update(node=xhat, edge=ehat, reencoded=zhat, reencoded_graph=ghat)
        return output

    def errors(self, output):
        x, _, _, mask = output['inputs']
        edge_error = (output['edge']-output['edge_target']).square().mean(-1)
        return dict(
            attribute=((output['node']-x).square().mean(-1)*mask).sum(1)/mask.sum(1),
            edge=scatter(edge_error, output['edge_graph'], dim=0, dim_size=len(x), reduce='mean'),
            node_cycle=((output['latent']-output['reencoded']).square().mean(-1)*mask).sum(1)/mask.sum(1),
            graph_cycle=(output['graph']-output['reencoded_graph']).square().mean(-1),
        )

    def objective(self, output, *, cycle_weight=1., contrast_weight=1.):
        losses = {name: value.mean() for name, value in self.errors(output).items()}
        objective = losses['attribute'] + losses['edge']
        objective = objective + cycle_weight * (losses['node_cycle'] + losses['graph_cycle'])
        if contrast_weight:
            losses['contrast'] = contrastive_loss(output['noisy_graph'], output['graph'])
            objective = objective + contrast_weight * losses['contrast']
        losses['objective'] = objective
        return losses

    def scores(self, output):
        scores = self.errors(output)
        scores['node'] = scores['attribute']
        scores['cycle'] = scores['node_cycle'] + scores['graph_cycle']
        scores['reconstruction'] = scores['node'] + scores['edge']
        scores['reconstruction_cycle'] = scores['reconstruction'] + scores['cycle']
        return scores


class _StandardizedJet(NetGeJet):
    """Standardized reconstruction shared by fraction and decoder extensions.

    Same dense inputs as NetGeJet. errors contains per-jet node/edge MSE and
    reconstruction (their sum). latent/graph expose node/max-pooled codes;
    edge_residual, edge_graph and edge_unique support score diagnostics.
    linear_edges omits the edge MLP for decoder extensions.
    """

    def __init__(self, *, latent_dim=2, linear_edges=False):
        super().__init__(latent_dim=latent_dim)
        self.shared_encoder.proj_head = nn.Identity()
        if linear_edges:
            self.edge_decoder = None
        self.register_buffer('node_center', torch.zeros(1))
        self.register_buffer('node_scale', torch.ones(1))
        self.register_buffer('edge_center', torch.zeros(3))
        self.register_buffer('edge_scale', torch.ones(3))

    def forward(self, x, adjacency, mask, edges, *, contrast=False):
        if contrast:
            raise ValueError('This reconstruction model has no contrastive branch')
        graph, source, target = adjacency.nonzero(as_tuple=True)
        edge_index = torch.stack([graph * x.shape[1] + source, graph * x.shape[1] + target])
        node_target = (x - self.node_center) / self.node_scale * mask[..., None]
        edge_target = (edges[graph, source, target] - self.edge_center) / self.edge_scale
        nodes, attributes = node_target, edge_target
        latent, graph_code = self.shared_encoder(nodes, edge_index, attributes, mask)
        flat = latent.flatten(0, 1)
        left, right = flat[edge_index[0]], flat[edge_index[1]]
        endpoints = torch.cat([torch.minimum(left, right), torch.maximum(left, right)], -1)
        predicted_edges = self.edge_direct(endpoints)
        if self.edge_decoder is not None:
            predicted_edges = self.edge_decoder(endpoints) + predicted_edges
        nodes = flat
        for block in self.node_decoder:
            nodes = block(nodes, edge_index)
        nodes = nodes.reshape_as(x)
        node_error = ((nodes - node_target).square()[..., 0] * mask).sum(1) / mask.sum(1)
        edge_residual = predicted_edges - edge_target
        edge_error = scatter(edge_residual.square().mean(-1), graph,
                             dim=0, dim_size=len(x), reduce='mean')
        return dict(errors=dict(node=node_error, edge=edge_error, reconstruction=node_error + edge_error),
                    edge_residual=edge_residual, edge_graph=graph, edge_unique=source < target,
                    latent=latent, graph=graph_code)

    def scores(self, output):
        return output['errors']

    def errors(self, output):
        return output['errors']

    def objective(self, output, *, cycle_weight=0., contrast_weight=0.):
        if cycle_weight or contrast_weight:
            raise ValueError('This model uses reconstruction only')
        losses = {key: value.mean() for key, value in output['errors'].items()}
        return dict(losses, objective=losses['reconstruction'])


def log_fractions(x, mask):
    """Convert (...,N,1) log-pT to log fractions over real nodes only."""
    log_total = torch.logsumexp(x.masked_fill(~mask[..., None], -torch.inf), dim=-2, keepdim=True)
    return (x - log_total).masked_fill(~mask[..., None], 0)


class NetGeFraction(_StandardizedJet):
    """Relative-pT nodes with training-standardized node/edge reconstruction.

    Node center/scale must be fitted on log fractions, not on raw ln(pT).
    Each graph's reconstruction score is invariant to a common pT multiplier.
    """

    variant = 'fraction'

    def forward(self, x, adjacency, mask, edges, *, contrast=False):
        return super().forward(log_fractions(x, mask), adjacency, mask, edges, contrast=contrast)


def build_netge(variant='original'):
    """Select a NetGe variant; all versions retain their own parameter names."""
    if variant == 'original':
        return NetGe()
    implementations = {'jet': NetGeJet, 'fraction': NetGeFraction}
    return implementations[variant]()
