# src/objectives

Compute regularization, cycle-consistency, and contrastive objectives for graph autoencoders.

| File | Purpose |
| --- | --- |
| [topo_reg.py](topo_reg.py) | Regularize neighboring representations on input or latent-space graphs |
| [latent_cycle.py](latent_cycle.py) | Re-encode reconstructed graphs and measure node-level and graph-level latent discrepancies |
| [perturbed_contrast.py](perturbed_contrast.py) | Compute GLADC-style graph contrast between clean and weight-perturbed encoders |
| [edge_relation_contrast.py](edge_relation_contrast.py) | Mask edge features and contrast true edge relations against candidates within each graph |
| [edge_relation_vicreg.py](edge_relation_vicreg.py) | Align clean and corrupted edge relations with a VICReg loss |
| [_masking.py](_masking.py) | Sample graph-local random masks for nodes or edges |
