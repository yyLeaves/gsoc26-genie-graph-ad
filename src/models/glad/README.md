# src/models/glad

Graph-level anomaly detection baselines and teacher-student distillation models.

| File | Purpose |
| --- | --- |
| [graph_oneclass.py](graph_oneclass.py) | Implement Deep-SVDD-style `OneClassGIN` and mutual-information-based `InfoGraphOneClass` with background-centered scoring |
| [glocal_kd.py](glocal_kd.py) | Distill node and graph representations from a frozen random teacher and score teacher-student errors |
| [baseline_teacher_kd.py](baseline_teacher_kd.py) | Distill a background-trained Graph-AE teacher and compute EB2, EB3, and graph-level representation errors |
| [fgsd.py](fgsd.py) | Extract FGSD graph fingerprints and score them with Isolation Forest, LOF, or One-Class SVM |
