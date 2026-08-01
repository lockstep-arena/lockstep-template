"""Train → export → stage: the whole pipeline behind `task train`.

Adapted from the dance-off reference training pipeline; small on purpose.
The interesting contract lives at the edges — the ONNX signature
(policy.py / export.py) and the bundle layout (main.py). Everything in
between is yours to replace.
"""
