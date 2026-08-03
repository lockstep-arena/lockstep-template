"""Train → export → stage: the whole pipeline behind `task train`.

Game-agnostic, one file per concern: ``core/`` holds the machinery
(entry-point discovery, spaces-derived policy, PPO loop, export/parity,
staging) and ``main.py`` chains it for whatever game ``--game`` names. The
interesting contract lives at the edges — the derived ONNX signature
(core/policy.py / core/export.py), the checkpoint format (core/train.py)
and the bundle layout (core/stage.py). Everything in between is yours to
replace.
"""
