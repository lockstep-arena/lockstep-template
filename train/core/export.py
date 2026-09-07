"""Export a trained policy to ONNX, and prove the export did not change it.

Two steps, and the second is the one that earns its keep. ``torch.onnx.export``
traces a graph; a traced graph that quietly differs from the module — a dropped
activation, a folded constant, a dtype change — still loads, still runs, and
still returns numbers of the right shape. Nothing downstream would notice. So
the export is immediately re-run under onnxruntime, the exact runtime the
platform's inference host uses, and compared against torch.

The signature is DERIVED from the net (obs Dict keys in, ``action`` out) —
see :mod:`train.core.policy`. This module is the stable seam: ANY policy
module exposing the same ``input_names`` / ``input_shapes`` / ``action_len``
surface can be exported, parity-checked and staged, whatever trained it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .policy import OUTPUT_ACTION, Policy, graph_dtype

#: Largest tolerated difference between torch and onnxruntime, per element.
#:
#: Not zero: ORT is free to fuse and reassociate float ops, so bit-equality is
#: not on offer. This is tight enough that a real graph difference (a missing
#: tanh, a transposed input) blows straight through it, which is what the check
#: is for.
PARITY_ATOL = 1e-4


def sample_inputs(net: Policy, batch: int = 1) -> tuple[torch.Tensor, ...]:
    """Representative inputs for tracing and for the parity check.

    Random rather than zeros on purpose: a zero observation would sail through
    a graph that had lost a bias or a nonlinearity. Keyed on the DECLARED
    dtype, never on shape: ``u8`` inputs get values in [0, 1] (the ÷ 255
    range the graph input carries), ``i32`` inputs small integers, ``f32``
    inputs standard normals.
    """
    generator = torch.Generator().manual_seed(0)
    dtypes = getattr(net, "input_dtypes", {})
    tensors = []
    for name in net.input_names:
        shape = (batch, *net.input_shapes[name])
        declared = dtypes.get(name, "float32")
        if declared == "int32":
            tensors.append(
                torch.randint(0, 4, shape, generator=generator, dtype=torch.int32)
            )
        elif declared == "uint8":
            tensors.append(torch.rand(*shape, generator=generator))
        else:
            tensors.append(torch.randn(*shape, generator=generator))
    return tuple(tensors)


def export(net: Policy, path: str | Path) -> Path:
    """Write ``path`` as ONNX with the net's derived signature."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net.eval()

    torch.onnx.export(
        net,
        sample_inputs(net),
        str(path),
        input_names=net.input_names,
        output_names=[OUTPUT_ACTION],
        # Fixed batch of 1. The shells send exactly one observation per tick,
        # and a dynamic axis would only add a degree of freedom nobody uses
        # and the host would have to validate.
        dynamic_axes=None,
        opset_version=17,
        # SELF-CONTAINED, and not the default. torch's exporter writes weights
        # to a `<name>.onnx.data` sidecar once they pass a size threshold,
        # leaving a ~4 KB graph behind. An agent bundle ships ONE artifact
        # path, so a sidecar is silently left out of it — and the parity check
        # below still passes, because it runs next to the sidecar. The result
        # is a bundle whose model loads and infers with uninitialized weights.
        external_data=False,
    )
    return path


def verify(net: Policy, path: str | Path, atol: float = PARITY_ATOL) -> float:
    """Run torch and onnxruntime on the same inputs; return the max abs diff.

    Raises if they disagree by more than ``atol``, or if the exported graph's
    declared signature is not the one the shells bind to by name.
    """
    import onnxruntime as ort

    path = Path(path)
    inputs = sample_inputs(net)
    net.eval()
    with torch.no_grad():
        expected = net(*inputs).numpy()

    # Belt and braces on the external-data trap above: assert it here too, so
    # a future exporter default cannot quietly reintroduce a sidecar. Loading
    # WITHOUT external data is the only way to see it — load it normally and
    # the sidecar is silently pulled in.
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    external = [
        init.name
        for init in graph.initializer
        if init.data_location == onnx.TensorProto.EXTERNAL
    ]
    if external:
        raise AssertionError(
            f"{len(external)} initializers live outside {path.name} "
            f"(e.g. {external[:3]}). An agent bundle ships one artifact file, "
            "so those weights would not be shipped and the model would infer "
            "from uninitialized memory."
        )

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    got_names = {i.name for i in session.get_inputs()}
    want_names = set(net.input_names)
    if got_names != want_names:
        raise AssertionError(
            f"exported input names {sorted(got_names)} != {sorted(want_names)}; "
            "the host binds tensors by NAME, so a rename silently breaks every shell"
        )
    # The declared signature, exactly: shape [1, *declared] and the dtype
    # the shell feeds (f32 for f32 and u8 ÷ 255, int32 for i32). A stream
    # that reshaped or re-typed its input would still trace and still run
    # here; only this check says it no longer matches the wire.
    dtypes = getattr(net, "input_dtypes", {})
    ort_types = {"float32": "tensor(float)", "int32": "tensor(int32)"}
    for inp in session.get_inputs():
        want_shape = [1, *net.input_shapes[inp.name]]
        if list(inp.shape) != want_shape:
            raise AssertionError(
                f"exported input {inp.name!r} has shape {list(inp.shape)}, the "
                f"declaration says {want_shape}; the shell feeds the declared "
                "shape verbatim and refuses anything else"
            )
        want_type = ort_types[graph_dtype(dtypes.get(inp.name, "float32"))]
        if inp.type != want_type:
            raise AssertionError(
                f"exported input {inp.name!r} is {inp.type}, the shell feeds "
                f"{want_type} for this declared dtype"
            )
    out_names = [o.name for o in session.get_outputs()]
    if OUTPUT_ACTION not in out_names:
        raise AssertionError(
            f"exported outputs {out_names} do not include {OUTPUT_ACTION!r}; "
            "the shells look this up by name and fall back to standing still"
        )

    (actual,) = session.run(
        [OUTPUT_ACTION],
        {name: t.numpy() for name, t in zip(net.input_names, inputs, strict=True)},
    )

    if actual.shape != expected.shape:
        raise AssertionError(f"onnx output {actual.shape} != torch {expected.shape}")
    if actual.shape[1] != net.action_len:
        raise AssertionError(
            f"exported action width {actual.shape[1]} != {net.action_len}; "
            "the shells refuse an action of the wrong width, so this bundle "
            "would stand still for the whole match"
        )

    diff = float(np.max(np.abs(actual - expected)))
    if diff > atol:
        raise AssertionError(
            f"torch and onnxruntime disagree by {diff:.3e} (> {atol:.1e}). "
            "The export changed the policy; the agent would not be the thing "
            "that was trained."
        )
    return diff
