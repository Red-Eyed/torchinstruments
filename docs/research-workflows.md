# Investigating model behavior

Use telemetry to locate a measured change, then test a proposed cause. There are no automatic
problem categories or scores in the artifacts. The definitions below belong to the research
workflow and runnable examples.

| Possible problem | Evidence to seek | Important alternative explanation |
| --- | --- | --- |
| Inactive ReLU block | Persistent all-zero outputs and zero gradients at the preceding boundary | Zero outputs can be expected for the sampled inputs |
| Sigmoid saturation | Outputs concentrated near the endpoints and weak gradients upstream | Small loss derivatives also produce small gradients |
| Excessive amplification | Increasing output or gradient scale across layers or time | A legitimate scale change or changing input distribution |
| Invalid arithmetic | Positive nonfinite fraction at an observed boundary | Invalid values may originate earlier in an unobserved operation |
| Disconnected branch | Forward observations continue but expected backward observations disappear | Inference, pending backward, or an intentionally unused branch |

The [examples](../examples/find_problems.py) create each mechanism deliberately, collect matched
baseline/broken/fixed histories, query the relevant measurements with Polars, and generate plots.
The measurements identify a signature; the controlled intervention establishes its cause in
that example. The same signature alone does not establish a cause in another model.

Compare the same model boundaries, input batches, precision, reduction settings, and sampling
policy. Time-based sampling may capture different batches in different runs. Prefer a fixed
forward cadence for controlled comparisons. A sample ID is not an optimizer-step counter.

Do not treat ordinary sparsity, scale, or variance as an error by itself. Check task loss and
held-out metrics separately. Tensor-wide statistics can hide a small number of failing channels.
For unexplained invalid backward operations, PyTorch's
[anomaly detection](https://docs.pytorch.org/docs/stable/autograd.html#anomaly-detection) provides
operation-level traceback support beyond these module-boundary measurements.
