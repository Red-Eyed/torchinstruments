# Measurement history

`Observation` is the typed boundary between sampled tensor reduction and storage. It contains
one scalar measurement with identity, timestamp, metadata, and an explicit absence reason.

`ParquetHistory` is the imperative shell: it buffers bounded rows, publishes completed Parquet
chunks atomically with Polars, and streams them into a readable snapshot on publication. It owns only its
provided destination. It does not know about training tasks or example problem definitions.

The summary module constructs Polars query plans: `aggregate_history` computes metric summaries;
`layer_results` pivots only the aggregated rows into metric names and groups them by tensor
and layer. JSON contains history distributions and adjacent window means, never sample records. `write_result` is the thin JSON IO adapter.
Callers provide the module catalog, source path, destination, and comparison window.

```python
from pathlib import Path
from torchinstruments.history.summary import aggregate_history

metrics = aggregate_history(Path("stats/history.parquet"), window=20)
# Filter this lazy query before collecting the relevant summaries.
```

Tests live in `tests/test_history.py`. Timestamps remain typed until serialization. Domain
absence is `Absent`; serialized nullable values carry `unavailable_reason` beside them.

Execution context adds per-module mode and autograd recording state to every observation and
aggregation identity. Gradients retain the original forward context. The reader rejects older
histories without these fields: an explicit migration needs externally known execution context.
