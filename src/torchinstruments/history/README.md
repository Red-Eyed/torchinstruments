# Measurement history

`Observation` is the typed boundary between sampled tensor reduction and storage. It contains
one scalar measurement with identity, timestamp, metadata, and an explicit absence reason.

`ParquetHistory` is the imperative shell: it buffers bounded rows, publishes completed Parquet
chunks atomically with Polars, and streams them into a final file on close. It owns only its
provided destination. It does not know about training tasks or example problem definitions.

The summary module constructs Polars query plans: `aggregate_history` computes metric summaries;
`layer_results` groups those by tensor and layer. `write_result` is the thin JSON IO adapter.
Callers provide the module catalog, source path, destination, and comparison window.

```python
from pathlib import Path
from torchinstruments.history.summary import aggregate_history

metrics = aggregate_history(Path("stats/history.parquet"), window=20)
# Filter this lazy query before collecting the relevant summaries.
```

Tests live in `tests/test_history.py`. Timestamps remain typed until serialization. Domain
absence is `Absent`; serialized nullable values carry `unavailable_reason` beside them.
