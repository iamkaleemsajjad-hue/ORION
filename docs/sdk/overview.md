# Python SDKs

`orion-100m` and `orion-300m` are separate installable packages. Their SDKs are
the canonical inference interfaces. SDK loading is local and hash-verified;
there is no implicit Hub lookup, telemetry, or pretrained-model fallback.

## ORION-100M

```python
from orion_100m import Orion100M

model = Orion100M.load(device="auto")
result = model.classify({"text": "...", "title": "..."})
batch = model.predict_batch(["...", "..."], batch_size=2)
metadata = model.extract({"text": "..."})
vectors = model.get_embeddings(["..."], dim=128)
info = model.model_info()
health = model.health()
```

Calls requiring predictions raise `ModelWeightsNotFound` if a trained local
release is absent. Labels, thresholds, calibration, and tokenizer version come
from the release metadata. `batch_size` bounds the public API contract; current
prediction batches are evaluated document by document.

## ORION-300M

```python
from orion_300m import Orion300M

model = Orion300M.load(device="auto")
answer = model.answer("Question", documents=[{"id": "lecture-1", "text": "..."}])
summary = model.summarize(documents=[{"id": "lecture-1", "text": "..."}])
cards = model.generate_flashcards("topic", documents=[{"id": "notes", "text": "..."}])
text = model.generate("Continue this prompt")
```

Grounded task methods return `status="no_context"` and make no model call when
context is absent. Free continuation is available as `generate`. Citation marker
IDs are mapped only to supplied documents; the host remains responsible for
checking that cited text supports a claim. `stream_generate` decodes output
incrementally, and the REST stream forwards those SDK events as server-sent
events. Clients must apply `replace` events as full-text replacements and use
the final `done` event for status and citation metadata.
`generate_batch` accepts up to 32 single-request objects and preserves order;
each prompt is evaluated independently, with no throughput claim.

## Artifact directories

ORION-100M reads local `model.safetensors`, its `.sha256` sidecar, release
`config.json`, and tokenizer files in the release root or its `tokenizer/`
subdirectory. ORION-300M reads `artifact.json`, `model.safetensors`, the matching
sidecar, `config.json`, and verified tokenizer files. Both reject artifacts not
marked as trained and accepted/preview. See `tools/verify_release.py` before
installing a future artifact.
