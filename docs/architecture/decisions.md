# Architecture decisions and open items

## Decisions preserved from the technical plan

| Decision | Implementation |
|---|---|
| Independent model family | Separate `orion_100m` and `orion_300m` packages and independently constructed model classes. Neither imports the other. |
| Shared tokenizer contract | `orion-tokenizer` owns the 32,768-ID v1.0 interface; both SDKs verify its version and hash before loading weights. No final tokenizer is present. |
| ORION-100M | 16-layer, width 640, 10-head bidirectional encoder; GeGLU 1,728; bias-free pre-LayerNorm; alternating 128-token local and global attention; RoPE. |
| ORION-100M objective | ELECTRA-style MLM generator and replaced-token discriminator; discriminator embedding gradients are detached from the shared table. Contrastive and task losses are implemented separately. |
| ORION-300M default | 28-layer, width 960, 15/5 GQA, SwiGLU 2,560, RoPE theta 100,000, RMSNorm, tied embeddings, causal decoder, static KV cache. |
| Safe artifacts | Safetensors weights with SHA-256 verification. Checkpoint recovery uses safetensors plus JSON rather than a pickle `training_state.pt`, resolving the plan's conflict between a `.pt` example and its no-unsafe-deserialization requirement. |
| Data integrity | Local-only pipeline stages with explicit provenance; dataset acquisition is a separate command that requires `--execute`. |
| MADAD isolation | Standalone model packages do not import MADAD. Adapter payloads are defined in a separate optional package; the adapter does not open files or databases. |

## Provisional special-token assignments

The supplied DOCX's §24.4 table contains literal `<` values where names should appear in IDs 0–7 and 16–63. The surrounding prose independently names `<pad>`, `<unk>`, `<cls>`, `<sep>`, `<mask>`, `<bos>`, `<eos>`, `<abstain>`, the eight metadata markers, chat role/control markers, document markers, and citation markers. The code keeps the stated ID ranges and uses the following provisional layout:

- IDs 0–7: pad, unk, cls, sep, mask, bos, eos, abstain.
- IDs 8–15: the eight metadata/field markers explicitly listed in §24.4.
- IDs 16–23: system, user, assistant, end, context, end-context, JSON, end-JSON.
- IDs 24–31: reserved; IDs 32–47: document markers; IDs 48–63: citation markers.
- IDs 64–255 remain reserved; IDs 256–511 are the byte alphabet; IDs 512–32,767 are learned BPE entries.

This is an engineering inference, not a verified plan fact. The release tokenizer CLI can build a local artifact, but the SDK will not treat a tokenizer as a production v1 release until a trained manifest records the exact mapping and the project owner has reviewed this table. No production tokenizer was trained here.

## Architecture experiment boundary

The plan keeps decoder-only ORION-300M as the default and identifies a 60M-scale causal/prefix-LM/encoder-decoder arbiter experiment. This repository implements the decoder-only model, prefix-LM attention, a pilot-scale GQA encoder-decoder, and a same-split E-300-1 runner. The runner was verified on tiny synthetic data only. It was not run on a real authorized corpus or approved compute, so no architecture recommendation is claimed; the decision remains open until that pilot is run.

## Unresolved policy decisions

- D8: whether teacher-generated synthetic examples are allowed, and which generator licenses/terms are acceptable.
- D9: legal review of the intended commercial/research release mode and the implications of ShareAlike data.
- D10: priority and acceptance thresholds for Urdu capability.
- D13: actual compute budget and target training hardware.
- Course-document authorization records and annotator availability.

These do not block implementation of generic pipeline and training software. They do block a defensible production corpus build and model-quality release.
