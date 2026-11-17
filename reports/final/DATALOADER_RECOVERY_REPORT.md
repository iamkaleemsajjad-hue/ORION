# DataLoader cursor recovery report

## Implementation

The sampler keeps dispatched and acknowledged/consumed positions separately. DataLoader worker prefetch may advance dispatch, but the trainer acknowledges a batch only as it enters the training loop; checkpoint state uses the consumed cursor and outstanding prefetched indices are replayed on resume. The implementation rejects out-of-order delivery and validates worker/prefetch options. Model runners consume the configured `num_workers`, `prefetch_factor`, and `persistent_workers` settings.

## Verification

The two-worker interrupted/resumed equivalence test checks that prefetch exceeds the consumed cursor and that uninterrupted and resumed model/RNG states agree. It is a real test, not mocked, but the current Windows sandbox skips it because worker named-pipe creation is denied. `wsl --list --quiet` returned `E_ACCESSDENIED`; Docker's Linux engine is unavailable. Thus Linux behavior is **implemented but not runtime-verified here**.

Exact recovery assumes ordered batches and deterministic/stateless worker transforms. Arbitrary custom worker-local random augmentation cannot be reconstructed from the main process cursor; the documented contract requires stateless per-sample randomness or `num_workers=0` for such transforms.

The required Ubuntu job is defined in `.github/workflows/ci.yml` but has not been observed running. Before relying on multi-worker recovery, run `tests/test_training_data.py::test_multiworker_prefetch_checkpoint_tracks_consumed_samples` on Linux.
