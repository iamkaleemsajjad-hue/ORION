# Model release checklist

- [ ] Approve the frozen tokenizer token spellings and ID table; verify golden vectors.
- [ ] Record corpus source, immutable revision, license evidence, provenance, PII policy, and checksums.
- [ ] Record exact training configuration, source commit, seed, runtime versions, and safe checkpoint hashes.
- [ ] Run architecture and evaluation suites; publish actual reports without replacing failed metrics with targets.
- [ ] Complete calibration, open-set, contamination, leakage, robustness, and human review gates from the implementation plan.
- [ ] Verify final weights and tokenizer hashes with `tools/verify_release.py`.
- [ ] Confirm release cards state what was and was not evaluated and list known limits.
- [ ] Create a private Hub repository only when explicitly requested and upload only after verification.
- [ ] Confirm no unsafe pickle files, downloaded datasets, credentials, private student records, or test fixtures are present.
- [ ] Obtain governance, licensing, privacy, and security review before any public release.

The current repository cannot pass this checklist: there is no production tokenizer,
training report, evaluation report, or trained model release.
