# orion-core

Shared configuration, error handling, tokenizer contracts, safe artifact loading, device detection, logging, and Transformer building blocks used by ORION packages.

This package contains software infrastructure only. It does not include trained model weights or a production-trained tokenizer. Install it from the project's approved package source together with the ORION package whose workflows you need.

The `orion doctor` command reports local environment checks without downloading artifacts or starting training. Its status does not substitute for model-specific preflight or hardware validation.
