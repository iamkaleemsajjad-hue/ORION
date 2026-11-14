# Offline installation and deployment

Build a wheelhouse on a connected, approved machine and transfer the wheels and
hash manifest through the organization's normal artifact process. Install the
packages in dependency order with `pip --no-index --find-links <wheelhouse>`.
Keep FastAPI/Uvicorn, Parquet, and Hub clients optional unless the deployment
needs them.

Place a future trained release under a local model directory and configure
`ORION_100M_WEIGHTS` or `ORION_300M_HOME`. Include the safetensors checksum,
versioned tokenizer files, model configuration, and data/evaluation provenance.
Run `tools/verify_release.py` before serving. Set `ORION_API_TOKEN` if local
clients need authentication. Both servers use loopback by default and never
download artifacts or send telemetry.
