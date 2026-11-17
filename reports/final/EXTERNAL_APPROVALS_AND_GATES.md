# External approvals and gates

| Gate | Owner/action | Current evidence | Blocks |
|---|---|---|---|
| Package resolver/network | Restore approved package index/proxy; install supported uv; generate and review a real lock; install all eight packages with dependencies | `uv` and lock absent; diagnostic proxy refused connections; Tokenizers 0.22.2 | Clean installation and training command verification |
| Token-map scope | Current digest approved for experimental research; production use remains provisional | Approval recorded in `tokenizer_approval.md`; trained artifact and corpus remain absent | Preserve this digest, authorize corpus sources separately, and review any broader release. |
| Data authorization | Approve source, license, attribution, annotations, splits, privacy policy | No real corpus acquired | Dataset preparation/training/evaluation |
| Training hardware | Supply approved GPU and driver/backend; run model-specific preflight and memory smoke | Current host is CPU-only with low free RAM | Full-scale training; CUDA/NCCL and performance verification |
| Linux process verification | Execute Ubuntu multiprocess CI nodes | Both Gloo and worker recovery tests skipped here; WSL denied and Docker daemon absent | DDP and multi-worker recovery claims |
| Vulnerability/SBOM | Run scanner against resolved environment and generate transitive SBOM | pip-audit service was unreachable; current SBOM is unresolved source declarations | Security/release gate |
| Monitored offline test | Run installed local artifact suite with outbound traffic denied/observed | Proxy-blocked subset only, no firewall trace | Offline-runtime verification |
| E-300-1 | Approve data/compute and run equal-budget pilot; document predeclared metrics | Synthetic infrastructure only | Empirical 300M architecture decision |
| Full tokenizer/model training | Explicitly authorize corpus/tokenizer and model runs | Intentionally not executed | Learned capabilities and quality evaluation |
| Legal/privacy/security review | Obtain qualified review for intended sources and release mode | Software policies only; no certification | Real data use and public deployment |
