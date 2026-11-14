# GitHub workflows

`ci.yml` runs compile checks, Ruff lint/format, and the synthetic test suite on
Python 3.10/3.12 for Linux and Windows. `security.yml` audits dependencies and
rejects pickle-based model loading references. Dependabot checks Python and
GitHub Actions dependencies weekly. `release.yml` is manual and builds Python
package wheels as an Actions artifact; it does not publish model weights or
upload packages to PyPI.

Keep checkpoints and datasets out of normal Git history. Use the model release
verifier and human governance review before any artifact distribution.
