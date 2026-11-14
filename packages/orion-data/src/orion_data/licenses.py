"""Conservative license policy used by dataset manifests and mix builders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LicensePolicy:
    spdx_id: str
    commercial_clean: bool
    attribution_required: bool
    evaluation_only: bool = False
    notes: str = ""


LICENSES = {
    "ODC-By-1.0": LicensePolicy("ODC-By-1.0", True, True, notes="Keep dataset attribution."),
    "MIT": LicensePolicy("MIT", True, True),
    "Apache-2.0": LicensePolicy("Apache-2.0", True, True),
    "BSD-2-Clause": LicensePolicy("BSD-2-Clause", True, True),
    "BSD-3-Clause": LicensePolicy("BSD-3-Clause", True, True),
    "CC0-1.0": LicensePolicy("CC0-1.0", True, False),
    "CC-BY-4.0": LicensePolicy("CC-BY-4.0", True, True),
    "CC-BY-SA-3.0": LicensePolicy(
        "CC-BY-SA-3.0", False, True, notes="ShareAlike source; requires legal review."
    ),
    "CC-BY-SA-4.0": LicensePolicy(
        "CC-BY-SA-4.0", False, True, notes="ShareAlike source; requires legal review."
    ),
    "CC-BY-NC-3.0": LicensePolicy(
        "CC-BY-NC-3.0", False, True, evaluation_only=True, notes="Non-commercial; evaluation only."
    ),
    "GFDL-1.3": LicensePolicy("GFDL-1.3", False, True, notes="ShareAlike; requires legal review."),
    "LicenseRef-Unknown": LicensePolicy(
        "LicenseRef-Unknown",
        False,
        False,
        evaluation_only=True,
        notes="Unknown license; never train by default.",
    ),
}


def assert_license_allowed(spdx_id: str, mode: str, *, purpose: str = "training") -> LicensePolicy:
    if mode not in {"research", "commercial-clean"}:
        raise ValueError("mode must be 'research' or 'commercial-clean'")
    policy = LICENSES.get(spdx_id)
    if policy is None:
        raise ValueError(f"License {spdx_id!r} is not present in the reviewed license ledger")
    if policy.evaluation_only and purpose != "evaluation":
        raise ValueError(f"License {spdx_id!r} is evaluation-only")
    if mode == "commercial-clean" and not policy.commercial_clean:
        raise ValueError(f"License {spdx_id!r} is not allowed in a commercial-clean mix")
    return policy
