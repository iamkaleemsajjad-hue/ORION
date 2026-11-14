"""Optional host contract adapters; this package imports no MADAD code."""

from .citations import CitationResult, VerifiedCitation, verify_citations
from .contracts import (
    CourseProfile,
    OrganizerDocument,
    OrganizerRequest,
    RetrievedDocument,
    TutorRequest,
)
from .organizer import create_organization_proposal, to_classifier_inputs
from .packing import PackedContext, pack_documents
from .sdk_bridge import Orion100MClassifierAdapter, Orion300MTutorAdapter
from .tutor import answer_from_retrieval

__all__ = [
    "CitationResult",
    "CourseProfile",
    "OrganizerDocument",
    "OrganizerRequest",
    "Orion100MClassifierAdapter",
    "Orion300MTutorAdapter",
    "PackedContext",
    "RetrievedDocument",
    "TutorRequest",
    "VerifiedCitation",
    "answer_from_retrieval",
    "create_organization_proposal",
    "pack_documents",
    "to_classifier_inputs",
    "verify_citations",
]
