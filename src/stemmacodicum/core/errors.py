class StemmaError(Exception):
    """Base error for all user-facing Stemma exceptions."""


class ConfigurationError(StemmaError):
    """Raised when configuration is invalid or incomplete."""


class ProjectNotInitializedError(StemmaError):
    """Raised when .stemma metadata is missing."""


class ValidationError(StemmaError):
    """Raised when model invariants fail."""


class ResourceIngestError(StemmaError):
    """Raised when a resource cannot be ingested."""


class ReferenceError(StemmaError):
    """Raised when citation/reference operations fail."""


class ExtractionError(StemmaError):
    """Raised when extraction runs fail."""


class ClaimError(StemmaError):
    """Raised when claim operations fail."""


class EvidenceBindingError(StemmaError):
    """Raised when evidence binding operations fail."""


class VerificationError(StemmaError):
    """Raised when verification operations fail."""


class ReportingError(StemmaError):
    """Raised when reporting operations fail."""


class TraceError(StemmaError):
    """Raised when trace requests cannot be resolved."""


class CEAPFError(StemmaError):
    """Raised when CEAPF graph operations fail."""
