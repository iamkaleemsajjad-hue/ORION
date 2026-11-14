"""Independent ORION-100M document intelligence package."""

from orion_100m.config import Orion100MConfig
from orion_100m.sdk import Orion100M

__version__ = "0.1.0"
__all__ = ["Orion100M", "Orion100MConfig", "__version__"]
