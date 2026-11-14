"""Independent ORION-300M educational generation package."""

from orion_300m.config import Orion300MConfig
from orion_300m.sdk import GenerationResult, Orion300M

__version__ = "0.1.0"
__all__ = ["GenerationResult", "Orion300M", "Orion300MConfig", "__version__"]
