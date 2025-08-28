"""Single-cell embedding comparison utilities and evaluation."""

from importlib.metadata import version

from .aggregation import scIBAggregator
from .evaluation import IntegrationEvaluator
from .factory import get_method_instance
from .logging import logger

__all__ = ["IntegrationEvaluator", "scIBAggregator", "get_method_instance", "logger"]

__version__ = version("scembed")
