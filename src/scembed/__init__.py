"""Single-cell embedding comparison utilities and evaluation."""

from importlib.metadata import version

from . import methods
from .aggregation import scIBAggregator
from .evaluation import IntegrationEvaluator
from .factory import get_method_instance
from .logging import logger

__all__ = ["IntegrationEvaluator", "scIBAggregator", "get_method_instance", "logger", "methods"]

__version__ = version("scembed")
