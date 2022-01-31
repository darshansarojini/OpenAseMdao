from pint import UnitRegistry
from pathlib import Path

ureg = UnitRegistry()
Q_ = ureg.Quantity

_REPO_ROOT_FOLDER = Path(__file__).parents[1]
_PACKAGE_ROOT_FOLDER = Path(__file__).parents[0]
_RESULTS_ROOT_FOLDER = _REPO_ROOT_FOLDER / 'results'