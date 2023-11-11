class NothingToDo(Exception):
    """Raised when there's nothing to do"""


class AlignError(Exception):
    """Raised when cannot align"""


class NumberOfElementError(Exception):
    """Raised when number of elements never met"""


class OverCorrection(Exception):
    """Raised when trying to correct a file multiple times"""


class ValueNotFound(Exception):
    """Raised when the key not found in the haeder"""
