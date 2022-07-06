from enum import Enum, auto


class SectionType(Enum):
    CS = auto()
    INTERP = auto()
    FORCE = auto()
    JOINT = auto()


class BoundaryType(Enum):
    CANTILEVER = auto()
    FREEFREE = auto()


class BeamCS(Enum):
    RECTANGULAR = 2
    BOX = 6
    CIRCULAR = 2
