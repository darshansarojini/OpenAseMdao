from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core import group
from abc import ABC, abstractmethod
import numpy as np
from casadi import *


class SymbolicBeam(group, ABC):
    """
        Group that contains the symbolic beam functions that will be used for the structure. It will include
        the definition of the different beam constants, as well as the proper point distribution based on the joints
        and point loads.
    """
    def initialize(self):
        self.options.declare("beam_definition", default=None)
        self.options.declare("applied_loads", default=[])
        self.options.declare("joints", default=[])

        # Beam axis node locations
        self.options.declare('r0')
        # Beam rotation sequence
        self.options.declare('seq')
        # Beam axis initial angles
        self.options.declare('th0')
        # Beam s0
        self.options.declare('delta_s0')
        # Beam node_lim and inter_node_lim
        self.options.declare('node_lim')
        self.options.declare('inter_node_lim')
        # Beam K0a
        self.options.declare('K0a')
        # Beam Initial Conditions:
        self.options.declare('x0')
        self.options.declare('xDot0')

        # Boundary Conditions Holder:
        self.BC = {'tip': 8888 * np.ones(12),
                   'root': 8888 * np.ones(12)}
        # Empty Casadi containers
        self.symbolic_expressions = {}
        self.symbolic_functions = {}
        self.symbolics = {}
        # Additional inputs at initialize
        self.declare_additional_beam_inputs()
        return

    @abstractmethod
    def declare_additional_beam_inputs(self):
        return

    def setup(self):
        beam_definition = self.options["beam_definition"]
        applied_loads = self.options["applied_loads"]
        joints = self.options["joints"]

        # Define basic beam parameters from containers:
        self.options["seq"] = beam_definition.orientation
