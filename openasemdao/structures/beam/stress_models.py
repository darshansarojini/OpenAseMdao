import openmdao.api as om
from casadi import *


class EulerBernoulliStressModel(om.ExplicitComponent):
    def initialize(self):
        # Make sure all the quantities necessary to make the system work are here
        self.options.declare('name', types=str)  # Just to tag the constraint in particular
        self.options.declare('num_divisions', types=int)  # To generate optional constraint mechanisms
        self.options.declare(
            'symbolic_variables')  # Where all the resultant cross-section symbolics come from beam parent

        # Attempting an experimental way to store casadi expressions within the component
        self.options.declare('symbolic_expressions', types=dict)
        self.options.declare('symbolic_stress_functions', types=dict)

        self.options.declare('axial_point_list')
        self.options.declare('vm_point_list')
        self.options.declare('shear_point_list')

        self.options['symbolic_expressions'] = {}
        self.options['symbolic_stress_functions'] = {}

    def setup(self):
        # Setup all the nice goodies that make this thing work
        pass
