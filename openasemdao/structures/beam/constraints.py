import openmdao.api as om
from casadi import *

class StrenghtAggregatedConstraint(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('name', types=str)  # Just to tag the constraint in particular
        self.options.declare('num_divisions', types=int)  # To generate optional constraint mechanisms
        self.options.declare('num_cs_variables', types=int)  # Important when defining the number of total cs variables
        self.options.declare(
            'symbolic_variables')  # Where all the resultant cross-section symbolics come from beam parent
        self.options.declare('sigmaY', default=276*10**6)  # Yield stress ADD DEFAULTS
        self.options.declare('rho_KS', default=60.0)  # stress_constraint penalty parameter
        self.symbolic_expressions = {}
        self.symbolic_stress_functions = {}

        self.symbolic_stress_functions['total_stress_constraint'] = {}
        self.symbolic_stress_functions['total_stress_constraint_jac'] = {}

    def setup(self):
        self.beam_symbolics = self.options['symbolic_variables']

        cs = self.beam_symbolics['cs']

        stress = self.beam_symbolics['sigma'] # Symbolic expression denoting the stress within the beam

        driving_parameters = self.beam_symbolics[
            'driving_parameters']  # Row vector symbolic parameters (apart from cs variables)
                                   # that drive the stress function: e.g. sigma = f(Mx, corner point symbolics)

        stress_constraint = stress / self.options['sigmaY'] - 1

        max_stress_constraint_space = mmax(stress_constraint)

        A = sum2(sum1(exp(self.options['rho_KS'] * (stress_constraint - max_stress_constraint_space))))

        aggregated_stress_constraint = max_stress_constraint_space + (1 / self.options['rho_KS']) * log(A)

        self.symbolic_expressions['stress'] = stress
        self.symbolic_stress_functions['stress'] = Function("stress", [cs, driving_parameters], [stress])
        self.symbolic_expressions['stress_constraint'] = stress
        self.symbolic_stress_functions['stress_constraint'] = Function("stress_constraint", [cs, driving_parameters],
                                                                [stress_constraint])
        self.symbolic_expressions['total_stress_constraint'] = aggregated_stress_constraint

        self.symbolic_expressions['total_stress_constraint_jac'] = jacobian(aggregated_stress_constraint,
                                                                                 cs)

        self.symbolic_stress_functions['total_stress_constraint'] = Function("aggregated_stress_constraint",
                                                                           [cs, driving_parameters],
                                                                           [self.symbolic_expressions[
                                                                                    'total_stress_constraint']])

        self.symbolic_stress_functions['total_stress_constraint_jac'] = Function("aggregated_stress_constraint_jac",
                                                                               [cs, driving_parameters],
                                                                               [self.symbolic_expressions[
                                                                                    'total_stress_constraint_jac']])

        self.symbolic_stress_functions['max_stress_constraint'] = Function("aggregated_stress_constraint",
                                                                    [cs, driving_parameters],
                                                                    [max_stress_constraint_space])