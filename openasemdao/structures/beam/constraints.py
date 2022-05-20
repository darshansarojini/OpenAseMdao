import openmdao.api as om
from casadi import *
from openasemdao.structures.utils.cs_variables import BeamCS
     
class StrenghtAggregatedConstraint(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('name', types=str)  # Just to tag the constraint in particular
        self.options.declare('num_divisions', types=int)  # To generate optional constraint mechanisms
        self.options.declare('beam_shape', types=str)  # Important when defining the number of total cs variables
        self.options.declare('stress_computation')         # Stress related to the aggregator
        self.options.declare('total_stress_constraint', types=dict)
        self.options.declare('debug_flag', types=bool, default=False)  # To enable or disable debugging
        self.options.declare('sigmaY', default=276*10**6)  # Yield stress ADD DEFAULTS
        self.options.declare('rho_KS', default=60.0)       # stress_constraint penalty parameter
        self.symbolic_expressions = {}
        self.symbolic_stress_functions = {}

        self.options['total_stress_constraint'] = {}

    def setup(self):
        self.stress_input = self.options['stress_computation']

        stress_input = SX.sym('sigma_in', self.stress_input.options['symbolic_stress_functions']['sigma'].size_out(0)[0], self.stress_input.options['symbolic_stress_functions']['sigma'].size_out(0)[1])

        # Tensile constraint

        stress_constraint = stress_input / self.options['sigmaY'] - 1

        max_stress_constraint_space = mmax(stress_constraint)

        A = sum2(sum1(exp(self.options['rho_KS'] * (stress_constraint - max_stress_constraint_space))))

        aggregated_stress_constraint_positive = max_stress_constraint_space + (1 / self.options['rho_KS']) * log(A)

        # Compressive constraint

        stress_constraint_n = -stress_input / self.options['sigmaY'] - 1

        max_stress_constraint_space_n = mmax(stress_constraint_n)

        A_n = sum2(sum1(exp(self.options['rho_KS'] * (stress_constraint_n - max_stress_constraint_space_n))))

        aggregated_stress_constraint_negative = max_stress_constraint_space_n + (1 / self.options['rho_KS']) * log(A_n)

        self.add_input('sigma', shape=(self.stress_input.options['symbolic_stress_functions']['sigma'].size_out(0)[0],
                                                   self.stress_input.options['symbolic_stress_functions']['sigma'].size_out(0)[1]))

        self.add_output('cs_strength', shape=(2, 1))

        self.symbolic_expressions['total_stress_constraint'] = vertcat(aggregated_stress_constraint_positive, aggregated_stress_constraint_negative)

        self.options['total_stress_constraint']['sym_funct'] = Function("aggregated_stress_constraint",
                                                                           [stress_input],
                                                                           [self.symbolic_expressions['total_stress_constraint']])
        pass

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options['debug_flag']:
            sigma_in = inputs['sigma']

            stress_constraint = sigma_in / self.options['sigmaY'] - 1
            stress_constraint_n = -sigma_in / self.options['sigmaY'] - 1

            max_stress_constraint_space = np.max(stress_constraint)
            max_stress_constraint_space_n = np.max(stress_constraint_n)

            A = np.sum(
                np.sum(np.exp(self.options['rho_KS'] * (stress_constraint - max_stress_constraint_space))))

            A_n = np.sum(
                np.sum(np.exp(self.options['rho_KS'] * (stress_constraint_n - max_stress_constraint_space_n))))

            aggregated_stress_constraint_positive = max_stress_constraint_space + (1 / self.options['rho_KS']) * np.log(
                A)

            aggregated_stress_constraint_negative = max_stress_constraint_space_n + (
                        1 / self.options['rho_KS']) * np.log(
                A_n)

            outputs['cs_strength'] = np.asarray([aggregated_stress_constraint_positive, aggregated_stress_constraint_negative])

        else:
            cs = self.options['total_stress_constraint']['sym_funct'](inputs['sigma'])
            outputs['cs_strength'] = cs.full()