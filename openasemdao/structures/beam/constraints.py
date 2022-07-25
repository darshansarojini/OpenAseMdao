import openmdao.api as om
from casadi import *
from openasemdao.structures.utils.beam_categories import BeamCS


class StrengthAggregatedConstraint(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('name', types=str)  # Just to tag the constraint in particular
        self.options.declare('num_divisions', types=int)  # To generate optional constraint mechanisms
        self.options.declare('beam_shape', types=BeamCS)  # Important when defining the number of total cs variables
        self.options.declare('stress_computation')         # Stress related to the aggregator
        self.options.declare('total_stress_constraint', types=dict)
        self.options.declare('debug_flag', types=bool, default=False)  # To enable or disable debugging
        self.options.declare('sigmaY', default=276*10**6)  # Yield stress ADD DEFAULTS
        self.options.declare('rho_KS', default=60.0)       # stress_constraint penalty parameter
        self.symbolic_expressions = {}
        self.symbolic_stress_functions = {}
        self.constraint_bounds = []
        self.options['total_stress_constraint'] = {}

    def setup(self):

        """
        RECTANGULAR
                            The following are the stresses modeled in the rectangular beam:
                            T x 0n -> 4n : Axial Stresses at the corners
                            T x 4n -> 8n : Von Misses Stress at the edges (center of edge)
                            T x 8n -> 9n : Shear stress at the horizontal beam direction
                            T x 9n -> 10n: Shear stress at the vertical beam direction
        """
        """
        BOX
                            The following are the stresses modeled in the rectangular beam:
                            T x 0n -> 4n : von-mises stress at the corners
                            T x 4n -> 8n : axial stress at the corners
                            T x 8n -> 12n : Shear stress at the flange centers
        """
        if self.options['beam_shape'] == BeamCS.RECTANGULAR:
            self.constraint_bounds = [[0, 4], [4, 8], [8, 9], [9, 10]]
        elif self.options['beam_shape'] == BeamCS.BOX:
            self.constraint_bounds = [[0, 4], [4, 8], [8, 12]]
        else:
            raise NotImplementedError("Will add more stress models shortly")

        self.stress_input = self.options['stress_computation']

        stress_input = SX.sym('sigma_in', self.stress_input.options['symbolic_stress_functions']['sigma'].size_out(0)[0],
                              self.stress_input.options['symbolic_stress_functions']['sigma'].size_out(0)[1])

        self.add_input('sigma', shape=(self.stress_input.options['symbolic_stress_functions']['sigma'].size_out(0)[0],
                                       self.stress_input.options['symbolic_stress_functions']['sigma'].size_out(0)[1]))

        self.add_output('cs_strength', shape=(len(self.constraint_bounds), 2))

        J = 0
        for a_constraint_bound in self.constraint_bounds:
            stress_constraint = stress_input[a_constraint_bound[0]*self.options['num_divisions']:a_constraint_bound[1]*self.options['num_divisions'], :] / self.options['sigmaY'] - 1
            max_stress_constraint_space = mmax(stress_constraint)
            A = sum2(sum1(exp(self.options['rho_KS'] * (stress_constraint - max_stress_constraint_space))))
            aggregated_stress_constraint_positive = max_stress_constraint_space + (1 / self.options['rho_KS']) * log(A)

            stress_constraint_n = -stress_input[a_constraint_bound[0]*self.options['num_divisions']:a_constraint_bound[1]*self.options['num_divisions'], :] / self.options['sigmaY'] - 1
            max_stress_constraint_space_n = mmax(stress_constraint_n)
            A_n = sum2(sum1(exp(self.options['rho_KS'] * (stress_constraint_n - max_stress_constraint_space_n))))
            aggregated_stress_constraint_negative = max_stress_constraint_space_n + (1 / self.options['rho_KS']) * log(A_n)

            self.symbolic_expressions['total_stress_constraint_'+str(J)] = vertcat(aggregated_stress_constraint_positive, aggregated_stress_constraint_negative)

            self.options['total_stress_constraint']['sym_funct_' + str(J)] = Function("aggregated_stress_constraint",
                                                                                      [stress_input[
                                                                                       a_constraint_bound[0] * self.options['num_divisions']:a_constraint_bound[1] * self.options['num_divisions'], :]],
                                                                                      [self.symbolic_expressions['total_stress_constraint_'+str(J)]])
            J += 1


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options['debug_flag']:
            sigma_in = inputs['sigma']

            J = 0
            for a_constraint_bound in self.constraint_bounds:

                stress_constraint = sigma_in[a_constraint_bound[0]*self.options['num_divisions']:a_constraint_bound[1]*self.options['num_divisions'], :] / self.options['sigmaY'] - 1
                stress_constraint_n = -sigma_in[a_constraint_bound[0]*self.options['num_divisions']:a_constraint_bound[1]*self.options['num_divisions'], :] / self.options['sigmaY'] - 1

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

                outputs['cs_strength'][J, :] = np.asarray([aggregated_stress_constraint_positive, aggregated_stress_constraint_negative])
                J += 1
        else:
            J = 0
            cs = np.zeros((len(self.constraint_bounds), 2))
            for a_constraint_bound in self.constraint_bounds:
                cs[J, :] = np.squeeze(self.options['total_stress_constraint']['sym_funct_' + str(J)](
                    inputs['sigma'][a_constraint_bound[0] * self.options['num_divisions']:a_constraint_bound[1] * self.options['num_divisions'], :]).full())
                J+=1
            outputs['cs_strength'] = cs
