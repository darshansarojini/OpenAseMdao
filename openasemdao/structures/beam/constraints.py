import openmdao.api as om
from casadi import *
from openasemdao.structures.utils.beam_categories import BeamCS
from scipy.sparse import csc_matrix

class StrengthAggregatedConstraint(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('name', types=str)  # Just to tag the constraint in particular
        self.options.declare('num_divisions', types=int)  # To generate optional constraint mechanisms
        self.options.declare('num_DvCs', types=int)  # To know the actual number of cross-sectional variables
        self.options.declare('beam_shape', types=BeamCS)  # Important when defining the number of total cs variables
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

            self.add_input('sigma_axial', shape=(self.stress_input.options['symbolic_stress_functions']['sigma_axial'].size_out(0)[0]))
            self.add_input('sigma_vm', shape=(self.stress_input.options['symbolic_stress_functions']['sigma_vm_w'].size_out(0)[0]))
            self.add_input('tau_max_c', shape=(self.stress_input.options['symbolic_stress_functions']['tau_max_c'].size_out(0)[0]))
            self.add_input('tau_max_n', shape=(self.stress_input.options['symbolic_stress_functions']['tau_max_n'].size_out(0)[0]))

            axial_input = SX.sym('axial_in', self.stress_input.options['symbolic_stress_functions']['sigma_axial'].size_out(0)[0],
                                  self.stress_input.options['symbolic_stress_functions']['sigma_axial'].size_out(0)[1])
            vm_input = SX.sym('vm_in', self.stress_input.options['symbolic_stress_functions']['sigma_vm_h'].size_out(0)[0],
                                 self.stress_input.options['symbolic_stress_functions']['sigma_vm_h'].size_out(0)[1])
            tau_max_c_input = SX.sym('tau_max_c_in', self.stress_input.options['symbolic_stress_functions']['tau_max_c'].size_out(0)[0],
                              self.stress_input.options['symbolic_stress_functions']['tau_max_c'].size_out(0)[1])
            tau_max_n_input = SX.sym('tau_max_n_in', self.stress_input.options['symbolic_stress_functions']['tau_max_n'].size_out(0)[0],
                              self.stress_input.options['symbolic_stress_functions']['tau_max_n'].size_out(0)[1])

            axial_stress_constraint = axial_input / self.options['sigmaY'] - 1
            max_axial_constraint_space = mmax(axial_stress_constraint)
            A_axial = sum2(sum1(exp(self.options['rho_KS'] * (axial_stress_constraint - max_axial_constraint_space))))
            aggregated_axial_constraint_positive = max_axial_constraint_space + (1 / self.options['rho_KS']) * log(A_axial)

            self.symbolic_expressions['c_axial'] = aggregated_axial_constraint_positive
            self.options['total_stress_constraint']['c_axial'] = Function("c_axial", [reshape(axial_input, (axial_input.shape[0] * axial_input.shape[1], 1))], [aggregated_axial_constraint_positive])

            self.symbolic_expressions['d_c_axial'] = jacobian(aggregated_axial_constraint_positive, reshape(axial_input, (axial_input.shape[0] * axial_input.shape[1], 1)))
            self.options['total_stress_constraint']['d_c_axial'] = Function("d_c_axial", [reshape(axial_input, (axial_input.shape[0] * axial_input.shape[1], 1))],
                                                                            [self.symbolic_expressions['d_c_axial']])

            self.add_output('c_axial')

            axial_stress_constraint_n = -axial_input / self.options['sigmaY'] - 1
            max_axial_constraint_space_n = mmax(axial_stress_constraint_n)
            A_axial_n = sum2(sum1(exp(self.options['rho_KS'] * (axial_stress_constraint_n - max_axial_constraint_space_n))))
            aggregated_axial_constraint_negative = max_axial_constraint_space_n + (1 / self.options['rho_KS']) * log(A_axial_n)

            self.symbolic_expressions['c_axial_n'] = aggregated_axial_constraint_negative
            self.options['total_stress_constraint']['c_axial_n'] = Function("c_axial_n", [reshape(axial_input, (axial_input.shape[0] * axial_input.shape[1], 1))], [aggregated_axial_constraint_negative])

            self.symbolic_expressions['d_c_axial_n'] = jacobian(aggregated_axial_constraint_negative, reshape(axial_input, (axial_input.shape[0] * axial_input.shape[1], 1)))
            self.options['total_stress_constraint']['d_c_axial_n'] = Function("d_c_axial_n", [reshape(axial_input, (axial_input.shape[0] * axial_input.shape[1], 1))],
                                                                            [self.symbolic_expressions['d_c_axial_n']])

            self.add_output('c_axial_n')

            vm_stress_constraint = vm_input / self.options['sigmaY'] - 1
            max_vm_constraint_space = mmax(vm_stress_constraint)
            A_vm = sum2(sum1(exp(self.options['rho_KS'] * (vm_stress_constraint - max_vm_constraint_space))))
            aggregated_vm_constraint = max_vm_constraint_space + (1 / self.options['rho_KS']) * log(A_vm)

            self.symbolic_expressions['c_vm'] = aggregated_vm_constraint
            self.options['total_stress_constraint']['c_vm'] = Function("c_vm", [reshape(vm_input, (vm_input.shape[0] * vm_input.shape[1], 1))], [aggregated_vm_constraint])

            self.symbolic_expressions['d_c_vm'] = jacobian(aggregated_vm_constraint, reshape(vm_input, (vm_input.shape[0] * vm_input.shape[1], 1)))
            self.options['total_stress_constraint']['d_c_vm'] = Function("d_c_vm", [reshape(vm_input, (vm_input.shape[0] * vm_input.shape[1], 1))],
                                                                              [self.symbolic_expressions['d_c_vm']])
            self.add_output('c_vm')

            tau_max_c_stress_constraint = tau_max_c_input / self.options['sigmaY'] - 1
            max_tau_max_c_constraint_space = mmax(tau_max_c_stress_constraint)
            A_tau_max_c = sum2(sum1(exp(self.options['rho_KS'] * (tau_max_c_stress_constraint - max_tau_max_c_constraint_space))))
            aggregated_tau_max_c_constraint_positive = max_tau_max_c_constraint_space + (1 / self.options['rho_KS']) * log(A_tau_max_c)

            self.symbolic_expressions['c_tau_max_c'] = aggregated_tau_max_c_constraint_positive
            self.options['total_stress_constraint']['c_tau_max_c'] = Function("c_tau_max_c", [reshape(tau_max_c_input, (tau_max_c_input.shape[0] * tau_max_c_input.shape[1], 1))],
                                                                              [aggregated_tau_max_c_constraint_positive])

            self.symbolic_expressions['d_c_tau_max_c'] = jacobian(aggregated_tau_max_c_constraint_positive, reshape(tau_max_c_input, (tau_max_c_input.shape[0] * tau_max_c_input.shape[1], 1)))
            self.options['total_stress_constraint']['d_c_tau_max_c'] = Function("d_c_tau_max_c", [reshape(tau_max_c_input, (tau_max_c_input.shape[0] * tau_max_c_input.shape[1], 1))],
                                                                         [self.symbolic_expressions['d_c_tau_max_c']])
            self.add_output('c_tau_max_c')

            tau_max_c_stress_constraint_n = -tau_max_c_input / self.options['sigmaY'] - 1
            max_tau_max_c_constraint_space_n = mmax(tau_max_c_stress_constraint_n)
            A_tau_max_c_n = sum2(sum1(exp(self.options['rho_KS'] * (tau_max_c_stress_constraint_n - max_tau_max_c_constraint_space_n))))
            aggregated_tau_max_c_constraint_negative = max_tau_max_c_constraint_space_n + (1 / self.options['rho_KS']) * log(A_tau_max_c_n)

            self.symbolic_expressions['c_tau_max_c_n'] = aggregated_tau_max_c_constraint_negative
            self.options['total_stress_constraint']['c_tau_max_c_n'] = Function("c_tau_max_c_n", [reshape(tau_max_c_input, (tau_max_c_input.shape[0] * tau_max_c_input.shape[1], 1))],
                                                                              [aggregated_tau_max_c_constraint_negative])

            self.symbolic_expressions['d_c_tau_max_c_n'] = jacobian(aggregated_tau_max_c_constraint_negative, reshape(tau_max_c_input, (tau_max_c_input.shape[0] * tau_max_c_input.shape[1], 1)))
            self.options['total_stress_constraint']['d_c_tau_max_c_n'] = Function("d_c_tau_max_c_n", [reshape(tau_max_c_input, (tau_max_c_input.shape[0] * tau_max_c_input.shape[1], 1))],
                                                                                [self.symbolic_expressions['d_c_tau_max_c_n']])

            self.add_output('c_tau_max_c_n')

            tau_max_n_stress_constraint = tau_max_n_input / self.options['sigmaY'] - 1
            max_tau_max_n_constraint_space = mmax(tau_max_n_stress_constraint)
            A_tau_max_n = sum2(sum1(exp(self.options['rho_KS'] * (tau_max_n_stress_constraint - max_tau_max_n_constraint_space))))
            aggregated_tau_max_n_constraint_positive = max_tau_max_n_constraint_space + (1 / self.options['rho_KS']) * log(A_tau_max_n)

            self.symbolic_expressions['c_tau_max_n'] = aggregated_tau_max_n_constraint_positive
            self.options['total_stress_constraint']['c_tau_max_n'] = Function("c_tau_max_n", [reshape(tau_max_n_input, (tau_max_n_input.shape[0] * tau_max_n_input.shape[1], 1))],
                                                                              [aggregated_tau_max_n_constraint_positive])

            self.symbolic_expressions['d_c_tau_max_n'] = jacobian(aggregated_tau_max_n_constraint_positive, reshape(tau_max_n_input, (tau_max_n_input.shape[0] * tau_max_n_input.shape[1], 1)))
            self.options['total_stress_constraint']['d_c_tau_max_n'] = Function("d_c_tau_max_n", [reshape(tau_max_n_input, (tau_max_n_input.shape[0] * tau_max_n_input.shape[1], 1))],
                                                                                  [self.symbolic_expressions['d_c_tau_max_n']])

            self.add_output('c_tau_max_n')

            tau_max_n_stress_constraint_n = -tau_max_n_input / self.options['sigmaY'] - 1
            max_tau_max_n_constraint_space_n = mmax(tau_max_n_stress_constraint_n)
            A_tau_max_n_n = sum2(sum1(exp(self.options['rho_KS'] * (tau_max_n_stress_constraint_n - max_tau_max_n_constraint_space_n))))
            aggregated_tau_max_n_constraint_negative = max_tau_max_n_constraint_space_n + (1 / self.options['rho_KS']) * log(A_tau_max_n_n)

            self.symbolic_expressions['c_tau_max_n_n'] = aggregated_tau_max_n_constraint_negative
            self.options['total_stress_constraint']['c_tau_max_n_n'] = Function("c_tau_max_n_n", [reshape(tau_max_n_input, (tau_max_n_input.shape[0] * tau_max_n_input.shape[1], 1))],
                                                                                [aggregated_tau_max_n_constraint_negative])

            self.symbolic_expressions['d_c_tau_max_n_n'] = jacobian(aggregated_tau_max_n_constraint_negative, reshape(tau_max_n_input, (tau_max_n_input.shape[0] * tau_max_n_input.shape[1], 1)))
            self.options['total_stress_constraint']['d_c_tau_max_n_n'] = Function("d_c_tau_max_n_n", [reshape(tau_max_n_input, (tau_max_n_input.shape[0] * tau_max_n_input.shape[1], 1))],
                                                                                [self.symbolic_expressions['d_c_tau_max_n_n']])
            self.add_output('c_tau_max_n_n')
            self.declare_partials('c_axial', 'sigma_axial', val=csc_matrix((self.symbolic_expressions['d_c_axial'].shape[0],self.symbolic_expressions['d_c_axial'].shape[1]), dtype=np.float64))
            self.declare_partials('c_axial_n', 'sigma_axial', val=csc_matrix((self.symbolic_expressions['d_c_axial_n'].shape[0],self.symbolic_expressions['d_c_axial_n'].shape[1]), dtype=np.float64))
            self.declare_partials('c_vm', 'sigma_vm', val=csc_matrix((self.symbolic_expressions['d_c_vm'].shape[0],self.symbolic_expressions['d_c_vm'].shape[1]), dtype=np.float64))
            self.declare_partials('c_tau_max_c', 'tau_max_c', val=csc_matrix((self.symbolic_expressions['d_c_tau_max_c'].shape[0],self.symbolic_expressions['d_c_tau_max_c'].shape[1]), dtype=np.float64))
            self.declare_partials('c_tau_max_c_n', 'tau_max_c', val=csc_matrix((self.symbolic_expressions['d_c_tau_max_c_n'].shape[0],self.symbolic_expressions['d_c_tau_max_c_n'].shape[1]), dtype=np.float64))
            self.declare_partials('c_tau_max_n', 'tau_max_n', val=csc_matrix((self.symbolic_expressions['d_c_tau_max_n'].shape[0],self.symbolic_expressions['d_c_tau_max_n'].shape[1]), dtype=np.float64))
            self.declare_partials('c_tau_max_n_n', 'tau_max_n', val=csc_matrix((self.symbolic_expressions['d_c_tau_max_n_n'].shape[0],self.symbolic_expressions['d_c_tau_max_n_n'].shape[1]), dtype=np.float64))

        elif self.options['beam_shape'] == BeamCS.BOX:
            self.add_input('sigma_axial', shape=(self.stress_input.options['symbolic_stress_functions']['sigma_axial'].size_out(0)[0]))
            self.add_input('sigma_vm', shape=(self.stress_input.options['symbolic_stress_functions']['sigma_vm'].size_out(0)[0]))
            self.add_input('tau_side', shape=(self.stress_input.options['symbolic_stress_functions']['tau_side'].size_out(0)[0]))

            axial_input = SX.sym('axial_in', self.stress_input.options['symbolic_stress_functions']['sigma_axial'].size_out(0)[0],
                                 self.stress_input.options['symbolic_stress_functions']['sigma_axial'].size_out(0)[1])
            vm_input = SX.sym('vm_in', self.stress_input.options['symbolic_stress_functions']['sigma_vm'].size_out(0)[0],
                              self.stress_input.options['symbolic_stress_functions']['sigma_vm'].size_out(0)[1])
            tau_side_input = SX.sym('tau_side_in', self.stress_input.options['symbolic_stress_functions']['tau_side'].size_out(0)[0],
                                     self.stress_input.options['symbolic_stress_functions']['tau_side'].size_out(0)[1])

            axial_stress_constraint = axial_input / self.options['sigmaY'] - 1
            max_axial_constraint_space = mmax(axial_stress_constraint)
            A_axial = sum2(sum1(exp(self.options['rho_KS'] * (axial_stress_constraint - max_axial_constraint_space))))
            aggregated_axial_constraint_positive = max_axial_constraint_space + (1 / self.options['rho_KS']) * log(A_axial)

            self.symbolic_expressions['c_axial'] = aggregated_axial_constraint_positive
            self.options['total_stress_constraint']['c_axial'] = Function("c_axial", [reshape(axial_input, (axial_input.shape[0] * axial_input.shape[1], 1))], [aggregated_axial_constraint_positive])

            self.symbolic_expressions['d_c_axial'] = jacobian(aggregated_axial_constraint_positive, reshape(axial_input, (axial_input.shape[0] * axial_input.shape[1], 1)))
            self.options['total_stress_constraint']['d_c_axial'] = Function("d_c_axial", [reshape(axial_input, (axial_input.shape[0] * axial_input.shape[1], 1))],
                                                                            [self.symbolic_expressions['d_c_axial']])

            self.add_output('c_axial')

            axial_stress_constraint_n = -axial_input / self.options['sigmaY'] - 1
            max_axial_constraint_space_n = mmax(axial_stress_constraint_n)
            A_axial_n = sum2(sum1(exp(self.options['rho_KS'] * (axial_stress_constraint_n - max_axial_constraint_space_n))))
            aggregated_axial_constraint_negative = max_axial_constraint_space_n + (1 / self.options['rho_KS']) * log(A_axial_n)

            self.symbolic_expressions['c_axial_n'] = aggregated_axial_constraint_negative
            self.options['total_stress_constraint']['c_axial_n'] = Function("c_axial_n", [reshape(axial_input, (axial_input.shape[0] * axial_input.shape[1], 1))],
                                                                            [aggregated_axial_constraint_negative])

            self.symbolic_expressions['d_c_axial_n'] = jacobian(aggregated_axial_constraint_negative, reshape(axial_input, (axial_input.shape[0] * axial_input.shape[1], 1)))
            self.options['total_stress_constraint']['d_c_axial_n'] = Function("d_c_axial_n", [reshape(axial_input, (axial_input.shape[0] * axial_input.shape[1], 1))],
                                                                            [self.symbolic_expressions['d_c_axial_n']])

            self.add_output('c_axial_n')

            vm_stress_constraint = vm_input / self.options['sigmaY'] - 1
            max_vm_constraint_space = mmax(vm_stress_constraint)
            A_vm = sum2(sum1(exp(self.options['rho_KS'] * (vm_stress_constraint - max_vm_constraint_space))))
            aggregated_vm_constraint = max_vm_constraint_space + (1 / self.options['rho_KS']) * log(A_vm)

            self.symbolic_expressions['c_vm'] = aggregated_vm_constraint
            self.options['total_stress_constraint']['c_vm'] = Function("c_vm", [reshape(vm_input, (vm_input.shape[0] * vm_input.shape[1], 1))], [aggregated_vm_constraint])

            self.symbolic_expressions['d_c_vm'] = jacobian(aggregated_vm_constraint, reshape(vm_input, (vm_input.shape[0] * vm_input.shape[1], 1)))
            self.options['total_stress_constraint']['d_c_vm'] = Function("d_c_vm", [reshape(vm_input, (vm_input.shape[0] * vm_input.shape[1], 1))],
                                                                              [self.symbolic_expressions['d_c_vm']])

            self.add_output('c_vm')

            tau_side_stress_constraint = tau_side_input / self.options['sigmaY'] - 1
            max_tau_side_constraint_space = mmax(tau_side_stress_constraint)
            A_tau_side = sum2(sum1(exp(self.options['rho_KS'] * (tau_side_stress_constraint - max_tau_side_constraint_space))))
            aggregated_tau_side_constraint_positive = max_tau_side_constraint_space + (1 / self.options['rho_KS']) * log(A_tau_side)

            self.symbolic_expressions['c_tau_side'] = aggregated_tau_side_constraint_positive
            self.options['total_stress_constraint']['c_tau_side'] = Function("c_tau_side", [reshape(tau_side_input, (tau_side_input.shape[0] * tau_side_input.shape[1], 1))],
                                                                             [aggregated_tau_side_constraint_positive])

            self.symbolic_expressions['d_c_tau_side'] = jacobian(aggregated_tau_side_constraint_positive, reshape(tau_side_input, (tau_side_input.shape[0] * tau_side_input.shape[1], 1)))
            self.options['total_stress_constraint']['d_c_tau_side'] = Function("d_c_tau_side", [reshape(tau_side_input, (tau_side_input.shape[0] * tau_side_input.shape[1], 1))],
                                                                         [self.symbolic_expressions['d_c_tau_side']])

            self.add_output('c_tau_side')

            tau_side_stress_constraint_n = -tau_side_input / self.options['sigmaY'] - 1
            max_tau_side_constraint_space_n = mmax(tau_side_stress_constraint_n)
            A_tau_side_n = sum2(sum1(exp(self.options['rho_KS'] * (tau_side_stress_constraint_n - max_tau_side_constraint_space_n))))
            aggregated_tau_side_constraint_negative = max_tau_side_constraint_space_n + (1 / self.options['rho_KS']) * log(A_tau_side_n)

            self.symbolic_expressions['c_tau_side_n'] = aggregated_tau_side_constraint_negative
            self.options['total_stress_constraint']['c_tau_side_n'] = Function("c_tau_side_n", [reshape(tau_side_input, (tau_side_input.shape[0] * tau_side_input.shape[1], 1))],
                                                                             [aggregated_tau_side_constraint_negative])

            self.symbolic_expressions['d_c_tau_side_n'] = jacobian(aggregated_tau_side_constraint_negative, reshape(tau_side_input, (tau_side_input.shape[0] * tau_side_input.shape[1], 1)))
            self.options['total_stress_constraint']['d_c_tau_side_n'] = Function("d_c_tau_side_n", [reshape(tau_side_input, (tau_side_input.shape[0] * tau_side_input.shape[1], 1))],
                                                                               [self.symbolic_expressions['d_c_tau_side_n']])

            self.add_output('c_tau_side_n')

            self.declare_partials('c_axial', 'sigma_axial', val=csc_matrix((self.symbolic_expressions['d_c_axial'].shape[0], self.symbolic_expressions['d_c_axial'].shape[1]), dtype=np.float64))
            self.declare_partials('c_axial_n', 'sigma_axial', val=csc_matrix((self.symbolic_expressions['d_c_axial_n'].shape[0], self.symbolic_expressions['d_c_axial_n'].shape[1]), dtype=np.float64))
            self.declare_partials('c_vm', 'sigma_vm', val=csc_matrix((self.symbolic_expressions['d_c_vm'].shape[0], self.symbolic_expressions['d_c_vm'].shape[1]), dtype=np.float64))
            self.declare_partials('c_tau_side', 'tau_side', val=csc_matrix((self.symbolic_expressions['d_c_tau_side'].shape[0], self.symbolic_expressions['d_c_tau_side'].shape[1]), dtype=np.float64))
            self.declare_partials('c_tau_side_n', 'tau_side', val=csc_matrix((self.symbolic_expressions['d_c_tau_side_n'].shape[0], self.symbolic_expressions['d_c_tau_side_n'].shape[1]), dtype=np.float64))
        else:
            raise NotImplementedError("Will add more stress models shortly")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options['debug_flag']:
            if self.options['beam_shape'] == BeamCS.RECTANGULAR:
                axial_in = inputs['sigma_axial']
                stress_constraint = axial_in / self.options['sigmaY'] - 1
                stress_constraint_n = -axial_in / self.options['sigmaY'] - 1
                max_stress_constraint_space = np.max(stress_constraint)
                max_stress_constraint_space_n = np.max(stress_constraint_n)
                A = np.sum(np.sum(np.exp(self.options['rho_KS'] * (stress_constraint - max_stress_constraint_space))))
                A_n = np.sum(np.sum(np.exp(self.options['rho_KS'] * (stress_constraint_n - max_stress_constraint_space_n))))
                aggregated_stress_constraint_positive = max_stress_constraint_space + (1 / self.options['rho_KS']) * np.log(A)
                aggregated_stress_constraint_negative = max_stress_constraint_space_n + (1 / self.options['rho_KS']) * np.log(A_n)
                outputs['c_axial'] = aggregated_stress_constraint_positive
                outputs['c_axial_n'] = aggregated_stress_constraint_negative

                vm_in = inputs['sigma_vm']
                stress_constraint = vm_in / self.options['sigmaY'] - 1
                max_stress_constraint_space = np.max(stress_constraint)
                A = np.sum(np.sum(np.exp(self.options['rho_KS'] * (stress_constraint - max_stress_constraint_space))))
                aggregated_stress_constraint = max_stress_constraint_space + (1 / self.options['rho_KS']) * np.log(A)
                outputs['c_vm'] = aggregated_stress_constraint

                tau_max_c_in = inputs['tau_max_c']
                stress_constraint = tau_max_c_in / self.options['sigmaY'] - 1
                stress_constraint_n = -tau_max_c_in / self.options['sigmaY'] - 1
                max_stress_constraint_space = np.max(stress_constraint)
                max_stress_constraint_space_n = np.max(stress_constraint_n)
                A = np.sum(np.sum(np.exp(self.options['rho_KS'] * (stress_constraint - max_stress_constraint_space))))
                A_n = np.sum(np.sum(np.exp(self.options['rho_KS'] * (stress_constraint_n - max_stress_constraint_space_n))))
                aggregated_stress_constraint_positive = max_stress_constraint_space + (1 / self.options['rho_KS']) * np.log(A)
                aggregated_stress_constraint_negative = max_stress_constraint_space_n + (1 / self.options['rho_KS']) * np.log(A_n)
                outputs['c_tau_max_c'] = aggregated_stress_constraint_positive
                outputs['c_tau_max_c_n'] = aggregated_stress_constraint_negative

                tau_max_n_in = inputs['tau_max_n']
                stress_constraint = tau_max_n_in / self.options['sigmaY'] - 1
                stress_constraint_n = -tau_max_n_in / self.options['sigmaY'] - 1
                max_stress_constraint_space = np.max(stress_constraint)
                max_stress_constraint_space_n = np.max(stress_constraint_n)
                A = np.sum(np.sum(np.exp(self.options['rho_KS'] * (stress_constraint - max_stress_constraint_space))))
                A_n = np.sum(np.sum(np.exp(self.options['rho_KS'] * (stress_constraint_n - max_stress_constraint_space_n))))
                aggregated_stress_constraint_positive = max_stress_constraint_space + (1 / self.options['rho_KS']) * np.log(A)
                aggregated_stress_constraint_negative = max_stress_constraint_space_n + (1 / self.options['rho_KS']) * np.log(A_n)
                outputs['c_tau_max_n'] = aggregated_stress_constraint_positive
                outputs['c_tau_max_n_n'] = aggregated_stress_constraint_negative
                pass
            elif self.options['beam_shape'] == BeamCS.BOX:
                axial_in = inputs['sigma_axial']
                stress_constraint = axial_in / self.options['sigmaY'] - 1
                stress_constraint_n = -axial_in / self.options['sigmaY'] - 1
                max_stress_constraint_space = np.max(stress_constraint)
                max_stress_constraint_space_n = np.max(stress_constraint_n)
                A = np.sum(np.sum(np.exp(self.options['rho_KS'] * (stress_constraint - max_stress_constraint_space))))
                A_n = np.sum(np.sum(np.exp(self.options['rho_KS'] * (stress_constraint_n - max_stress_constraint_space_n))))
                aggregated_stress_constraint_positive = max_stress_constraint_space + (1 / self.options['rho_KS']) * np.log(A)
                aggregated_stress_constraint_negative = max_stress_constraint_space_n + (1 / self.options['rho_KS']) * np.log(A_n)
                outputs['c_axial'] = aggregated_stress_constraint_positive
                outputs['c_axial_n'] = aggregated_stress_constraint_negative

                vm_in = inputs['sigma_vm']
                stress_constraint = vm_in / self.options['sigmaY'] - 1
                max_stress_constraint_space = np.max(stress_constraint)
                A = np.sum(np.sum(np.exp(self.options['rho_KS'] * (stress_constraint - max_stress_constraint_space))))
                aggregated_stress_constraint = max_stress_constraint_space + (1 / self.options['rho_KS']) * np.log(A)
                outputs['c_vm'] = aggregated_stress_constraint

                tau_side_in = inputs['tau_side']
                stress_constraint = tau_side_in / self.options['sigmaY'] - 1
                stress_constraint_n = -tau_side_in / self.options['sigmaY'] - 1
                max_stress_constraint_space = np.max(stress_constraint)
                max_stress_constraint_space_n = np.max(stress_constraint_n)
                A = np.sum(np.sum(np.exp(self.options['rho_KS'] * (stress_constraint - max_stress_constraint_space))))
                A_n = np.sum(np.sum(np.exp(self.options['rho_KS'] * (stress_constraint_n - max_stress_constraint_space_n))))
                aggregated_stress_constraint_positive = max_stress_constraint_space + (1 / self.options['rho_KS']) * np.log(A)
                aggregated_stress_constraint_negative = max_stress_constraint_space_n + (1 / self.options['rho_KS']) * np.log(A_n)
                outputs['c_tau_side'] = aggregated_stress_constraint_positive
                outputs['c_tau_side_n'] = aggregated_stress_constraint_negative
            else:
                raise NotImplementedError("Will add more stress models shortly")
        else:
            if self.options['beam_shape'] == BeamCS.RECTANGULAR:
                outputs['c_axial'] = self.options['total_stress_constraint']['c_axial'](inputs['sigma_axial']).full()
                outputs['c_axial_n'] = self.options['total_stress_constraint']['c_axial_n'](inputs['sigma_axial']).full()
                outputs['c_vm'] = self.options['total_stress_constraint']['c_vm'](inputs['sigma_vm']).full()
                outputs['c_tau_max_c'] = self.options['total_stress_constraint']['c_tau_max_c'](inputs['tau_max_c']).full()
                outputs['c_tau_max_c_n'] = self.options['total_stress_constraint']['c_tau_max_c_n'](inputs['tau_max_c']).full()
                outputs['c_tau_max_n'] = self.options['total_stress_constraint']['c_tau_max_n'](inputs['tau_max_n']).full()
                outputs['c_tau_max_n_n'] = self.options['total_stress_constraint']['c_tau_max_n_n'](inputs['tau_max_n']).full()
            elif self.options['beam_shape'] == BeamCS.BOX:
                outputs['c_axial'] = self.options['total_stress_constraint']['c_axial'](inputs['sigma_axial']).full()
                outputs['c_axial_n'] = self.options['total_stress_constraint']['c_axial_n'](inputs['sigma_axial']).full()
                outputs['c_vm'] = self.options['total_stress_constraint']['c_vm'](inputs['sigma_vm']).full()
                outputs['c_tau_side'] = self.options['total_stress_constraint']['c_tau_side'](inputs['tau_side']).full()
                outputs['c_tau_side_n'] = self.options['total_stress_constraint']['c_tau_side_n'](inputs['tau_side']).full()
            else:
                raise NotImplementedError("Will add more stress models shortly")

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if self.options['beam_shape'] == BeamCS.RECTANGULAR:
            d_c_axial = self.options['total_stress_constraint']['d_c_axial'](inputs['sigma_axial']).sparse()
            d_c_axial_n = self.options['total_stress_constraint']['d_c_axial_n'](inputs['sigma_axial']).sparse()
            d_c_vm = self.options['total_stress_constraint']['d_c_vm'](inputs['sigma_vm']).sparse()
            d_c_tau_max_c = self.options['total_stress_constraint']['d_c_tau_max_c'](inputs['tau_max_c']).sparse()
            d_c_tau_max_c_n = self.options['total_stress_constraint']['d_c_tau_max_c_n'](inputs['tau_max_c']).sparse()
            d_c_tau_max_n = self.options['total_stress_constraint']['d_c_tau_max_n'](inputs['tau_max_n']).sparse()
            d_c_tau_max_n_n = self.options['total_stress_constraint']['d_c_tau_max_n_n'](inputs['tau_max_n']).sparse()

            partials['c_axial', 'sigma_axial'] = d_c_axial
            partials['c_axial_n', 'sigma_axial'] = d_c_axial_n
            partials['c_vm', 'sigma_vm'] = d_c_vm
            partials['c_tau_max_c', 'tau_max_c'] = d_c_tau_max_c
            partials['c_tau_max_c_n', 'tau_max_c'] = d_c_tau_max_c_n
            partials['c_tau_max_n', 'tau_max_n'] = d_c_tau_max_n
            partials['c_tau_max_n_n', 'tau_max_n'] = d_c_tau_max_n_n

        elif self.options['beam_shape'] == BeamCS.BOX:
            d_c_axial = self.options['total_stress_constraint']['d_c_axial'](inputs['sigma_axial']).sparse()
            d_c_axial_n = self.options['total_stress_constraint']['d_c_axial_n'](inputs['sigma_axial']).sparse()
            d_c_vm = self.options['total_stress_constraint']['d_c_vm'](inputs['sigma_vm']).sparse()
            d_c_tau_side = self.options['total_stress_constraint']['d_c_tau_side'](inputs['tau_side']).sparse()
            d_c_tau_side_n = self.options['total_stress_constraint']['d_c_tau_side_n'](inputs['tau_side']).sparse()

            partials['c_axial', 'sigma_axial'] = d_c_axial
            partials['c_axial_n', 'sigma_axial'] = d_c_axial_n
            partials['c_vm', 'sigma_vm'] = d_c_vm
            partials['c_tau_side', 'tau_side'] = d_c_tau_side
            partials['c_tau_side_n', 'tau_side'] = d_c_tau_side_n

        else:
            raise NotImplementedError("Will add more stress models shortly")


class SpecificStrengthAggregatedConstraint(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('name', types=str)  # Just to tag the constraint in particular
        self.options.declare('num_divisions', types=int)  # To generate optional constraint mechanisms
        self.options.declare('num_DvCs', types=int)  # To know the actual number of cross-sectional variables
        self.options.declare('beam_shape', types=BeamCS)  # Important when defining the number of total cs variables
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

        self.add_input('x', shape=(self.stress_input.options['symbolic_variables']['x'].shape[0], self.stress_input.options['symbolic_variables']['x'].shape[1]))
        self.add_input('cs', shape=(self.stress_input.options['symbolic_variables']['cs'].shape[0]))
        self.add_input('corner_points', shape=(self.stress_input.options['symbolic_variables']['corner_points'].shape[0] * self.stress_input.options['symbolic_variables']['corner_points'].shape[1]))

        self.add_output('mass', shape=1)

        self.symbolic_expressions['d_mass'] = jacobian(self.stress_input.options['symbolic_variables']['total_mass'], self.stress_input.options['symbolic_variables']['cs'])

        self.options['total_stress_constraint']['mass'] = Function("mass", [self.stress_input.options['symbolic_variables']['cs']], [self.stress_input.options['symbolic_variables']['total_mass']])
        self.options['total_stress_constraint']['d_mass'] = Function("d_mass", [self.stress_input.options['symbolic_variables']['cs']], [self.symbolic_expressions['d_mass']])

        if self.options['beam_shape'] == BeamCS.RECTANGULAR:

            axial_input = self.stress_input.options['symbolic_expressions']['sigma_axial']
            vm_input = self.stress_input.options['symbolic_expressions']['sigma_vm_w']
            tau_max_c_input = self.stress_input.options['symbolic_expressions']['tau_max_c']
            tau_max_n_input = self.stress_input.options['symbolic_expressions']['tau_max_n']

            axial_stress_constraint = axial_input / self.options['sigmaY'] - 1
            max_axial_constraint_space = mmax(axial_stress_constraint)
            A_axial = sum2(sum1(exp(self.options['rho_KS'] * (axial_stress_constraint - max_axial_constraint_space))))
            aggregated_axial_constraint_positive = max_axial_constraint_space + (1 / self.options['rho_KS']) * log(A_axial)

            self.symbolic_expressions['c_axial'] = aggregated_axial_constraint_positive
            self.options['total_stress_constraint']['c_axial'] = Function("c_axial", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'],
                                                                                      self.stress_input.options['symbolic_variables']['symb_stress_points']], [aggregated_axial_constraint_positive])

            self.symbolic_expressions['d_c_axial'] = jacobian(aggregated_axial_constraint_positive, self.stress_input.options['symbolic_variables']['cs'])
            self.options['total_stress_constraint']['d_c_axial'] = Function("d_c_axial", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']], [self.symbolic_expressions['d_c_axial']])

            self.add_output('c_axial')

            axial_stress_constraint_n = -axial_input / self.options['sigmaY'] - 1
            max_axial_constraint_space_n = mmax(axial_stress_constraint_n)
            A_axial_n = sum2(sum1(exp(self.options['rho_KS'] * (axial_stress_constraint_n - max_axial_constraint_space_n))))
            aggregated_axial_constraint_negative = max_axial_constraint_space_n + (1 / self.options['rho_KS']) * log(A_axial_n)

            self.symbolic_expressions['c_axial_n'] = aggregated_axial_constraint_negative
            self.options['total_stress_constraint']['c_axial_n'] = Function("c_axial_n", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']], [aggregated_axial_constraint_negative])

            self.symbolic_expressions['d_c_axial_n'] = jacobian(aggregated_axial_constraint_negative, self.stress_input.options['symbolic_variables']['cs'])
            self.options['total_stress_constraint']['d_c_axial_n'] = Function("d_c_axial_n", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']],
                                                                            [self.symbolic_expressions['d_c_axial_n']])

            self.add_output('c_axial_n')

            vm_stress_constraint = vm_input / self.options['sigmaY'] - 1
            max_vm_constraint_space = mmax(vm_stress_constraint)
            A_vm = sum2(sum1(exp(self.options['rho_KS'] * (vm_stress_constraint - max_vm_constraint_space))))
            aggregated_vm_constraint = max_vm_constraint_space + (1 / self.options['rho_KS']) * log(A_vm)

            self.symbolic_expressions['c_vm'] = aggregated_vm_constraint
            self.options['total_stress_constraint']['c_vm'] = Function("c_vm", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']], [aggregated_vm_constraint])

            self.symbolic_expressions['d_c_vm'] = jacobian(aggregated_vm_constraint, self.stress_input.options['symbolic_variables']['cs'])
            self.options['total_stress_constraint']['d_c_vm'] = Function("d_c_vm", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']],
                                                                              [self.symbolic_expressions['d_c_vm']])
            self.add_output('c_vm')

            tau_max_c_stress_constraint = tau_max_c_input / self.options['sigmaY'] - 1
            max_tau_max_c_constraint_space = mmax(tau_max_c_stress_constraint)
            A_tau_max_c = sum2(sum1(exp(self.options['rho_KS'] * (tau_max_c_stress_constraint - max_tau_max_c_constraint_space))))
            aggregated_tau_max_c_constraint_positive = max_tau_max_c_constraint_space + (1 / self.options['rho_KS']) * log(A_tau_max_c)

            self.symbolic_expressions['c_tau_max_c'] = aggregated_tau_max_c_constraint_positive
            self.options['total_stress_constraint']['c_tau_max_c'] = Function("c_tau_max_c", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']],
                                                                              [aggregated_tau_max_c_constraint_positive])

            self.symbolic_expressions['d_c_tau_max_c'] = jacobian(aggregated_tau_max_c_constraint_positive, self.stress_input.options['symbolic_variables']['cs'])
            self.options['total_stress_constraint']['d_c_tau_max_c'] = Function("d_c_tau_max_c", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']],
                                                                         [self.symbolic_expressions['d_c_tau_max_c']])
            self.add_output('c_tau_max_c')

            tau_max_c_stress_constraint_n = -tau_max_c_input / self.options['sigmaY'] - 1
            max_tau_max_c_constraint_space_n = mmax(tau_max_c_stress_constraint_n)
            A_tau_max_c_n = sum2(sum1(exp(self.options['rho_KS'] * (tau_max_c_stress_constraint_n - max_tau_max_c_constraint_space_n))))
            aggregated_tau_max_c_constraint_negative = max_tau_max_c_constraint_space_n + (1 / self.options['rho_KS']) * log(A_tau_max_c_n)

            self.symbolic_expressions['c_tau_max_c_n'] = aggregated_tau_max_c_constraint_negative
            self.options['total_stress_constraint']['c_tau_max_c_n'] = Function("c_tau_max_c_n", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']],
                                                                              [aggregated_tau_max_c_constraint_negative])

            self.symbolic_expressions['d_c_tau_max_c_n'] = jacobian(aggregated_tau_max_c_constraint_negative, self.stress_input.options['symbolic_variables']['cs'])
            self.options['total_stress_constraint']['d_c_tau_max_c_n'] = Function("d_c_tau_max_c_n", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']],
                                                                                [self.symbolic_expressions['d_c_tau_max_c_n']])

            self.add_output('c_tau_max_c_n')

            tau_max_n_stress_constraint = tau_max_n_input / self.options['sigmaY'] - 1
            max_tau_max_n_constraint_space = mmax(tau_max_n_stress_constraint)
            A_tau_max_n = sum2(sum1(exp(self.options['rho_KS'] * (tau_max_n_stress_constraint - max_tau_max_n_constraint_space))))
            aggregated_tau_max_n_constraint_positive = max_tau_max_n_constraint_space + (1 / self.options['rho_KS']) * log(A_tau_max_n)

            self.symbolic_expressions['c_tau_max_n'] = aggregated_tau_max_n_constraint_positive
            self.options['total_stress_constraint']['c_tau_max_n'] = Function("c_tau_max_n", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']],
                                                                              [aggregated_tau_max_n_constraint_positive])

            self.symbolic_expressions['d_c_tau_max_n'] = jacobian(aggregated_tau_max_n_constraint_positive, self.stress_input.options['symbolic_variables']['cs'])
            self.options['total_stress_constraint']['d_c_tau_max_n'] = Function("d_c_tau_max_n", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']],
                                                                                  [self.symbolic_expressions['d_c_tau_max_n']])

            self.add_output('c_tau_max_n')

            tau_max_n_stress_constraint_n = -tau_max_n_input / self.options['sigmaY'] - 1
            max_tau_max_n_constraint_space_n = mmax(tau_max_n_stress_constraint_n)
            A_tau_max_n_n = sum2(sum1(exp(self.options['rho_KS'] * (tau_max_n_stress_constraint_n - max_tau_max_n_constraint_space_n))))
            aggregated_tau_max_n_constraint_negative = max_tau_max_n_constraint_space_n + (1 / self.options['rho_KS']) * log(A_tau_max_n_n)

            self.symbolic_expressions['c_tau_max_n_n'] = aggregated_tau_max_n_constraint_negative
            self.options['total_stress_constraint']['c_tau_max_n_n'] = Function("c_tau_max_n_n", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']],
                                                                                [aggregated_tau_max_n_constraint_negative])

            self.symbolic_expressions['d_c_tau_max_n_n'] = jacobian(aggregated_tau_max_n_constraint_negative, self.stress_input.options['symbolic_variables']['cs'])
            self.options['total_stress_constraint']['d_c_tau_max_n_n'] = Function("d_c_tau_max_n_n", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['symb_stress_points']],
                                                                                [self.symbolic_expressions['d_c_tau_max_n_n']])
            self.add_output('c_tau_max_n_n')

            self.declare_partials('c_axial', 'cs')
            self.declare_partials('c_axial_n', 'cs')
            self.declare_partials('c_vm', 'cs')
            self.declare_partials('c_tau_max_c', 'cs')
            self.declare_partials('c_tau_max_c_n', 'cs')
            self.declare_partials('c_tau_max_n', 'cs')
            self.declare_partials('c_tau_max_n_n', 'cs')

        elif self.options['beam_shape'] == BeamCS.BOX:

            axial_input = self.stress_input.options['symbolic_expressions']['sigma_axial']
            vm_input = self.stress_input.options['symbolic_expressions']['sigma_vm']
            tau_side_input = self.stress_input.options['symbolic_expressions']['tau_side']

            axial_stress_constraint = axial_input / self.options['sigmaY'] - 1
            max_axial_constraint_space = mmax(axial_stress_constraint)
            A_axial = sum2(sum1(exp(self.options['rho_KS'] * (axial_stress_constraint - max_axial_constraint_space))))
            aggregated_axial_constraint_positive = max_axial_constraint_space + (1 / self.options['rho_KS']) * log(A_axial)

            self.symbolic_expressions['c_axial'] = aggregated_axial_constraint_positive
            self.options['total_stress_constraint']['c_axial'] = Function("c_axial", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['corner_points']], [aggregated_axial_constraint_positive])

            self.symbolic_expressions['d_c_axial'] = jacobian(aggregated_axial_constraint_positive, self.stress_input.options['symbolic_variables']['cs'])
            self.options['total_stress_constraint']['d_c_axial'] = Function("d_c_axial", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['corner_points']],
                                                                            [self.symbolic_expressions['d_c_axial']])

            self.add_output('c_axial')

            axial_stress_constraint_n = -axial_input / self.options['sigmaY'] - 1
            max_axial_constraint_space_n = mmax(axial_stress_constraint_n)
            A_axial_n = sum2(sum1(exp(self.options['rho_KS'] * (axial_stress_constraint_n - max_axial_constraint_space_n))))
            aggregated_axial_constraint_negative = max_axial_constraint_space_n + (1 / self.options['rho_KS']) * log(A_axial_n)

            self.symbolic_expressions['c_axial_n'] = aggregated_axial_constraint_negative
            self.options['total_stress_constraint']['c_axial_n'] = Function("c_axial_n", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['corner_points']],
                                                                            [aggregated_axial_constraint_negative])

            self.symbolic_expressions['d_c_axial_n'] = jacobian(aggregated_axial_constraint_negative, self.stress_input.options['symbolic_variables']['cs'])
            self.options['total_stress_constraint']['d_c_axial_n'] = Function("d_c_axial_n", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['corner_points']],
                                                                            [self.symbolic_expressions['d_c_axial_n']])

            self.add_output('c_axial_n')

            vm_stress_constraint = vm_input / self.options['sigmaY'] - 1
            max_vm_constraint_space = mmax(vm_stress_constraint)
            A_vm = sum2(sum1(exp(self.options['rho_KS'] * (vm_stress_constraint - max_vm_constraint_space))))
            aggregated_vm_constraint = max_vm_constraint_space + (1 / self.options['rho_KS']) * log(A_vm)

            self.symbolic_expressions['c_vm'] = aggregated_vm_constraint
            self.options['total_stress_constraint']['c_vm'] = Function("c_vm", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['corner_points']], [aggregated_vm_constraint])

            self.symbolic_expressions['d_c_vm'] = jacobian(aggregated_vm_constraint, self.stress_input.options['symbolic_variables']['cs'])
            self.options['total_stress_constraint']['d_c_vm'] = Function("d_c_vm", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['corner_points']],
                                                                              [self.symbolic_expressions['d_c_vm']])

            self.add_output('c_vm')

            tau_side_stress_constraint = tau_side_input / self.options['sigmaY'] - 1
            max_tau_side_constraint_space = mmax(tau_side_stress_constraint)
            A_tau_side = sum2(sum1(exp(self.options['rho_KS'] * (tau_side_stress_constraint - max_tau_side_constraint_space))))
            aggregated_tau_side_constraint_positive = max_tau_side_constraint_space + (1 / self.options['rho_KS']) * log(A_tau_side)

            self.symbolic_expressions['c_tau_side'] = aggregated_tau_side_constraint_positive
            self.options['total_stress_constraint']['c_tau_side'] = Function("c_tau_side", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['corner_points']],
                                                                             [aggregated_tau_side_constraint_positive])

            self.symbolic_expressions['d_c_tau_side'] = jacobian(aggregated_tau_side_constraint_positive, self.stress_input.options['symbolic_variables']['cs'])
            self.options['total_stress_constraint']['d_c_tau_side'] = Function("d_c_tau_side", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['corner_points']],
                                                                         [self.symbolic_expressions['d_c_tau_side']])

            self.add_output('c_tau_side')

            tau_side_stress_constraint_n = -tau_side_input / self.options['sigmaY'] - 1
            max_tau_side_constraint_space_n = mmax(tau_side_stress_constraint_n)
            A_tau_side_n = sum2(sum1(exp(self.options['rho_KS'] * (tau_side_stress_constraint_n - max_tau_side_constraint_space_n))))
            aggregated_tau_side_constraint_negative = max_tau_side_constraint_space_n + (1 / self.options['rho_KS']) * log(A_tau_side_n)

            self.symbolic_expressions['c_tau_side_n'] = aggregated_tau_side_constraint_negative
            self.options['total_stress_constraint']['c_tau_side_n'] = Function("c_tau_side_n", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['corner_points']],
                                                                             [aggregated_tau_side_constraint_negative])

            self.symbolic_expressions['d_c_tau_side_n'] = jacobian(aggregated_tau_side_constraint_negative, self.stress_input.options['symbolic_variables']['cs'])
            self.options['total_stress_constraint']['d_c_tau_side_n'] = Function("d_c_tau_side_n", [self.stress_input.options['symbolic_variables']['x'], self.stress_input.options['symbolic_variables']['cs'], self.stress_input.options['symbolic_variables']['corner_points']],
                                                                               [self.symbolic_expressions['d_c_tau_side_n']])

            self.add_output('c_tau_side_n')

            self.declare_partials('c_axial', 'cs', val=csc_matrix(np.zeros((self.symbolic_expressions['d_c_axial'].shape[0], self.symbolic_expressions['d_c_axial'].shape[1]))))
            self.declare_partials('c_axial_n', 'cs', val=csc_matrix(np.zeros((self.symbolic_expressions['d_c_axial_n'].shape[0], self.symbolic_expressions['d_c_axial_n'].shape[1]))))
            self.declare_partials('c_vm', 'cs', val=csc_matrix(np.zeros((self.symbolic_expressions['d_c_vm'].shape[0], self.symbolic_expressions['d_c_vm'].shape[1]))))
            self.declare_partials('c_tau_side', 'cs', val=csc_matrix(np.zeros((self.symbolic_expressions['d_c_tau_side'].shape[0], self.symbolic_expressions['d_c_tau_side'].shape[1]))))
            self.declare_partials('c_tau_side_n', 'cs', val=csc_matrix(np.zeros((self.symbolic_expressions['d_c_tau_side_n'].shape[0], self.symbolic_expressions['d_c_tau_side_n'].shape[1]))))
        else:
            raise NotImplementedError("Will add more stress models shortly")
        self.declare_partials('mass', 'cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        corner_points = np.reshape(inputs['corner_points'], (self.stress_input.options['symbolic_variables']['corner_points'].shape[0], self.stress_input.options['symbolic_variables']['corner_points'].shape[1]))
        outputs['mass'] = self.options['total_stress_constraint']['mass'](inputs['cs']).full()
        print(inputs['cs'])
        if self.options['beam_shape'] == BeamCS.RECTANGULAR:
            outputs['c_axial'] = self.options['total_stress_constraint']['c_axial'](inputs['x'], inputs['cs'], corner_points).full()
            outputs['c_axial_n'] = self.options['total_stress_constraint']['c_axial_n'](inputs['x'], inputs['cs'], corner_points).full()
            outputs['c_vm'] = self.options['total_stress_constraint']['c_vm'](inputs['x'], inputs['cs'], corner_points).full()
            outputs['c_tau_max_c'] = self.options['total_stress_constraint']['c_tau_max_c'](inputs['x'], inputs['cs'], corner_points).full()
            outputs['c_tau_max_c_n'] = self.options['total_stress_constraint']['c_tau_max_c_n'](inputs['x'], inputs['cs'], corner_points).full()
            outputs['c_tau_max_n'] = self.options['total_stress_constraint']['c_tau_max_n'](inputs['x'], inputs['cs'], corner_points).full()
            outputs['c_tau_max_n_n'] = self.options['total_stress_constraint']['c_tau_max_n_n'](inputs['x'], inputs['cs'], corner_points).full()
        elif self.options['beam_shape'] == BeamCS.BOX:
            outputs['c_axial'] = self.options['total_stress_constraint']['c_axial'](inputs['x'], inputs['cs'], corner_points).full()
            outputs['c_axial_n'] = self.options['total_stress_constraint']['c_axial_n'](inputs['x'], inputs['cs'], corner_points).full()
            outputs['c_vm'] = self.options['total_stress_constraint']['c_vm'](inputs['x'], inputs['cs'], corner_points).full()
            outputs['c_tau_side'] = self.options['total_stress_constraint']['c_tau_side'](inputs['x'], inputs['cs'], corner_points).full()
            outputs['c_tau_side_n'] = self.options['total_stress_constraint']['c_tau_side_n'](inputs['x'], inputs['cs'], corner_points).full()
        else:
            raise NotImplementedError("Will add more stress models shortly")

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        corner_points = np.reshape(inputs['corner_points'], (self.stress_input.options['symbolic_variables']['corner_points'].shape[0], self.stress_input.options['symbolic_variables']['corner_points'].shape[1]))
        partials['mass', 'cs'] = self.options['total_stress_constraint']['d_mass'](inputs['cs']).full()
        if self.options['beam_shape'] == BeamCS.RECTANGULAR:
            d_c_axial = self.options['total_stress_constraint']['d_c_axial'](inputs['x'], inputs['cs'], corner_points).full()
            d_c_axial_n = self.options['total_stress_constraint']['d_c_axial_n'](inputs['x'], inputs['cs'], corner_points).full()
            d_c_vm = self.options['total_stress_constraint']['d_c_vm'](inputs['x'], inputs['cs'], corner_points).full()
            d_c_tau_max_c = self.options['total_stress_constraint']['d_c_tau_max_c'](inputs['x'], inputs['cs'], corner_points).full()
            d_c_tau_max_c_n = self.options['total_stress_constraint']['d_c_tau_max_c_n'](inputs['x'], inputs['cs'], corner_points).full()
            d_c_tau_max_n = self.options['total_stress_constraint']['d_c_tau_max_n'](inputs['x'], inputs['cs'], corner_points).full()
            d_c_tau_max_n_n = self.options['total_stress_constraint']['d_c_tau_max_n_n'](inputs['x'], inputs['cs'], corner_points).full()

            partials['c_axial', 'cs'] = d_c_axial
            partials['c_axial_n', 'cs'] = d_c_axial_n
            partials['c_vm', 'cs'] = d_c_vm
            partials['c_tau_max_c', 'cs'] = d_c_tau_max_c
            partials['c_tau_max_c_n', 'cs'] = d_c_tau_max_c_n
            partials['c_tau_max_n', 'cs'] = d_c_tau_max_n
            partials['c_tau_max_n_n', 'cs'] = d_c_tau_max_n_n

        elif self.options['beam_shape'] == BeamCS.BOX:
            d_c_axial = self.options['total_stress_constraint']['d_c_axial'](inputs['x'], inputs['cs'], corner_points).sparse()
            d_c_axial_n = self.options['total_stress_constraint']['d_c_axial_n'](inputs['x'], inputs['cs'], corner_points).sparse()
            d_c_vm = self.options['total_stress_constraint']['d_c_vm'](inputs['x'], inputs['cs'], corner_points).sparse()
            d_c_tau_side = self.options['total_stress_constraint']['d_c_tau_side'](inputs['x'], inputs['cs'], corner_points).sparse()
            d_c_tau_side_n = self.options['total_stress_constraint']['d_c_tau_side_n'](inputs['x'], inputs['cs'], corner_points).sparse()

            partials['c_axial', 'cs'] = d_c_axial
            partials['c_axial_n', 'cs'] = d_c_axial_n
            partials['c_vm', 'cs'] = d_c_vm
            partials['c_tau_side', 'cs'] = d_c_tau_side
            partials['c_tau_side_n', 'cs'] = d_c_tau_side_n

        else:
            raise NotImplementedError("Will add more stress models shortly")

