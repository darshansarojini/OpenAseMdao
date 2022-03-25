import openmdao.api as om
from casadi import *

from openasemdao.structures.utils.utils import CalcNodalT


class EulerBernoulliStressModel(om.ExplicitComponent):
    def initialize(self):
        # Make sure all the quantities necessary to make the system work are here
        self.options.declare('name', types=str)  # Just to tag the constraint in particular
        self.options.declare('num_divisions', types=int)  # To generate optional constraint mechanisms
        self.options.declare('symbolic_variables', types=dict)  # Where all the resultant cross-section symbolics come from beam parent

        # Attempting an experimental way to store casadi expressions within the component
        self.options.declare('symbolic_expressions', types=dict)
        self.options.declare('symbolic_stress_functions', types=dict)

        self.options.declare('axial_point_list')
        self.options.declare('vm_point_list')
        self.options.declare('shear_point_list')

        self.options['symbolic_variables'] = {}
        self.options['symbolic_expressions'] = {}
        self.options['symbolic_stress_functions'] = {}

    def setup(self):
        # Setup all the nice goodies that make this thing work
        pass

    def stress_formulae_box(self,
                        n, T,
                        number_of_stress_points,
                        t_left, t_top, t_right, t_bot,
                        symb_stress_points,
                        stress_rec_points):
        """
        If T = 0, we are looking at the slice
        If T = 1, static analysis
        If T > 1, dynamic analysis
        """
        sol_x = SX.sym('x_sol', self.options['symbolic_variables']['x'].shape[0], T + 1)

        # region Internal Forces and Moments
        Mc = SX.zeros(n, T + 1)
        Ms = SX.zeros(n, T + 1)
        Mn = SX.zeros(n, T + 1)
        Fc = SX.zeros(n, T + 1)
        Fs = SX.zeros(n, T + 1)
        Fn = SX.zeros(n, T + 1)

        T0, T0a = CalcNodalT(th=self.options['th0'], seq=self.options['seq'], n=n)
        for i in range(n):
            x_node_at_all_timesteps = sol_x[i * 18: (i + 1) * 18, :]
            M_csn = SX.zeros(3, T + 1)
            F_csn = SX.zeros(3, T + 1)
            for j in range(T + 1):
                M_csn[:, j] = mtimes(T0[i], x_node_at_all_timesteps[9:12, j])
                F_csn[:, j] = mtimes(T0[i], x_node_at_all_timesteps[6:9, j])
            Mc[i, :] = M_csn[0, :]
            Ms[i, :] = M_csn[1, :]
            Mn[i, :] = M_csn[2, :]

            Fc[i, :] = F_csn[0, :]
            Fs[i, :] = F_csn[1, :]
            Fn[i, :] = F_csn[2, :]

        sol_x = reshape(sol_x, sol_x.shape[0] * sol_x.shape[1], 1)
        # endregion

        # region Torsional Shear
        cs_ordered = SX.sym('cs_ordered', 4, n)
        cs_ordered[0, :] = (t_top ** (-2) + t_left ** (-2) - (t_top ** (-1)) * t_left ** (-1)) ** (-0.5)
        cs_ordered[1, :] = (t_top ** (-2) + t_right ** (-2) - (t_top ** (-1)) * t_right ** (-1)) ** (-0.5)
        cs_ordered[2, :] = (t_bot ** (-2) + t_right ** (-2) - (t_bot ** (-1)) * t_right ** (-1)) ** (-0.5)
        cs_ordered[3, :] = (t_bot ** (-2) + t_left ** (-2) - (t_bot ** (-1)) * t_left ** (-1)) ** (-0.5)

        sign = np.array([-1, 1, 1, -1])

        tau_torsion = SX.sym('tau_torsion', number_of_stress_points * n, T + 1)

        for j in range(cs_ordered.shape[0]):
            for i in range(n):
                tau_torsion[j * n + i, :] = sign[j] * Ms[i, :] / \
                                                  (2 * self.options['symbolic_expressions']['A_inner'][i] * cs_ordered[j, i])
        if T == 0:
            self.options['symbolic_expressions']['tau_torsion_slice'] = tau_torsion
        else:
            self.options['symbolic_expressions']['tau_torsion'] = tau_torsion
            self.options['symbolic_stress_functions']['tau_torsion'] = Function('tau_torsional',
                                                              [sol_x,
                                                               self.options['symbolic_variables']['h'], self.options['symbolic_variables']['w'],
                                                               self.options['symbolic_variables']['t_left'],
                                                               self.options['symbolic_variables']['t_top'],
                                                               self.options['symbolic_variables']['t_right'],
                                                               self.options['symbolic_variables']['t_bot']],
                                                              [self.options['symbolic_expressions']['tau_torsion']])
        # endregion

        # region Transverse Shear
        cs_ordered = SX.sym('cs_ordered', 4, n)
        cs_ordered[0, :] = t_right + t_left
        cs_ordered[1, :] = t_top + t_bot
        cs_ordered[2, :] = t_right + t_left
        cs_ordered[3, :] = t_top + t_bot
        cs_torsion = SX.sym('cs_torsion', 4, n)
        cs_torsion[0, :] = t_left
        cs_torsion[1, :] = t_top
        cs_torsion[2, :] = t_right
        cs_torsion[3, :] = t_bot
        sign_rl = np.array([1, 0, 1, 0])
        sign_ud = np.array([0, 1, 0, 1])
        sign_t = np.array([-1, 1, 1, -1])

        tau_side = SX.sym('tau_side', number_of_stress_points * n, T + 1)
        tau_shear = SX.sym('tau_shear', number_of_stress_points * n, T + 1)

        for j in range(cs_ordered.shape[0]):
            for i in range(n):
                Icc = self.options['symbolic_expressions']['E'][i][0, 0] / self.options['E']
                Inn = self.options['symbolic_expressions']['E'][i][2, 2] / self.options['E']
                tau_side[j * n + i, :] = \
                    sign_rl[j] * self.options['symbolic_expressions']['Q_max_z'][i] * Fn[i, :] / (Icc * cs_ordered[j, i]) + \
                    sign_ud[j] * self.options['symbolic_expressions']['Q_max_x'][i] * Fc[i, :] / (Inn * cs_ordered[j, i]) + \
                    sign_t[j] * Ms[i, :] / (2 * self.options['symbolic_expressions']['A_inner'][i] * cs_torsion[j, i])
                tau_shear[j * n + i, :] = \
                    sign_rl[j] * self.options['symbolic_expressions']['Q_max_z'][i] * Fn[i, :] / (Icc * cs_ordered[j, i]) + \
                    sign_ud[j] * self.options['symbolic_expressions']['Q_max_x'][i] * Fc[i, :] / (Inn * cs_ordered[j, i])
        if T == 0:
            self.options['symbolic_expressions']['tau_side_slice'] = tau_side
            self.options['symbolic_expressions']['tau_shear_slice'] = tau_shear
            self.options['symbolic_stress_functions']['tau_shear_slice'] = Function('tau_shear_slice',
                                                                  [sol_x,
                                                                   self.options['symbolic_variables']['h'], self.options['symbolic_variables']['w'],
                                                                   self.options['symbolic_variables']['t_left'],
                                                                   self.options['symbolic_variables']['t_top'],
                                                                   self.options['symbolic_variables']['t_right'],
                                                                   self.options['symbolic_variables']['t_bot']],
                                                                  [self.options['symbolic_expressions']['tau_shear_slice']])
            self.options['symbolic_stress_functions']['tau_side_slice'] = Function('tau_side_slice',
                                                                 [sol_x,
                                                                  self.options['symbolic_variables']['h'], self.options['symbolic_variables']['w'],
                                                                  self.options['symbolic_variables']['t_left'],
                                                                  self.options['symbolic_variables']['t_top'],
                                                                  self.options['symbolic_variables']['t_right'],
                                                                  self.options['symbolic_variables']['t_bot']],
                                                                 [self.options['symbolic_expressions']['tau_side_slice']])
        else:
            self.options['symbolic_expressions']['tau_side'] = tau_side
            self.options['symbolic_expressions']['tau_shear'] = tau_shear
            self.options['symbolic_stress_functions']['tau_shear'] = Function('tau_shear',
                                                            [sol_x,
                                                             self.options['symbolic_variables']['h'], self.options['symbolic_variables']['w'],
                                                             self.options['symbolic_variables']['t_left'],
                                                             self.options['symbolic_variables']['t_top'],
                                                             self.options['symbolic_variables']['t_right'],
                                                             self.options['symbolic_variables']['t_bot']],
                                                            [self.options['symbolic_expressions']['tau_shear']])
        # endregion

        # region Axial Stress
        # sigma_axial at a time step is
        #   0:n is 1st point (top-left)
        #   n+1:2n is 2nd point (top-right)
        #   2n+1:3n is 3rd point (bottom-right)
        #   3n+1:4n is 4th point (bottom-left)
        sigma_axial = SX.sym('sigma_a', number_of_stress_points * n, T + 1)
        for j in range(number_of_stress_points):
            for i in range(n):
                x1 = symb_stress_points[2 * j, i]
                x3 = symb_stress_points[2 * j + 1, i]
                EIcc = self.options['symbolic_expressions']['E'][i][0, 0]
                EInn = self.options['symbolic_expressions']['E'][i][2, 2]
                EA = self.options['symbolic_expressions']['EA'][i]
                sigma_axial[j * n + i, :] = self.options['E'] * (Fs[i, :] / EA -
                                                                 x3 * Mc[i, :] / (EIcc) +
                                                                 x1 * Mn[i, :] / (EInn))
                pass
        if T == 0:
            self.options['symbolic_expressions']['sigma_axial_slice'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial_slice'] = Function('sigma_yy_slice',
                                                                    [sol_x,
                                                                     self.options['symbolic_variables']['h'], self.options['symbolic_variables']['w'],
                                                                     self.options['symbolic_variables']['t_left'],
                                                                     self.options['symbolic_variables']['t_top'],
                                                                     self.options['symbolic_variables']['t_right'],
                                                                     self.options['symbolic_variables']['t_bot'],
                                                                     symb_stress_points],
                                                                    [self.options['symbolic_expressions']['sigma_axial_slice']])
        else:
            self.options['symbolic_expressions']['sigma_axial'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial'] = Function('sigma_yy',
                                                              [sol_x,
                                                               self.options['symbolic_variables']['h'], self.options['symbolic_variables']['w'],
                                                               self.options['symbolic_variables']['t_left'],
                                                               self.options['symbolic_variables']['t_top'],
                                                               self.options['symbolic_variables']['t_right'],
                                                               self.options['symbolic_variables']['t_bot'],
                                                               symb_stress_points],
                                                              [self.options['symbolic_expressions']['sigma_axial']])
        # endregion

        # region von-Mises Stress
        sigma = SX.sym('sigma', number_of_stress_points * n, T + 1)
        for j in range(number_of_stress_points):
            for i in range(n):
                sign_1 = 1
                sign_3 = 1
                t_3_selected = t_top[i] / 2
                t_1_selected = t_left[i] / 2
                x1 = symb_stress_points[2 * j, i]
                x3 = symb_stress_points[2 * j + 1, i]
                if stress_rec_points[2 * j, i] < 0:
                    sign_1 = -sign_1
                    t_1_selected = t_right[i] / 2
                if stress_rec_points[2 * j + 1, i] < 0:
                    sign_3 = -sign_3
                    t_3_selected = t_bot[i]
                EIcc = self.options['symbolic_expressions']['E'][i][0, 0]
                EInn = self.options['symbolic_expressions']['E'][i][2, 2]
                EA = self.options['symbolic_expressions']['EA'][i]
                sigma[j * n + i, :] = self.options['E'] * (Fs[i, :] / EA -
                                                                 (x3 - sign_3 * t_3_selected) * Mc[i, :] / (EIcc) +
                                                                 (x1 - sign_1 * t_1_selected) * Mn[i, :] / (EInn))


        sigma_vm = (sigma ** 2 + tau_torsion ** 2 - sigma * tau_torsion + 1e-15) ** 0.5
        if T == 0:
            self.options['symbolic_expressions']['sigma_vm_slice'] = sigma_vm
            self.options['symbolic_stress_functions']['sigma_vm_slice'] = Function('sigma_vm_slice',
                                                                 [sol_x,
                                                                  self.options['symbolic_variables']['h'], self.options['symbolic_variables']['w'],
                                                                  self.options['symbolic_variables']['t_left'],
                                                                  self.options['symbolic_variables']['t_top'],
                                                                  self.options['symbolic_variables']['t_right'],
                                                                  self.options['symbolic_variables']['t_bot'],
                                                                  symb_stress_points],
                                                                 [self.options['symbolic_expressions']['sigma_vm_slice']])
        else:
            self.options['symbolic_expressions']['sigma_vm'] = sigma_vm
            self.options['symbolic_stress_functions']['sigma_vm'] = Function('sigma_vm',
                                                           [sol_x,
                                                            self.options['symbolic_variables']['h'], self.options['symbolic_variables']['w'],
                                                            self.options['symbolic_variables']['t_left'],
                                                            self.options['symbolic_variables']['t_top'],
                                                            self.options['symbolic_variables']['t_right'],
                                                            self.options['symbolic_variables']['t_bot'],
                                                            symb_stress_points],
                                                           [self.options['symbolic_expressions']['sigma_vm']])

        sigma = horzcat(sigma_vm, sigma_axial, tau_side)
        if T == 0:
            pass
        else:
            self.options['symbolic_expressions']['sigma'] = sigma
            self.options['symbolic_stress_functions']['sigma'] = Function('sigma',
                                                        [sol_x,
                                                         self.options['symbolic_variables']['h'], self.options['symbolic_variables']['w'],
                                                         self.options['symbolic_variables']['t_left'],
                                                         self.options['symbolic_variables']['t_top'],
                                                         self.options['symbolic_variables']['t_right'],
                                                         self.options['symbolic_variables']['t_bot'],
                                                         symb_stress_points],
                                                        [self.options['symbolic_expressions']['sigma']])
        # endregion

        return sol_x