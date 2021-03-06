import openmdao.api as om
from casadi import *

from openasemdao.structures.utils.utils import CalcNodalT
from openasemdao.structures.utils.beam_categories import BeamCS
from scipy.sparse import csc_matrix


class EulerBernoulliStressModel(om.ExplicitComponent):
    def initialize(self):
        # Make sure all the quantities necessary to make the system work are here
        self.options.declare('name', types=str)  # Just to tag the constraint in particular
        self.options.declare('debug_flag', types=bool, default=False)  # To enable or disable debugging
        self.options.declare('num_divisions', types=int)  # To generate optional constraint mechanisms
        self.options.declare('num_DvCs', types=int)  # To know the actual number of cross-sectional variables
        self.options.declare('symbolic_variables',
                             types=dict)  # Where all the resultant cross-section symbolics come from beam parent
        self.options.declare('beam_shape', types=BeamCS)  # To check what beam type is coming through
        # Where helper and needed symbolic expressions (Einv) and values (x) are stored
        self.options.declare('symbolic_expressions', types=dict)
        self.options.declare('symbolic_stress_functions', types=dict)

        self.options.declare('corner_points')
        self.options.declare('num_timesteps')

        # Necessary Beam Characteristics
        self.options.declare('r0')
        self.options.declare('th0')
        self.options.declare('seq')
        self.options.declare('E')
        self.options.declare('G')

        self.options['symbolic_variables'] = {}
        self.options['symbolic_expressions'] = {}
        self.options['symbolic_stress_functions'] = {}

    def setup(self):
        # Setup all the nice goodies that make this thing work
        self.add_input('cs', shape=self.options['beam_shape'].value * self.options['num_DvCs'])
        self.add_input('x', shape=(18 * self.options['num_divisions'], self.options['num_timesteps']))
        # Time to generate the stress formulas:
        cs = self.options['symbolic_variables']['cs']
        if self.options['beam_shape'] == BeamCS.RECTANGULAR:  # Rectangular beam with 2 degrees of freedom per cross-section h w
            self.options['corner_points'] = np.zeros((self.options['symbolic_variables']['corner_points'].shape[0], self.options['symbolic_variables']['corner_points'].shape[1]))
            self.add_input('corner_points', shape=(self.options['symbolic_variables']['corner_points'].shape[0] * self.options['symbolic_variables']['corner_points'].shape[1]))

            """
                    The following are the stresses modeled in the rectangular beam:
                    T x 0n -> 4n : Axial Stresses at the corners
                    T x 4n -> 8n : Von Misses Stress at the edges (center of edge)
                    T x 8n -> 9n : Shear stress at the horizontal beam direction
                    T x 9n -> 10n: Shear stress at the vertical beam direction
            """
            self.stress_formulae_rect(self.options['num_divisions'], self.options['num_timesteps'], cs)

            self.add_output('sigma_axial', shape=(4 * self.options['num_divisions'] * (self.options['num_timesteps'])))
            self.add_output('sigma_vm', shape=(4 * self.options['num_divisions'] * (self.options['num_timesteps'])))
            self.add_output('tau_max_c', shape=(self.options['num_divisions'] * (self.options['num_timesteps'])))
            self.add_output('tau_max_n', shape=(self.options['num_divisions'] * (self.options['num_timesteps'])))

            self.declare_partials('sigma_axial', 'cs', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['d_sigma_axial'].shape[0], self.options['symbolic_expressions']['d_sigma_axial'].shape[1]))))
            self.declare_partials('sigma_vm', 'cs', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['d_sigma_vm_w'].shape[0], self.options['symbolic_expressions']['d_sigma_vm_w'].shape[1]))))
            self.declare_partials('tau_max_c', 'cs', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['d_tau_max_c'].shape[0], self.options['symbolic_expressions']['d_tau_max_c'].shape[1]))))
            self.declare_partials('tau_max_n', 'cs', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['d_tau_max_n'].shape[0], self.options['symbolic_expressions']['d_tau_max_n'].shape[1]))))

            self.declare_partials('sigma_axial', 'corner_points', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dc_sigma_axial'].shape[0], self.options['symbolic_expressions']['dc_sigma_axial'].shape[1]))))
            self.declare_partials('sigma_vm', 'corner_points', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dc_sigma_vm_w'].shape[0], self.options['symbolic_expressions']['dc_sigma_vm_w'].shape[1]))))
            self.declare_partials('tau_max_c', 'corner_points', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dc_tau_max_c'].shape[0], self.options['symbolic_expressions']['dc_tau_max_c'].shape[1]))))
            self.declare_partials('tau_max_n', 'corner_points', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dc_tau_max_n'].shape[0], self.options['symbolic_expressions']['dc_tau_max_n'].shape[1]))))

            self.declare_partials('sigma_axial', 'x', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dx_sigma_axial'].shape[0], self.options['symbolic_expressions']['dx_sigma_axial'].shape[1]))))
            self.declare_partials('sigma_vm', 'x', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dx_sigma_vm_w'].shape[0], self.options['symbolic_expressions']['dx_sigma_vm_w'].shape[1]))))
            self.declare_partials('tau_max_c', 'x', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dx_tau_max_c'].shape[0], self.options['symbolic_expressions']['dx_tau_max_c'].shape[1]))))
            self.declare_partials('tau_max_n', 'x', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dx_tau_max_n'].shape[0], self.options['symbolic_expressions']['dx_tau_max_n'].shape[1]))))


        if self.options['beam_shape'] == BeamCS.BOX:
            self.options['corner_points'] = np.zeros((self.options['symbolic_variables']['corner_points'].shape[0],
                                                      self.options['symbolic_variables']['corner_points'].shape[1]))
            self.add_input('corner_points', shape=(self.options['symbolic_variables']['corner_points'].shape[0] * self.options['symbolic_variables']['corner_points'].shape[1]))

            """
                                The following are the stresses modeled in the rectangular beam:
                                T x 0n -> 4n : von-mises stress at the corners
                                T x 4n -> 8n : axial stress at the corners
                                T x 8n -> 12n : Shear stress at the flange centers
            """
            self.stress_formulae_box(self.options['num_divisions'], self.options['num_timesteps'], cs)

            self.add_output('sigma_axial', shape=(4 * self.options['num_divisions'] * (self.options['num_timesteps'])))
            self.add_output('sigma_vm', shape=(4 * self.options['num_divisions'] * (self.options['num_timesteps'])))
            self.add_output('tau_side', shape=(4 * self.options['num_divisions'] * (self.options['num_timesteps'])))

            self.declare_partials('sigma_axial', 'cs', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['d_sigma_axial'].shape[0], self.options['symbolic_expressions']['d_sigma_axial'].shape[1]))))
            self.declare_partials('sigma_vm', 'cs', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['d_sigma_vm'].shape[0], self.options['symbolic_expressions']['d_sigma_vm'].shape[1]))))
            self.declare_partials('tau_side', 'cs', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['d_tau_side'].shape[0], self.options['symbolic_expressions']['d_tau_side'].shape[1]))))

            self.declare_partials('sigma_axial', 'corner_points', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dc_sigma_axial'].shape[0], self.options['symbolic_expressions']['dc_sigma_axial'].shape[1]))))
            self.declare_partials('sigma_vm', 'corner_points', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dc_sigma_vm'].shape[0], self.options['symbolic_expressions']['dc_sigma_vm'].shape[1]))))
            self.declare_partials('tau_side', 'corner_points', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dc_tau_side'].shape[0], self.options['symbolic_expressions']['dc_tau_side'].shape[1]))))

            self.declare_partials('sigma_axial', 'x', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dx_sigma_axial'].shape[0], self.options['symbolic_expressions']['dx_sigma_axial'].shape[1]))))
            self.declare_partials('sigma_vm', 'x', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dx_sigma_vm'].shape[0], self.options['symbolic_expressions']['dx_sigma_vm'].shape[1]))))
            self.declare_partials('tau_side', 'x', val=csc_matrix(np.zeros((self.options['symbolic_expressions']['dx_tau_side'].shape[0], self.options['symbolic_expressions']['dx_tau_side'].shape[1]))))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        corner_points = np.reshape(inputs['corner_points'], (self.options['symbolic_variables']['corner_points'].shape[0], self.options['symbolic_variables']['corner_points'].shape[1]))
        if self.options['debug_flag']:
            # Point 1
            assert (corner_points[0, :] < 0).all()  # x coordinate must be negative
            assert (corner_points[1, :] > 0).all()  # y coordinate must be positive
            # Point 2
            assert (corner_points[2, :] > 0).all()  # x coordinate must be positive
            assert (corner_points[3, :] > 0).all()  # y coordinate must be positive
            # Point 3
            assert (corner_points[4, :] > 0).all()  # x coordinate must be positive
            assert (corner_points[5, :] < 0).all()  # y coordinate must be negative
            # Point 4
            assert (corner_points[6, :] < 0).all()  # x coordinate must be negative
            assert (corner_points[7, :] < 0).all()  # y coordinate must be negative

        if self.options['beam_shape'] == BeamCS.RECTANGULAR:
            h = inputs['cs'][0:self.options['num_DvCs']]
            w = inputs['cs'][self.options['num_DvCs']:2 * self.options['num_DvCs']]

            sigma_axial = self.options['symbolic_stress_functions']['sigma_axial'](inputs['x'], inputs['cs'], corner_points).full()
            tau_max_c = self.options['symbolic_stress_functions']['tau_max_c'](inputs['x'], inputs['cs'], corner_points).full()
            tau_max_n = self.options['symbolic_stress_functions']['tau_max_n'](inputs['x'], inputs['cs'], corner_points).full()

            if np.linalg.norm(h) < np.linalg.norm(w):
                sigma_vm = self.options['symbolic_stress_functions']['sigma_vm_w'](inputs['x'], inputs['cs'], corner_points).full()
            else:
                sigma_vm = self.options['symbolic_stress_functions']['sigma_vm_h'](inputs['x'], inputs['cs'], corner_points).full()

            outputs['sigma_axial'] = sigma_axial
            outputs['sigma_vm'] = sigma_vm
            outputs['tau_max_c'] = tau_max_c
            outputs['tau_max_n'] = tau_max_n
            pass
        if self.options['beam_shape'] == BeamCS.BOX:
            sigma_axial = self.options['symbolic_stress_functions']['sigma_axial'](inputs['x'], inputs['cs'], corner_points).full()
            sigma_vm = self.options['symbolic_stress_functions']['sigma_vm'](inputs['x'], inputs['cs'], corner_points).full()
            tau_side = self.options['symbolic_stress_functions']['tau_side'](inputs['x'], inputs['cs'], corner_points).full()

            outputs['sigma_axial'] = sigma_axial
            outputs['sigma_vm'] = sigma_vm
            outputs['tau_side'] = tau_side

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        corner_points = np.reshape(inputs['corner_points'], (self.options['symbolic_variables']['corner_points'].shape[0], self.options['symbolic_variables']['corner_points'].shape[1]))
        if self.options['beam_shape'] == BeamCS.RECTANGULAR:
            h = inputs['cs'][0:self.options['num_DvCs']]
            w = inputs['cs'][self.options['num_DvCs']:2 * self.options['num_DvCs']]

            if np.linalg.norm(h) < np.linalg.norm(w):
                d_sigma_vm = self.options['symbolic_stress_functions']['d_sigma_vm_w'](inputs['x'], inputs['cs'], corner_points).sparse()
                dc_sigma_vm = self.options['symbolic_stress_functions']['dc_sigma_vm_w'](inputs['x'], inputs['cs'], corner_points).sparse()
                dx_sigma_vm = self.options['symbolic_stress_functions']['dx_sigma_vm_w'](inputs['x'], inputs['cs'], corner_points).sparse()
            else:
                d_sigma_vm = self.options['symbolic_stress_functions']['d_sigma_vm_h'](inputs['x'], inputs['cs'], corner_points).sparse()
                dc_sigma_vm = self.options['symbolic_stress_functions']['dc_sigma_vm_h'](inputs['x'], inputs['cs'], corner_points).sparse()
                dx_sigma_vm = self.options['symbolic_stress_functions']['dx_sigma_vm_h'](inputs['x'], inputs['cs'], corner_points).sparse()

            d_sigma_axial = self.options['symbolic_stress_functions']['d_sigma_axial'](inputs['x'], inputs['cs'], corner_points).sparse()
            d_tau_max_c = self.options['symbolic_stress_functions']['d_tau_max_c'](inputs['x'], inputs['cs'], corner_points).sparse()
            d_tau_max_n = self.options['symbolic_stress_functions']['d_tau_max_n'](inputs['x'], inputs['cs'], corner_points).sparse()

            dc_sigma_axial = self.options['symbolic_stress_functions']['dc_sigma_axial'](inputs['x'], inputs['cs'], corner_points).sparse()
            dc_tau_max_c = self.options['symbolic_stress_functions']['dc_tau_max_c'](inputs['x'], inputs['cs'], corner_points).sparse()
            dc_tau_max_n = self.options['symbolic_stress_functions']['dc_tau_max_n'](inputs['x'], inputs['cs'], corner_points).sparse()

            dx_sigma_axial = self.options['symbolic_stress_functions']['dx_sigma_axial'](inputs['x'], inputs['cs'], corner_points).sparse()
            dx_tau_max_c = self.options['symbolic_stress_functions']['dx_tau_max_c'](inputs['x'], inputs['cs'], corner_points).sparse()
            dx_tau_max_n = self.options['symbolic_stress_functions']['dx_tau_max_n'](inputs['x'], inputs['cs'], corner_points).sparse()

            partials['sigma_axial', 'cs'] = d_sigma_axial
            partials['sigma_vm', 'cs'] = d_sigma_vm
            partials['tau_max_c', 'cs'] = d_tau_max_c
            partials['tau_max_n', 'cs'] = d_tau_max_n

            partials['sigma_axial', 'corner_points'] = dc_sigma_axial
            partials['sigma_vm', 'corner_points'] = dc_sigma_vm
            partials['tau_max_c', 'corner_points'] = dc_tau_max_c
            partials['tau_max_n', 'corner_points'] = dc_tau_max_n

            partials['sigma_axial', 'x'] = dx_sigma_axial
            partials['sigma_vm', 'x'] = dx_sigma_vm
            partials['tau_max_c', 'x'] = dx_tau_max_c
            partials['tau_max_n', 'x'] = dx_tau_max_n

        elif self.options['beam_shape'] == BeamCS.BOX:
            d_sigma_axial = self.options['symbolic_stress_functions']['d_sigma_axial'](inputs['x'], inputs['cs'], corner_points).sparse()
            d_sigma_vm = self.options['symbolic_stress_functions']['d_sigma_vm'](inputs['x'], inputs['cs'], corner_points).sparse()
            d_tau_side = self.options['symbolic_stress_functions']['d_tau_side'](inputs['x'], inputs['cs'], corner_points).sparse()

            dc_sigma_axial = self.options['symbolic_stress_functions']['dc_sigma_axial'](inputs['x'], inputs['cs'], corner_points).sparse()
            dc_sigma_vm = self.options['symbolic_stress_functions']['dc_sigma_vm'](inputs['x'], inputs['cs'], corner_points).sparse()
            dc_tau_side = self.options['symbolic_stress_functions']['dc_tau_side'](inputs['x'], inputs['cs'], corner_points).sparse()

            dx_sigma_axial = self.options['symbolic_stress_functions']['dx_sigma_axial'](inputs['x'], inputs['cs'], corner_points).sparse()
            dx_sigma_vm = self.options['symbolic_stress_functions']['dx_sigma_vm'](inputs['x'], inputs['cs'], corner_points).sparse()
            dx_tau_side = self.options['symbolic_stress_functions']['dx_tau_side'](inputs['x'], inputs['cs'], corner_points).sparse()

            partials['sigma_axial', 'cs'] = d_sigma_axial
            partials['sigma_vm', 'cs'] = d_sigma_vm
            partials['tau_side', 'cs'] = d_tau_side

            partials['sigma_axial', 'corner_points'] = dc_sigma_axial
            partials['sigma_vm', 'corner_points'] = dc_sigma_vm
            partials['tau_side', 'corner_points'] = dc_tau_side

            partials['sigma_axial', 'x'] = dx_sigma_axial
            partials['sigma_vm', 'x'] = dx_sigma_vm
            partials['tau_side', 'x'] = dx_tau_side

    def stress_formulae_rect(self, n, T, cs):
        """
        If T = 0, we are looking at the slice
        If T = 1, static analysis
        If T > 1, dynamic analysis
        """
        stress_rec_points = self.options['corner_points']
        assert stress_rec_points.shape[1] == n
        assert int(stress_rec_points.shape[0] / 2) == 4, \
            'Implementation right now assumes 4 points at the 4 corners of the box CS'
        number_of_stress_points = int(stress_rec_points.shape[0] / 2)
        """
        The points must be ordered in the following format
            1 ------------------------------------- 2
              -                y                  -
              -                |                  -
              -                --> x              -
              -                                   -
              -                                   -
            4 ------------------------------------- 3
        """

        symb_stress_points = SX.sym('stress_pts', stress_rec_points.shape[1], stress_rec_points.shape[0]).T

        h = self.options['symbolic_variables']['h']
        w = self.options['symbolic_variables']['w']

        sol_x = SX.sym('x_sol', 18 * self.options['num_divisions'], T)

        # region Internal Forces and Moments
        Mc = SX.zeros(n, T)
        Ms = SX.zeros(n, T)
        Mn = SX.zeros(n, T)
        Fc = SX.zeros(n, T)
        Fs = SX.zeros(n, T)
        Fn = SX.zeros(n, T)

        T0, T0a = CalcNodalT(th=self.options['th0'], seq=self.options['seq'], n=n)
        for i in range(n):
            x_node_at_all_timesteps = sol_x[i * 18: (i + 1) * 18, :]
            M_csn = SX.zeros(3, T)
            F_csn = SX.zeros(3, T)
            for j in range(T):
                M_csn[:, j] = mtimes(T0[i], x_node_at_all_timesteps[9:12, j])
                F_csn[:, j] = mtimes(T0[i], x_node_at_all_timesteps[6:9, j])
            Mc[i, :] = M_csn[0, :]
            Ms[i, :] = M_csn[1, :]
            Mn[i, :] = M_csn[2, :]

            Fc[i, :] = F_csn[0, :]
            Fs[i, :] = F_csn[1, :]
            Fn[i, :] = F_csn[2, :]

        # endregion

        # Torsion calculation on a solid cross-section follows Roark's formula for Stress:

        # When w > h:
        tau_torsion_w_slice = ((3 * Ms) / (w * h ** 2)) * (1 + 0.6095 * (h / w) + 0.8865 * (h / w) ** 2 - 1.8023 * (h / w) ** 3 + 0.91 * (h / w) ** 4)

        # When h > w:
        tau_torsion_h_slice = ((3 * Ms) / (h * w ** 2)) * (1 + 0.6095 * (w / h) + 0.8865 * (w / h) ** 2 - 1.8023 * (w / h) ** 3 + 0.91 * (w / h) ** 4)

        # Sign convention to make sure torsion goes in the convention chosen in the csn axes:
        sign = np.array([-1, 1, 1, -1])

        tau_torsion_w = vertcat(sign[0] * tau_torsion_w_slice, sign[1] * tau_torsion_w_slice,
                                sign[2] * tau_torsion_w_slice, sign[3] * tau_torsion_w_slice)

        tau_torsion_h = vertcat(sign[0] * tau_torsion_h_slice, sign[1] * tau_torsion_h_slice,
                                sign[2] * tau_torsion_h_slice, sign[3] * tau_torsion_h_slice)
        if T == 0:
            self.options['symbolic_expressions']['tau_torsion_slice_w'] = tau_torsion_w
            self.options['symbolic_expressions']['tau_torsion_slice_h'] = tau_torsion_h
        else:
            self.options['symbolic_expressions']['tau_torsion_w'] = tau_torsion_w
            self.options['symbolic_expressions']['tau_torsion_h'] = tau_torsion_h
            self.options['symbolic_stress_functions']['tau_torsion_w'] = Function('tau_torsional_w', [sol_x, cs], [self.options['symbolic_expressions']['tau_torsion_w']])
            self.options['symbolic_stress_functions']['tau_torsion_h'] = Function('tau_torsional_h', [sol_x, cs], [self.options['symbolic_expressions']['tau_torsion_h']])
        # endregion

        # region Transverse Shear
        tau_max_c = 1.5 * Fc / (h * w)
        tau_max_n = 1.5 * Fn / (h * w)

        self.options['symbolic_expressions']['tau_max_c_slice'] = tau_max_c
        self.options['symbolic_expressions']['tau_max_n_slice'] = tau_max_n

        if T == 0:
            self.options['symbolic_expressions']['tau_shear_slice'] = vertcat(tau_max_c, tau_max_n)
            self.options['symbolic_stress_functions']['tau_shear_slice'] = Function('tau_shear_slice', [sol_x, cs], [self.options['symbolic_expressions']['tau_shear_slice']])
        else:
            self.options['symbolic_expressions']['tau_shear'] = vertcat(tau_max_c, tau_max_n)
            self.options['symbolic_stress_functions']['tau_shear'] = Function('tau_shear', [sol_x, cs], [self.options['symbolic_expressions']['tau_shear']])
        # endregion

        # region Axial Stress
        # sigma_axial at a time step is
        #   0:n is 1st point (top-left)
        #   n+1:2n is 2nd point (top-right)
        #   2n+1:3n is 3rd point (bottom-right)
        #   3n+1:4n is 4th point (bottom-left)
        sigma_axial = SX.sym('sigma_a', number_of_stress_points * n, T)
        for j in range(number_of_stress_points):
            for i in range(n):
                x1 = symb_stress_points[2 * j, i]
                x3 = symb_stress_points[2 * j + 1, i]
                EIcc = self.options['symbolic_variables']['E'][i][0, 0]
                EInn = self.options['symbolic_variables']['E'][i][2, 2]
                EA = self.options['symbolic_variables']['EA'][i]
                sigma_axial[j * n + i, :] = self.options['E'] * (Fs[i, :] / EA - x3 * Mc[i, :] / (EIcc) + x1 * Mn[i, :] / (EInn))
                pass
        if T == 0:
            self.options['symbolic_expressions']['sigma_axial_slice'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial_slice'] = Function('sigma_yy_slice', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['sigma_axial_slice']])
        else:
            self.options['symbolic_expressions']['sigma_axial'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial'] = Function('sigma_yy', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['sigma_axial']])
        # endregion

        # region von-Mises Stress

        sigma_vm_w = (sigma_axial ** 2 + tau_torsion_w ** 2 - sigma_axial * tau_torsion_w + 1e-15) ** 0.5

        sigma_vm_h = (sigma_axial ** 2 + tau_torsion_h ** 2 - sigma_axial * tau_torsion_h + 1e-15) ** 0.5

        if T == 0:
            self.options['symbolic_expressions']['sigma_vm_slice_w'] = sigma_vm_w
            self.options['symbolic_stress_functions']['sigma_vm_slice_w'] = Function('sigma_vm_slice_w', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['sigma_vm_slice_w']])
            self.options['symbolic_expressions']['sigma_vm_slice_h'] = sigma_vm_h
            self.options['symbolic_stress_functions']['sigma_vm_slice_h'] = Function('sigma_vm_slice_h', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['sigma_vm_slice_h']])
        else:
            self.options['symbolic_expressions']['sigma_vm_w'] = sigma_vm_w
            self.options['symbolic_stress_functions']['sigma_vm_w'] = Function('sigma_vm_w', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['sigma_vm_w']])
            self.options['symbolic_expressions']['sigma_vm_h'] = sigma_vm_h
            self.options['symbolic_stress_functions']['sigma_vm_h'] = Function('sigma_vm_h', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['sigma_vm_h']])
        # endregion

        # Region Stress Function Buildup

        self.options['symbolic_expressions']['d_sigma_axial'] = jacobian(reshape(sigma_axial, (sigma_axial.shape[0] * sigma_axial.shape[1], 1)), cs)
        self.options['symbolic_expressions']['d_sigma_vm_w'] = jacobian(reshape(sigma_vm_w, (sigma_vm_w.shape[0] * sigma_vm_w.shape[1], 1)), cs)
        self.options['symbolic_expressions']['d_sigma_vm_h'] = jacobian(reshape(sigma_vm_h, (sigma_vm_h.shape[0] * sigma_vm_h.shape[1], 1)), cs)
        self.options['symbolic_expressions']['d_tau_max_c'] = jacobian(reshape(tau_max_c, (tau_max_c.shape[0] * tau_max_c.shape[1], 1)), cs)
        self.options['symbolic_expressions']['d_tau_max_n'] = jacobian(reshape(tau_max_n, (tau_max_n.shape[0] * tau_max_n.shape[1], 1)), cs)

        self.options['symbolic_expressions']['dc_sigma_axial'] = jacobian(reshape(sigma_axial, (sigma_axial.shape[0] * sigma_axial.shape[1], 1)), transpose(symb_stress_points))
        self.options['symbolic_expressions']['dc_sigma_vm_w'] = jacobian(reshape(sigma_vm_w, (sigma_vm_w.shape[0] * sigma_vm_w.shape[1], 1)), transpose(symb_stress_points))
        self.options['symbolic_expressions']['dc_sigma_vm_h'] = jacobian(reshape(sigma_vm_h, (sigma_vm_h.shape[0] * sigma_vm_h.shape[1], 1)), transpose(symb_stress_points))
        self.options['symbolic_expressions']['dc_tau_max_c'] = jacobian(reshape(tau_max_c, (tau_max_c.shape[0] * tau_max_c.shape[1], 1)), transpose(symb_stress_points))
        self.options['symbolic_expressions']['dc_tau_max_n'] = jacobian(reshape(tau_max_n, (tau_max_n.shape[0] * tau_max_n.shape[1], 1)), transpose(symb_stress_points))

        self.options['symbolic_expressions']['dx_sigma_axial'] = jacobian(reshape(sigma_axial, (sigma_axial.shape[0] * sigma_axial.shape[1], 1)), transpose(sol_x))
        self.options['symbolic_expressions']['dx_sigma_vm_w'] = jacobian(reshape(sigma_vm_w, (sigma_vm_w.shape[0] * sigma_vm_w.shape[1], 1)), transpose(sol_x))
        self.options['symbolic_expressions']['dx_sigma_vm_h'] = jacobian(reshape(sigma_vm_h, (sigma_vm_h.shape[0] * sigma_vm_h.shape[1], 1)), transpose(sol_x))
        self.options['symbolic_expressions']['dx_tau_max_c'] = jacobian(reshape(tau_max_c, (tau_max_c.shape[0] * tau_max_c.shape[1], 1)), transpose(sol_x))
        self.options['symbolic_expressions']['dx_tau_max_n'] = jacobian(reshape(tau_max_n, (tau_max_n.shape[0] * tau_max_n.shape[1], 1)), transpose(sol_x))

        if T == 0:
            pass
        else:
            self.options['symbolic_expressions']['sigma_axial'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial'] = Function('sigma_axial', [sol_x, cs, symb_stress_points], [reshape(sigma_axial, (sigma_axial.shape[0] * sigma_axial.shape[1], 1))])
            self.options['symbolic_expressions']['sigma_vm_w'] = sigma_vm_w
            self.options['symbolic_stress_functions']['sigma_vm_w'] = Function('sigma_vm_w', [sol_x, cs, symb_stress_points], [reshape(sigma_vm_w, (sigma_vm_w.shape[0] * sigma_vm_w.shape[1], 1))])
            self.options['symbolic_expressions']['sigma_vm_h'] = sigma_vm_h
            self.options['symbolic_stress_functions']['sigma_vm_h'] = Function('sigma_vm_h', [sol_x, cs, symb_stress_points], [reshape(sigma_vm_h, (sigma_vm_h.shape[0] * sigma_vm_h.shape[1], 1))])
            self.options['symbolic_expressions']['tau_max_c'] = tau_max_c
            self.options['symbolic_stress_functions']['tau_max_c'] = Function('tau_max_c', [sol_x, cs, symb_stress_points], [reshape(tau_max_c, (tau_max_c.shape[0] * tau_max_c.shape[1], 1))])
            self.options['symbolic_expressions']['tau_max_n'] = sigma_vm_h
            self.options['symbolic_stress_functions']['tau_max_n'] = Function('tau_max_n', [sol_x, cs, symb_stress_points], [reshape(tau_max_n, (tau_max_n.shape[0] * tau_max_n.shape[1], 1))])

            self.options['symbolic_stress_functions']['d_sigma_axial'] = Function('d_sigma_axial', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['d_sigma_axial']])
            self.options['symbolic_stress_functions']['d_sigma_vm_w'] = Function('d_sigma_vm_w', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['d_sigma_vm_w']])
            self.options['symbolic_stress_functions']['d_sigma_vm_h'] = Function('d_sigma_vm_h', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['d_sigma_vm_h']])
            self.options['symbolic_stress_functions']['d_tau_max_c'] = Function('d_tau_max_c', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['d_tau_max_c']])
            self.options['symbolic_stress_functions']['d_tau_max_n'] = Function('d_tau_max_n', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['d_tau_max_n']])

            self.options['symbolic_stress_functions']['dc_sigma_axial'] = Function('dc_sigma_axial', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dc_sigma_axial']])
            self.options['symbolic_stress_functions']['dc_sigma_vm_w'] = Function('dc_sigma_vm_w', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dc_sigma_vm_w']])
            self.options['symbolic_stress_functions']['dc_sigma_vm_h'] = Function('dc_sigma_vm_h', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dc_sigma_vm_h']])
            self.options['symbolic_stress_functions']['dc_tau_max_c'] = Function('dc_tau_max_c', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dc_tau_max_c']])
            self.options['symbolic_stress_functions']['dc_tau_max_n'] = Function('dc_tau_max_n', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dc_tau_max_n']])

            self.options['symbolic_stress_functions']['dx_sigma_axial'] = Function('dx_sigma_axial', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dx_sigma_axial']])
            self.options['symbolic_stress_functions']['dx_sigma_vm_w'] = Function('dx_sigma_vm_w', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dx_sigma_vm_w']])
            self.options['symbolic_stress_functions']['dx_sigma_vm_h'] = Function('dx_sigma_vm_h', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dx_sigma_vm_h']])
            self.options['symbolic_stress_functions']['dx_tau_max_c'] = Function('dx_tau_max_c', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dx_tau_max_c']])
            self.options['symbolic_stress_functions']['dx_tau_max_n'] = Function('dx_tau_max_n', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dx_tau_max_n']])


        # endregion

        return sol_x

    def stress_formulae_box(self, n, T, cs):
        """
        If T = 0, we are looking at the slice
        If T = 1, static analysis
        If T > 1, dynamic analysis
        """
        stress_rec_points = self.options['corner_points']
        assert stress_rec_points.shape[1] == n
        assert int(stress_rec_points.shape[0] / 2) == 4, \
            'Implementation right now assumes 4 points at the 4 corners of the box CS'
        number_of_stress_points = int(stress_rec_points.shape[0] / 2)
        """
        The points must be ordered in the following format
            1 ------------------------------------- 2
              -                y                  -
              -                |                  -
              -                --> x              -
              -                                   -
              -                                   -
            4 ------------------------------------- 3
        """
        symb_stress_points = SX.sym('stress_pts', stress_rec_points.shape[1], stress_rec_points.shape[0]).T

        t_left = self.options['symbolic_variables']['t_left']
        t_top = self.options['symbolic_variables']['t_top']
        t_right = self.options['symbolic_variables']['t_right']
        t_bot = self.options['symbolic_variables']['t_bot']

        sol_x = SX.sym('x_sol', self.options['symbolic_variables']['x'].shape[0], T)

        # region Internal Forces and Moments
        Mc = SX.zeros(n, T)
        Ms = SX.zeros(n, T)
        Mn = SX.zeros(n, T)
        Fc = SX.zeros(n, T)
        Fs = SX.zeros(n, T)
        Fn = SX.zeros(n, T)

        T0, T0a = CalcNodalT(th=self.options['th0'], seq=self.options['seq'], n=n)
        for i in range(n):
            x_node_at_all_timesteps = sol_x[i * 18: (i + 1) * 18, :]
            M_csn = SX.zeros(3, T)
            F_csn = SX.zeros(3, T)
            for j in range(T):
                M_csn[:, j] = mtimes(T0[i], x_node_at_all_timesteps[9:12, j])
                F_csn[:, j] = mtimes(T0[i], x_node_at_all_timesteps[6:9, j])
            Mc[i, :] = M_csn[0, :]
            Ms[i, :] = M_csn[1, :]
            Mn[i, :] = M_csn[2, :]

            Fc[i, :] = F_csn[0, :]
            Fs[i, :] = F_csn[1, :]
            Fn[i, :] = F_csn[2, :]

        # endregion

        # region Torsional Shear
        cs_ordered = SX.sym('cs_ordered', 4, n)
        cs_ordered[0, :] = (t_top ** (-2) + t_left ** (-2) - (t_top ** (-1)) * t_left ** (-1)) ** (-0.5)
        cs_ordered[1, :] = (t_top ** (-2) + t_right ** (-2) - (t_top ** (-1)) * t_right ** (-1)) ** (-0.5)
        cs_ordered[2, :] = (t_bot ** (-2) + t_right ** (-2) - (t_bot ** (-1)) * t_right ** (-1)) ** (-0.5)
        cs_ordered[3, :] = (t_bot ** (-2) + t_left ** (-2) - (t_bot ** (-1)) * t_left ** (-1)) ** (-0.5)

        sign = np.array([-1, 1, 1, -1])

        tau_torsion = SX.sym('tau_torsion', number_of_stress_points * n, T)

        for j in range(cs_ordered.shape[0]):
            for i in range(n):
                tau_torsion[j * n + i, :] = sign[j] * Ms[i, :] / \
                                            (2 * self.options['symbolic_variables']['A_inner'][i] * cs_ordered[j, i])
        if T == 0:
            self.options['symbolic_expressions']['tau_torsion_slice'] = tau_torsion
        else:
            self.options['symbolic_expressions']['tau_torsion'] = tau_torsion
            self.options['symbolic_stress_functions']['tau_torsion'] = Function('tau_torsional', [sol_x, cs], [self.options['symbolic_expressions']['tau_torsion']])
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

        tau_side = SX.sym('tau_side', number_of_stress_points * n, T)
        tau_shear = SX.sym('tau_shear', number_of_stress_points * n, T)

        for j in range(cs_ordered.shape[0]):
            for i in range(n):
                Icc = self.options['symbolic_variables']['E'][i][0, 0] / self.options['E']
                Inn = self.options['symbolic_variables']['E'][i][2, 2] / self.options['E']
                tau_side[j * n + i, :] = \
                    sign_rl[j] * self.options['symbolic_variables']['Q_max_z'][i] * Fn[i, :] / (Icc * cs_ordered[j, i]) + \
                    sign_ud[j] * self.options['symbolic_variables']['Q_max_x'][i] * Fc[i, :] / (Inn * cs_ordered[j, i]) + \
                    sign_t[j] * Ms[i, :] / (2 * self.options['symbolic_variables']['A_inner'][i] * cs_torsion[j, i])
                tau_shear[j * n + i, :] = \
                    sign_rl[j] * self.options['symbolic_variables']['Q_max_z'][i] * Fn[i, :] / (Icc * cs_ordered[j, i]) + \
                    sign_ud[j] * self.options['symbolic_variables']['Q_max_x'][i] * Fc[i, :] / (Inn * cs_ordered[j, i])
        if T == 0:
            self.options['symbolic_expressions']['tau_side_slice'] = tau_side
            self.options['symbolic_expressions']['tau_shear_slice'] = tau_shear
            self.options['symbolic_stress_functions']['tau_shear_slice'] = Function('tau_shear_slice', [sol_x, cs], [self.options['symbolic_expressions']['tau_shear_slice']])
            self.options['symbolic_stress_functions']['tau_side_slice'] = Function('tau_side_slice', [sol_x, cs], [self.options['symbolic_expressions']['tau_side_slice']])
        else:
            self.options['symbolic_expressions']['tau_side'] = tau_side
            self.options['symbolic_expressions']['tau_shear'] = tau_shear
            self.options['symbolic_stress_functions']['tau_shear'] = Function('tau_shear', [sol_x, cs], [self.options['symbolic_expressions']['tau_shear']])
        # endregion

        # region Axial Stress
        # sigma_axial at a time step is
        #   0:n is 1st point (top-left)
        #   n+1:2n is 2nd point (top-right)
        #   2n+1:3n is 3rd point (bottom-right)
        #   3n+1:4n is 4th point (bottom-left)
        sigma_axial = SX.sym('sigma_a', number_of_stress_points * n, T)
        for j in range(number_of_stress_points):
            for i in range(n):
                x1 = symb_stress_points[2 * j, i]
                x3 = symb_stress_points[2 * j + 1, i]
                EIcc = self.options['symbolic_variables']['E'][i][0, 0]
                EInn = self.options['symbolic_variables']['E'][i][2, 2]
                EA = self.options['symbolic_variables']['EA'][i]
                sigma_axial[j * n + i, :] = self.options['E'] * (Fs[i, :] / EA - x3 * Mc[i, :] / (EIcc) + x1 * Mn[i, :] / (EInn))
                pass
        if T == 0:
            self.options['symbolic_expressions']['sigma_axial_slice'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial_slice'] = Function('sigma_yy_slice', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['sigma_axial_slice']])
        else:
            self.options['symbolic_expressions']['sigma_axial'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial'] = Function('sigma_yy', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['sigma_axial']])
        # endregion

        # region von-Mises Stress
        sigma = SX.sym('sigma', number_of_stress_points * n, T)
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
                EIcc = self.options['symbolic_variables']['E'][i][0, 0]
                EInn = self.options['symbolic_variables']['E'][i][2, 2]
                EA = self.options['symbolic_variables']['EA'][i]
                sigma[j * n + i, :] = self.options['E'] * (Fs[i, :] / EA - (x3 - sign_3 * t_3_selected) * Mc[i, :] / (EIcc) + (x1 - sign_1 * t_1_selected) * Mn[i, :] / (EInn))

        sigma_vm = (sigma ** 2 + tau_torsion ** 2 - sigma * tau_torsion + 1e-15) ** 0.5
        if T == 0:
            self.options['symbolic_expressions']['sigma_vm_slice'] = sigma_vm
            self.options['symbolic_stress_functions']['sigma_vm_slice'] = Function('sigma_vm_slice', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['sigma_vm_slice']])
        else:
            self.options['symbolic_expressions']['sigma_vm'] = sigma_vm
            self.options['symbolic_stress_functions']['sigma_vm'] = Function('sigma_vm', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['sigma_vm']])

            sigma = vertcat(sigma_vm, sigma_axial, tau_side)
            self.options['symbolic_expressions']['d_sigma_axial'] = jacobian(reshape(sigma_axial, (sigma_axial.shape[0] * sigma_axial.shape[1], 1)), cs)
            self.options['symbolic_expressions']['d_sigma_vm'] = jacobian(reshape(sigma_vm, (sigma_vm.shape[0] * sigma_vm.shape[1], 1)), cs)
            self.options['symbolic_expressions']['d_tau_side'] = jacobian(reshape(tau_side, (tau_side.shape[0] * tau_side.shape[1], 1)), cs)

            self.options['symbolic_expressions']['dc_sigma_axial'] = jacobian(reshape(sigma_axial, (sigma_axial.shape[0] * sigma_axial.shape[1], 1)), transpose(symb_stress_points))
            self.options['symbolic_expressions']['dc_sigma_vm'] = jacobian(reshape(sigma_vm, (sigma_vm.shape[0] * sigma_vm.shape[1], 1)), transpose(symb_stress_points))
            self.options['symbolic_expressions']['dc_tau_side'] = jacobian(reshape(tau_side, (tau_side.shape[0] * tau_side.shape[1], 1)), transpose(symb_stress_points))

            self.options['symbolic_expressions']['dx_sigma_axial'] = jacobian(reshape(sigma_axial, (sigma_axial.shape[0] * sigma_axial.shape[1], 1)), transpose(sol_x))
            self.options['symbolic_expressions']['dx_sigma_vm'] = jacobian(reshape(sigma_vm, (sigma_vm.shape[0] * sigma_vm.shape[1], 1)), transpose(sol_x))
            self.options['symbolic_expressions']['dx_tau_side'] = jacobian(reshape(tau_side, (tau_side.shape[0] * tau_side.shape[1], 1)), transpose(sol_x))
        if T == 0:
            pass
        else:
            self.options['symbolic_expressions']['sigma_axial'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial'] = Function('sigma_axial', [sol_x, cs, symb_stress_points], [reshape(sigma_axial, (sigma_axial.shape[0] * sigma_axial.shape[1], 1))])
            self.options['symbolic_expressions']['sigma_vm'] = sigma_vm
            self.options['symbolic_stress_functions']['sigma_vm'] = Function('sigma_vm', [sol_x, cs, symb_stress_points], [reshape(sigma_vm, (sigma_vm.shape[0] * sigma_vm.shape[1], 1))])
            self.options['symbolic_expressions']['tau_side'] = tau_side
            self.options['symbolic_stress_functions']['tau_side'] = Function('tau_side', [sol_x, cs, symb_stress_points], [reshape(tau_side, (tau_side.shape[0] * tau_side.shape[1], 1))])

            self.options['symbolic_stress_functions']['d_sigma_axial'] = Function('d_sigma_axial', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['d_sigma_axial']])
            self.options['symbolic_stress_functions']['d_sigma_vm'] = Function('d_sigma_vm', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['d_sigma_vm']])
            self.options['symbolic_stress_functions']['d_tau_side'] = Function('d_tau_side', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['d_tau_side']])

            self.options['symbolic_stress_functions']['dc_sigma_axial'] = Function('dc_sigma_axial', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dc_sigma_axial']])
            self.options['symbolic_stress_functions']['dc_sigma_vm'] = Function('dc_sigma_vm', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dc_sigma_vm']])
            self.options['symbolic_stress_functions']['dc_tau_side'] = Function('dc_tau_side', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dc_tau_side']])

            self.options['symbolic_stress_functions']['dx_sigma_axial'] = Function('dx_sigma_axial', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dx_sigma_axial']])
            self.options['symbolic_stress_functions']['dx_sigma_vm'] = Function('dx_sigma_vm', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dx_sigma_vm']])
            self.options['symbolic_stress_functions']['dx_tau_side'] = Function('dx_tau_side', [sol_x, cs, symb_stress_points], [self.options['symbolic_expressions']['dx_tau_side']])
        # endregion

        return sol_x


class BasicEulerBernoulliStressModel(om.ExplicitComponent):
    def initialize(self):
        # Make sure all the quantities necessary to make the system work are here
        self.options.declare('name', types=str)  # Just to tag the constraint in particular
        self.options.declare('debug_flag', types=bool, default=False)  # To enable or disable debugging
        self.options.declare('num_divisions', types=int)  # To generate optional constraint mechanisms
        self.options.declare('num_DvCs', types=int)  # To know the actual number of cross-sectional variables
        self.options.declare('symbolic_variables',
                             types=dict)  # Where all the resultant cross-section symbolics come from beam parent
        self.options.declare('beam_shape', types=BeamCS)  # To check what beam type is coming through
        # Where helper and needed symbolic expressions (Einv) and values (x) are stored
        self.options.declare('symbolic_expressions', types=dict)
        self.options.declare('symbolic_stress_functions', types=dict)

        self.options.declare('corner_points')
        self.options.declare('num_timesteps')

        # Necessary Beam Characteristics
        self.options.declare('r0')
        self.options.declare('th0')
        self.options.declare('seq')
        self.options.declare('E')
        self.options.declare('G')

        self.options['symbolic_variables'] = {}
        self.options['symbolic_expressions'] = {}
        self.options['symbolic_stress_functions'] = {}

    def setup(self):
        # Setup all the nice goodies that make this thing work
        self.add_input('cs', shape=self.options['beam_shape'].value * self.options['num_DvCs'])
        self.add_input('x', shape=(18 * self.options['num_divisions'], self.options['num_timesteps']))
        # Time to generate the stress formulas:
        cs = self.options['symbolic_variables']['cs']
        if self.options['beam_shape'] == BeamCS.RECTANGULAR:  # Rectangular beam with 2 degrees of freedom per cross-section h w
            self.options['corner_points'] = np.zeros((self.options['symbolic_variables']['corner_points'].shape[0], self.options['symbolic_variables']['corner_points'].shape[1]))
            self.add_input('corner_points', shape=(self.options['symbolic_variables']['corner_points'].shape[0] * self.options['symbolic_variables']['corner_points'].shape[1]))

            """
                    The following are the stresses modeled in the rectangular beam:
                    T x 0n -> 4n : Axial Stresses at the corners
                    T x 4n -> 8n : Von Misses Stress at the edges (center of edge)
                    T x 8n -> 9n : Shear stress at the horizontal beam direction
                    T x 9n -> 10n: Shear stress at the vertical beam direction
            """
            self.stress_formulae_rect(self.options['num_divisions'], self.options['num_timesteps'], cs)

            self.add_output('sigma_axial', shape=(4 * self.options['num_divisions'] * (self.options['num_timesteps'])))
            self.add_output('sigma_vm', shape=(4 * self.options['num_divisions'] * (self.options['num_timesteps'])))
            self.add_output('tau_max_c', shape=(self.options['num_divisions'] * (self.options['num_timesteps'])))
            self.add_output('tau_max_n', shape=(self.options['num_divisions'] * (self.options['num_timesteps'])))

        if self.options['beam_shape'] == BeamCS.BOX:
            self.options['corner_points'] = np.zeros((self.options['symbolic_variables']['corner_points'].shape[0],
                                                      self.options['symbolic_variables']['corner_points'].shape[1]))
            self.add_input('corner_points', shape=(self.options['symbolic_variables']['corner_points'].shape[0] * self.options['symbolic_variables']['corner_points'].shape[1]))

            """
                                The following are the stresses modeled in the rectangular beam:
                                T x 0n -> 4n : von-mises stress at the corners
                                T x 4n -> 8n : axial stress at the corners
                                T x 8n -> 12n : Shear stress at the flange centers
            """
            self.stress_formulae_box(self.options['num_divisions'], self.options['num_timesteps'], cs)

            self.add_output('sigma_axial', shape=(4 * self.options['num_divisions'] * (self.options['num_timesteps'])))
            self.add_output('sigma_vm', shape=(4 * self.options['num_divisions'] * (self.options['num_timesteps'])))
            self.add_output('tau_side', shape=(4 * self.options['num_divisions'] * (self.options['num_timesteps'])))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        corner_points = np.reshape(inputs['corner_points'], (self.options['symbolic_variables']['corner_points'].shape[0], self.options['symbolic_variables']['corner_points'].shape[1]))
        if self.options['debug_flag']:
            # Point 1
            assert (corner_points[0, :] < 0).all()  # x coordinate must be negative
            assert (corner_points[1, :] > 0).all()  # y coordinate must be positive
            # Point 2
            assert (corner_points[2, :] > 0).all()  # x coordinate must be positive
            assert (corner_points[3, :] > 0).all()  # y coordinate must be positive
            # Point 3
            assert (corner_points[4, :] > 0).all()  # x coordinate must be positive
            assert (corner_points[5, :] < 0).all()  # y coordinate must be negative
            # Point 4
            assert (corner_points[6, :] < 0).all()  # x coordinate must be negative
            assert (corner_points[7, :] < 0).all()  # y coordinate must be negative

        if self.options['beam_shape'] == BeamCS.RECTANGULAR:
            print(inputs['cs'])
            h = inputs['cs'][0:self.options['num_DvCs']]
            w = inputs['cs'][self.options['num_DvCs']:2 * self.options['num_DvCs']]

            sigma_axial = self.options['symbolic_stress_functions']['sigma_axial'](inputs['x'], inputs['cs'], corner_points).full()
            tau_max_c = self.options['symbolic_stress_functions']['tau_max_c'](inputs['x'], inputs['cs'], corner_points).full()
            tau_max_n = self.options['symbolic_stress_functions']['tau_max_n'](inputs['x'], inputs['cs'], corner_points).full()

            if np.linalg.norm(h) < np.linalg.norm(w):
                sigma_vm = self.options['symbolic_stress_functions']['sigma_vm_w'](inputs['x'], inputs['cs'], corner_points).full()
            else:
                sigma_vm = self.options['symbolic_stress_functions']['sigma_vm_h'](inputs['x'], inputs['cs'], corner_points).full()

            outputs['sigma_axial'] = sigma_axial
            outputs['sigma_vm'] = sigma_vm
            outputs['tau_max_c'] = tau_max_c
            outputs['tau_max_n'] = tau_max_n
            pass
        if self.options['beam_shape'] == BeamCS.BOX:
            sigma_axial = self.options['symbolic_stress_functions']['sigma_axial'](inputs['x'], inputs['cs'], corner_points).full()
            sigma_vm = self.options['symbolic_stress_functions']['sigma_vm'](inputs['x'], inputs['cs'], corner_points).full()
            tau_side = self.options['symbolic_stress_functions']['tau_side'](inputs['x'], inputs['cs'], corner_points).full()

            outputs['sigma_axial'] = sigma_axial
            outputs['sigma_vm'] = sigma_vm
            outputs['tau_side'] = tau_side

    def stress_formulae_rect(self, n, T, cs):
        """
        If T = 0, we are looking at the slice
        If T = 1, static analysis
        If T > 1, dynamic analysis
        """
        stress_rec_points = self.options['corner_points']
        assert stress_rec_points.shape[1] == n
        assert int(stress_rec_points.shape[0] / 2) == 4, \
            'Implementation right now assumes 4 points at the 4 corners of the box CS'
        number_of_stress_points = int(stress_rec_points.shape[0] / 2)
        """
        The points must be ordered in the following format
            1 ------------------------------------- 2
              -                y                  -
              -                |                  -
              -                --> x              -
              -                                   -
              -                                   -
            4 ------------------------------------- 3
        """

        symb_stress_points = SX.sym('stress_pts', stress_rec_points.shape[1], stress_rec_points.shape[0]).T

        h = self.options['symbolic_variables']['h']
        w = self.options['symbolic_variables']['w']

        sol_x = SX.sym('x_sol', 18 * self.options['num_divisions'], T)

        self.options['symbolic_variables']['x'] = sol_x
        self.options['symbolic_variables']['symb_stress_points'] = symb_stress_points
        self.options['symbolic_variables']['total_mass'] = self.options['symbolic_variables']['total_mass']

        # region Internal Forces and Moments
        Mc = SX.zeros(n, T)
        Ms = SX.zeros(n, T)
        Mn = SX.zeros(n, T)
        Fc = SX.zeros(n, T)
        Fs = SX.zeros(n, T)
        Fn = SX.zeros(n, T)

        T0, T0a = CalcNodalT(th=self.options['th0'], seq=self.options['seq'], n=n)
        for i in range(n):
            x_node_at_all_timesteps = sol_x[i * 18: (i + 1) * 18, :]
            M_csn = SX.zeros(3, T)
            F_csn = SX.zeros(3, T)
            for j in range(T):
                M_csn[:, j] = mtimes(T0[i], x_node_at_all_timesteps[9:12, j])
                F_csn[:, j] = mtimes(T0[i], x_node_at_all_timesteps[6:9, j])
            Mc[i, :] = M_csn[0, :]
            Ms[i, :] = M_csn[1, :]
            Mn[i, :] = M_csn[2, :]

            Fc[i, :] = F_csn[0, :]
            Fs[i, :] = F_csn[1, :]
            Fn[i, :] = F_csn[2, :]

        # endregion

        # Torsion calculation on a solid cross-section follows Roark's formula for Stress:

        # When w > h:
        tau_torsion_w_slice = ((3 * Ms) / (w * h ** 2)) * (
                1 + 0.6095 * (h / w) + 0.8865 * (h / w) ** 2 - 1.8023 * (h / w) ** 3 + 0.91 * (h / w) ** 4)

        # When h > w:
        tau_torsion_h_slice = ((3 * Ms) / (h * w ** 2)) * (
                1 + 0.6095 * (w / h) + 0.8865 * (w / h) ** 2 - 1.8023 * (w / h) ** 3 + 0.91 * (w / h) ** 4)

        # Sign convention to make sure torsion goes in the convention chosen in the csn axes:
        sign = np.array([-1, 1, 1, -1])

        tau_torsion_w = vertcat(sign[0] * tau_torsion_w_slice, sign[1] * tau_torsion_w_slice,
                                sign[2] * tau_torsion_w_slice, sign[3] * tau_torsion_w_slice)

        tau_torsion_h = vertcat(sign[0] * tau_torsion_h_slice, sign[1] * tau_torsion_h_slice,
                                sign[2] * tau_torsion_h_slice, sign[3] * tau_torsion_h_slice)
        if T == 0:
            self.options['symbolic_expressions']['tau_torsion_slice_w'] = tau_torsion_w
            self.options['symbolic_expressions']['tau_torsion_slice_h'] = tau_torsion_h
        else:
            self.options['symbolic_expressions']['tau_torsion_w'] = tau_torsion_w
            self.options['symbolic_expressions']['tau_torsion_h'] = tau_torsion_h
            self.options['symbolic_stress_functions']['tau_torsion_w'] = Function('tau_torsional_w',
                                                                                  [sol_x,
                                                                                   cs],
                                                                                  [self.options['symbolic_expressions'][
                                                                                       'tau_torsion_w']])
            self.options['symbolic_stress_functions']['tau_torsion_h'] = Function('tau_torsional_h',
                                                                                  [sol_x,
                                                                                   cs],
                                                                                  [self.options['symbolic_expressions'][
                                                                                       'tau_torsion_h']])
        # endregion

        # region Transverse Shear
        tau_max_c = 1.5 * Fc / (h * w)
        tau_max_n = 1.5 * Fn / (h * w)

        self.options['symbolic_expressions']['tau_max_c_slice'] = tau_max_c
        self.options['symbolic_expressions']['tau_max_n_slice'] = tau_max_n

        if T == 0:
            self.options['symbolic_expressions']['tau_shear_slice'] = vertcat(tau_max_c, tau_max_n)
            self.options['symbolic_stress_functions']['tau_shear_slice'] = Function('tau_shear_slice',
                                                                                    [sol_x,
                                                                                     cs],
                                                                                    [self.options[
                                                                                         'symbolic_expressions'][
                                                                                         'tau_shear_slice']])
        else:
            self.options['symbolic_expressions']['tau_shear'] = vertcat(tau_max_c, tau_max_n)
            self.options['symbolic_stress_functions']['tau_shear'] = Function('tau_shear',
                                                                              [sol_x,
                                                                               cs],
                                                                              [self.options['symbolic_expressions'][
                                                                                   'tau_shear']])
        # endregion

        # region Axial Stress
        # sigma_axial at a time step is
        #   0:n is 1st point (top-left)
        #   n+1:2n is 2nd point (top-right)
        #   2n+1:3n is 3rd point (bottom-right)
        #   3n+1:4n is 4th point (bottom-left)
        sigma_axial = SX.sym('sigma_a', number_of_stress_points * n, T)
        for j in range(number_of_stress_points):
            for i in range(n):
                x1 = symb_stress_points[2 * j, i]
                x3 = symb_stress_points[2 * j + 1, i]
                EIcc = self.options['symbolic_variables']['E'][i][0, 0]
                EInn = self.options['symbolic_variables']['E'][i][2, 2]
                EA = self.options['symbolic_variables']['EA'][i]
                sigma_axial[j * n + i, :] = self.options['E'] * (Fs[i, :] / EA -
                                                                 x3 * Mc[i, :] / (EIcc) +
                                                                 x1 * Mn[i, :] / (EInn))
                pass
        if T == 0:
            self.options['symbolic_expressions']['sigma_axial_slice'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial_slice'] = Function('sigma_yy_slice',
                                                                                      [sol_x,
                                                                                       cs,
                                                                                       symb_stress_points],
                                                                                      [self.options[
                                                                                           'symbolic_expressions'][
                                                                                           'sigma_axial_slice']])
        else:
            self.options['symbolic_expressions']['sigma_axial'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial'] = Function('sigma_yy',
                                                                                [sol_x,
                                                                                 cs,
                                                                                 symb_stress_points],
                                                                                [self.options['symbolic_expressions'][
                                                                                     'sigma_axial']])
        # endregion

        # region von-Mises Stress

        sigma_vm_w = (sigma_axial ** 2 + tau_torsion_w ** 2 - sigma_axial * tau_torsion_w + 1e-15) ** 0.5

        sigma_vm_h = (sigma_axial ** 2 + tau_torsion_h ** 2 - sigma_axial * tau_torsion_h + 1e-15) ** 0.5

        if T == 0:
            self.options['symbolic_expressions']['sigma_vm_slice_w'] = sigma_vm_w
            self.options['symbolic_stress_functions']['sigma_vm_slice_w'] = Function('sigma_vm_slice_w',
                                                                                     [sol_x,
                                                                                      cs,
                                                                                      symb_stress_points],
                                                                                     [self.options[
                                                                                          'symbolic_expressions'][
                                                                                          'sigma_vm_slice_w']])
            self.options['symbolic_expressions']['sigma_vm_slice_h'] = sigma_vm_h
            self.options['symbolic_stress_functions']['sigma_vm_slice_h'] = Function('sigma_vm_slice_h',
                                                                                     [sol_x,
                                                                                      cs,
                                                                                      symb_stress_points],
                                                                                     [self.options[
                                                                                          'symbolic_expressions'][
                                                                                          'sigma_vm_slice_h']])
        else:
            self.options['symbolic_expressions']['sigma_vm_w'] = sigma_vm_w
            self.options['symbolic_stress_functions']['sigma_vm_w'] = Function('sigma_vm_w',
                                                                               [sol_x,
                                                                                cs,
                                                                                symb_stress_points],
                                                                               [self.options['symbolic_expressions'][
                                                                                    'sigma_vm_w']])
            self.options['symbolic_expressions']['sigma_vm_h'] = sigma_vm_h
            self.options['symbolic_stress_functions']['sigma_vm_h'] = Function('sigma_vm_h',
                                                                               [sol_x,
                                                                                cs,
                                                                                symb_stress_points],
                                                                               [self.options['symbolic_expressions'][
                                                                                    'sigma_vm_h']])
        # endregion

        # Region Stress Function Buildup

        self.options['symbolic_expressions']['d_sigma_axial'] = jacobian(reshape(sigma_axial, (sigma_axial.shape[0] * sigma_axial.shape[1], 1)), cs)
        self.options['symbolic_expressions']['d_sigma_vm_w'] = jacobian(reshape(sigma_vm_w, (sigma_vm_w.shape[0] * sigma_vm_w.shape[1], 1)), cs)
        self.options['symbolic_expressions']['d_sigma_vm_h'] = jacobian(reshape(sigma_vm_h, (sigma_vm_h.shape[0] * sigma_vm_h.shape[1], 1)), cs)
        self.options['symbolic_expressions']['d_tau_max_c'] = jacobian(reshape(tau_max_c, (tau_max_c.shape[0] * tau_max_c.shape[1], 1)), cs)
        self.options['symbolic_expressions']['d_tau_max_n'] = jacobian(reshape(tau_max_n, (tau_max_n.shape[0] * tau_max_n.shape[1], 1)), cs)

        if T == 0:
            pass
        else:
            self.options['symbolic_expressions']['sigma_axial'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial'] = Function('sigma_axial',
                                                                                [sol_x,
                                                                                 cs,
                                                                                 symb_stress_points],
                                                                                [reshape(sigma_axial, (sigma_axial.shape[0] * sigma_axial.shape[1], 1))])
            self.options['symbolic_expressions']['sigma_vm_w'] = sigma_vm_w
            self.options['symbolic_stress_functions']['sigma_vm_w'] = Function('sigma_vm_w',
                                                                               [sol_x,
                                                                                cs,
                                                                                symb_stress_points],
                                                                               [reshape(sigma_vm_w, (sigma_vm_w.shape[0] * sigma_vm_w.shape[1], 1))])
            self.options['symbolic_expressions']['sigma_vm_h'] = sigma_vm_h
            self.options['symbolic_stress_functions']['sigma_vm_h'] = Function('sigma_vm_h',
                                                                               [sol_x,
                                                                                cs,
                                                                                symb_stress_points],
                                                                               [reshape(sigma_vm_h, (sigma_vm_h.shape[0] * sigma_vm_h.shape[1], 1))])
            self.options['symbolic_expressions']['tau_max_c'] = tau_max_c
            self.options['symbolic_stress_functions']['tau_max_c'] = Function('tau_max_c',
                                                                              [sol_x,
                                                                               cs,
                                                                               symb_stress_points],
                                                                              [reshape(tau_max_c, (tau_max_c.shape[0] * tau_max_c.shape[1], 1))])
            self.options['symbolic_expressions']['tau_max_n'] = sigma_vm_h
            self.options['symbolic_stress_functions']['tau_max_n'] = Function('tau_max_n',
                                                                              [sol_x,
                                                                               cs,
                                                                               symb_stress_points],
                                                                              [reshape(tau_max_n, (tau_max_n.shape[0] * tau_max_n.shape[1], 1))])

            self.options['symbolic_stress_functions']['d_sigma_axial'] = Function('d_sigma_axial',
                                                                                  [sol_x,
                                                                                   cs,
                                                                                   symb_stress_points],
                                                                                  [self.options['symbolic_expressions']['d_sigma_axial']])
            self.options['symbolic_stress_functions']['d_sigma_vm_w'] = Function('d_sigma_vm_w',
                                                                                 [sol_x,
                                                                                  cs,
                                                                                  symb_stress_points],
                                                                                 [self.options['symbolic_expressions']['d_sigma_vm_w']])
            self.options['symbolic_stress_functions']['d_sigma_vm_h'] = Function('d_sigma_vm_h',
                                                                                 [sol_x,
                                                                                  cs,
                                                                                  symb_stress_points],
                                                                                 [self.options['symbolic_expressions']['d_sigma_vm_h']])
            self.options['symbolic_stress_functions']['d_tau_max_c'] = Function('d_tau_max_c',
                                                                                [sol_x,
                                                                                 cs,
                                                                                 symb_stress_points],
                                                                                [self.options['symbolic_expressions']['d_tau_max_c']])
            self.options['symbolic_stress_functions']['d_tau_max_n'] = Function('d_tau_max_n',
                                                                                [sol_x,
                                                                                 cs,
                                                                                 symb_stress_points],
                                                                                [self.options['symbolic_expressions']['d_tau_max_n']])
        # endregion

        return sol_x

    def stress_formulae_box(self, n, T, cs):
        """
        If T = 0, we are looking at the slice
        If T = 1, static analysis
        If T > 1, dynamic analysis
        """
        stress_rec_points = self.options['corner_points']
        assert stress_rec_points.shape[1] == n
        assert int(stress_rec_points.shape[0] / 2) == 4, \
            'Implementation right now assumes 4 points at the 4 corners of the box CS'
        number_of_stress_points = int(stress_rec_points.shape[0] / 2)
        """
        The points must be ordered in the following format
            1 ------------------------------------- 2
              -                y                  -
              -                |                  -
              -                --> x              -
              -                                   -
              -                                   -
            4 ------------------------------------- 3
        """
        symb_stress_points = SX.sym('stress_pts', stress_rec_points.shape[1], stress_rec_points.shape[0]).T

        t_left = self.options['symbolic_variables']['t_left']
        t_top = self.options['symbolic_variables']['t_top']
        t_right = self.options['symbolic_variables']['t_right']
        t_bot = self.options['symbolic_variables']['t_bot']

        sol_x = SX.sym('x_sol', self.options['symbolic_variables']['x'].shape[0], T)

        self.options['symbolic_variables']['x'] = sol_x
        self.options['symbolic_variables']['symb_stress_points'] = symb_stress_points

        # region Internal Forces and Moments
        Mc = SX.zeros(n, T)
        Ms = SX.zeros(n, T)
        Mn = SX.zeros(n, T)
        Fc = SX.zeros(n, T)
        Fs = SX.zeros(n, T)
        Fn = SX.zeros(n, T)

        T0, T0a = CalcNodalT(th=self.options['th0'], seq=self.options['seq'], n=n)
        for i in range(n):
            x_node_at_all_timesteps = sol_x[i * 18: (i + 1) * 18, :]
            M_csn = SX.zeros(3, T)
            F_csn = SX.zeros(3, T)
            for j in range(T):
                M_csn[:, j] = mtimes(T0[i], x_node_at_all_timesteps[9:12, j])
                F_csn[:, j] = mtimes(T0[i], x_node_at_all_timesteps[6:9, j])
            Mc[i, :] = M_csn[0, :]
            Ms[i, :] = M_csn[1, :]
            Mn[i, :] = M_csn[2, :]

            Fc[i, :] = F_csn[0, :]
            Fs[i, :] = F_csn[1, :]
            Fn[i, :] = F_csn[2, :]

        # sol_x = reshape(sol_x, sol_x.shape[0] * sol_x.shape[1], 1)
        # endregion

        # region Torsional Shear
        cs_ordered = SX.sym('cs_ordered', 4, n)
        cs_ordered[0, :] = (t_top ** (-2) + t_left ** (-2) - (t_top ** (-1)) * t_left ** (-1)) ** (-0.5)
        cs_ordered[1, :] = (t_top ** (-2) + t_right ** (-2) - (t_top ** (-1)) * t_right ** (-1)) ** (-0.5)
        cs_ordered[2, :] = (t_bot ** (-2) + t_right ** (-2) - (t_bot ** (-1)) * t_right ** (-1)) ** (-0.5)
        cs_ordered[3, :] = (t_bot ** (-2) + t_left ** (-2) - (t_bot ** (-1)) * t_left ** (-1)) ** (-0.5)

        sign = np.array([-1, 1, 1, -1])

        tau_torsion = SX.sym('tau_torsion', number_of_stress_points * n, T)

        for j in range(cs_ordered.shape[0]):
            for i in range(n):
                tau_torsion[j * n + i, :] = sign[j] * Ms[i, :] / \
                                            (2 * self.options['symbolic_variables']['A_inner'][i] * cs_ordered[j, i])
        if T == 0:
            self.options['symbolic_expressions']['tau_torsion_slice'] = tau_torsion
        else:
            self.options['symbolic_expressions']['tau_torsion'] = tau_torsion
            self.options['symbolic_stress_functions']['tau_torsion'] = Function('tau_torsional',
                                                                                [sol_x,
                                                                                 cs],
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

        tau_side = SX.sym('tau_side', number_of_stress_points * n, T)
        tau_shear = SX.sym('tau_shear', number_of_stress_points * n, T)

        for j in range(cs_ordered.shape[0]):
            for i in range(n):
                Icc = self.options['symbolic_variables']['E'][i][0, 0] / self.options['E']
                Inn = self.options['symbolic_variables']['E'][i][2, 2] / self.options['E']
                tau_side[j * n + i, :] = \
                    sign_rl[j] * self.options['symbolic_variables']['Q_max_z'][i] * Fn[i, :] / (Icc * cs_ordered[j, i]) + \
                    sign_ud[j] * self.options['symbolic_variables']['Q_max_x'][i] * Fc[i, :] / (Inn * cs_ordered[j, i]) + \
                    sign_t[j] * Ms[i, :] / (2 * self.options['symbolic_variables']['A_inner'][i] * cs_torsion[j, i])
                tau_shear[j * n + i, :] = \
                    sign_rl[j] * self.options['symbolic_variables']['Q_max_z'][i] * Fn[i, :] / (Icc * cs_ordered[j, i]) + \
                    sign_ud[j] * self.options['symbolic_variables']['Q_max_x'][i] * Fc[i, :] / (Inn * cs_ordered[j, i])
        if T == 0:
            self.options['symbolic_expressions']['tau_side_slice'] = tau_side
            self.options['symbolic_expressions']['tau_shear_slice'] = tau_shear
            self.options['symbolic_stress_functions']['tau_shear_slice'] = Function('tau_shear_slice',
                                                                                    [sol_x,
                                                                                     cs],
                                                                                    [self.options['symbolic_expressions']['tau_shear_slice']])
            self.options['symbolic_stress_functions']['tau_side_slice'] = Function('tau_side_slice',
                                                                                   [sol_x,
                                                                                    cs],
                                                                                   [self.options['symbolic_expressions']['tau_side_slice']])
        else:
            self.options['symbolic_expressions']['tau_side'] = tau_side
            self.options['symbolic_expressions']['tau_shear'] = tau_shear
            self.options['symbolic_stress_functions']['tau_shear'] = Function('tau_shear',
                                                                              [sol_x,
                                                                               cs],
                                                                              [self.options['symbolic_expressions']['tau_shear']])
        # endregion

        # region Axial Stress
        # sigma_axial at a time step is
        #   0:n is 1st point (top-left)
        #   n+1:2n is 2nd point (top-right)
        #   2n+1:3n is 3rd point (bottom-right)
        #   3n+1:4n is 4th point (bottom-left)
        sigma_axial = SX.sym('sigma_a', number_of_stress_points * n, T)
        for j in range(number_of_stress_points):
            for i in range(n):
                x1 = symb_stress_points[2 * j, i]
                x3 = symb_stress_points[2 * j + 1, i]
                EIcc = self.options['symbolic_variables']['E'][i][0, 0]
                EInn = self.options['symbolic_variables']['E'][i][2, 2]
                EA = self.options['symbolic_variables']['EA'][i]
                sigma_axial[j * n + i, :] = self.options['E'] * (Fs[i, :] / EA -
                                                                 x3 * Mc[i, :] / (EIcc) +
                                                                 x1 * Mn[i, :] / (EInn))
                pass
        if T == 0:
            self.options['symbolic_expressions']['sigma_axial_slice'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial_slice'] = Function('sigma_yy_slice',
                                                                                      [sol_x,
                                                                                       cs,
                                                                                       symb_stress_points],
                                                                                      [self.options['symbolic_expressions']['sigma_axial_slice']])
        else:
            self.options['symbolic_expressions']['sigma_axial'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial'] = Function('sigma_yy',
                                                                                [sol_x,
                                                                                 cs,
                                                                                 symb_stress_points],
                                                                                [self.options['symbolic_expressions']['sigma_axial']])
        # endregion

        # region von-Mises Stress
        sigma = SX.sym('sigma', number_of_stress_points * n, T)
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
                EIcc = self.options['symbolic_variables']['E'][i][0, 0]
                EInn = self.options['symbolic_variables']['E'][i][2, 2]
                EA = self.options['symbolic_variables']['EA'][i]
                sigma[j * n + i, :] = self.options['E'] * (Fs[i, :] / EA -
                                                           (x3 - sign_3 * t_3_selected) * Mc[i, :] / (EIcc) +
                                                           (x1 - sign_1 * t_1_selected) * Mn[i, :] / (EInn))

        sigma_vm = (sigma ** 2 + tau_torsion ** 2 - sigma * tau_torsion + 1e-15) ** 0.5
        if T == 0:
            self.options['symbolic_expressions']['sigma_vm_slice'] = sigma_vm
            self.options['symbolic_stress_functions']['sigma_vm_slice'] = Function('sigma_vm_slice',
                                                                                   [sol_x,
                                                                                    cs,
                                                                                    symb_stress_points],
                                                                                   [self.options['symbolic_expressions']['sigma_vm_slice']])
        else:
            self.options['symbolic_expressions']['sigma_vm'] = sigma_vm
            self.options['symbolic_stress_functions']['sigma_vm'] = Function('sigma_vm',
                                                                             [sol_x,
                                                                              cs,
                                                                              symb_stress_points],
                                                                             [self.options['symbolic_expressions']['sigma_vm']])

            sigma = vertcat(sigma_vm, sigma_axial, tau_side)
            self.options['symbolic_expressions']['d_sigma_axial'] = jacobian(reshape(sigma_axial, (sigma_axial.shape[0] * sigma_axial.shape[1], 1)), cs)
            self.options['symbolic_expressions']['d_sigma_vm'] = jacobian(reshape(sigma_vm, (sigma_vm.shape[0] * sigma_vm.shape[1], 1)), cs)
            self.options['symbolic_expressions']['d_tau_side'] = jacobian(reshape(tau_side, (tau_side.shape[0] * tau_side.shape[1], 1)), cs)
        if T == 0:
            pass
        else:
            self.options['symbolic_expressions']['sigma_axial'] = sigma_axial
            self.options['symbolic_stress_functions']['sigma_axial'] = Function('sigma_axial',
                                                                                [sol_x,
                                                                                 cs,
                                                                                 symb_stress_points],
                                                                                [reshape(sigma_axial, (sigma_axial.shape[0] * sigma_axial.shape[1], 1))])
            self.options['symbolic_expressions']['sigma_vm'] = sigma_vm
            self.options['symbolic_stress_functions']['sigma_vm'] = Function('sigma_vm',
                                                                             [sol_x,
                                                                              cs,
                                                                              symb_stress_points],
                                                                             [reshape(sigma_vm, (sigma_vm.shape[0] * sigma_vm.shape[1], 1))])
            self.options['symbolic_expressions']['tau_side'] = tau_side
            self.options['symbolic_stress_functions']['tau_side'] = Function('tau_side',
                                                                             [sol_x,
                                                                              cs,
                                                                              symb_stress_points],
                                                                             [reshape(tau_side, (tau_side.shape[0] * tau_side.shape[1], 1))])

            self.options['symbolic_stress_functions']['d_sigma_axial'] = Function('d_sigma_axial',
                                                                                  [sol_x,
                                                                                   cs,
                                                                                   symb_stress_points],
                                                                                  [self.options['symbolic_expressions']['d_sigma_axial']])
            self.options['symbolic_stress_functions']['d_sigma_vm'] = Function('d_sigma_vm',
                                                                               [sol_x,
                                                                                cs,
                                                                                symb_stress_points],
                                                                               [self.options['symbolic_expressions']['d_sigma_vm']])
            self.options['symbolic_stress_functions']['d_tau_side'] = Function('d_tau_side',
                                                                               [sol_x,
                                                                                cs,
                                                                                symb_stress_points],
                                                                               [self.options['symbolic_expressions']['d_tau_side']])
        # endregion

        return sol_x
