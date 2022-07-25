import openmdao.api as om
from abc import ABC, abstractmethod
from openasemdao.structures.utils.utils import calculate_th0, CalcNodalT
from casadi import *
from openasemdao.structures.beam.constraints import StrengthAggregatedConstraint
from openasemdao.structures.utils.beam_categories import BeamCS, BoundaryType, SectionType


class BeamInterface(om.ExplicitComponent):
    # The purpose of this class is to serve as the numerical face of the Beam group, giving numerical
    # outputs to all the different symbolic expressions defined within Beam, regardless of their type or
    # number of variables. This one does NOT connect to the constraints
    def initialize(self):
        self.options.declare('name', types=str)    # Name of the beam interface (not modifiable to the user)
        self.options.declare('delta_s0')           # Needed for the mass computation of the beam
        self.options.declare('num_divisions', types=int)    # Number of cross-sections accounting for modifications
        self.options.declare('num_DvCs', types=int)    # Number of cross-sections that have design variables
        self.options.declare('beam_shape', types=BeamCS)  # Type of cross-section. An enumeration pointing to number of independent cs variables
        self.options.declare('symbolic_parent')  # Symbolic expressions that are needed for computations
        self.options.declare('num_timesteps')
        self.options.declare('debug_flag', types=bool, default=False)
        # Function per constraint:
        self.symbolic_functions = {}

    def setup(self):
        self.symbolic_functions = self.options['symbolic_parent']

        # self.symbolic_functions['corner_points']

        self.options['num_divisions'] = self.symbolic_functions['mu'].size_out(0)[0] + 1

        # Traditional input outputs:
        self.add_input('cs', shape=self.options['beam_shape'].value * self.options['num_DvCs'])
        self.add_output('cs_out', shape=self.options['beam_shape'].value * self.options['num_DvCs'])
        self.add_output('mass', shape=1)

        # Symbolic numerical channels:
        self.add_output('D', shape=(self.options['num_divisions'], 3, 3))
        self.add_output('oneover', shape=(self.options['num_divisions'], 3, 3))
        self.add_output('mu', shape=self.options['num_divisions'] - 1)
        self.add_output('i_matrix', shape=(self.options['num_divisions'] - 1, 3, 3))
        self.add_output('delta_r_CG_tilde', shape=(self.options['num_divisions'] - 1, 3, 3))
        self.add_output('Einv', shape=(self.options['num_divisions'], 3, 3))
        self.add_output('E', shape=(self.options['num_divisions'], 3, 3))
        self.add_output('EA', shape=self.options['num_divisions'])
        self.add_output('corner_points', shape=(self.symbolic_functions['corner_points'].size_out(0)[0], self.symbolic_functions['corner_points'].size_out(0)[1]))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cs_num = inputs['cs']

        outputs['D'] = np.zeros([self.options['num_divisions'], 3, 3])
        outputs['Einv'] = np.zeros([self.options['num_divisions'], 3, 3])
        outputs['oneover'] = np.zeros([self.options['num_divisions'], 3, 3])
        outputs['i_matrix'] = np.zeros([self.options['num_divisions'] - 1, 3, 3])
        outputs['mu'] = np.zeros(self.options['num_divisions'] - 1)


        outputs['cs_out'] = cs_num

        D_num = self.symbolic_functions['D'](cs_num)
        Einv_num = self.symbolic_functions['Einv'](cs_num)
        oneover_num = self.symbolic_functions['oneover'](cs_num)
        mu_num = self.symbolic_functions['mu'](cs_num)
        i_matrix_num = self.symbolic_functions['i_matrix'](cs_num)

        for i in range(self.options['num_divisions']):
            if i < self.options['num_divisions'] - 1:  # Both nodal and element quantities
                outputs['D'][i] = D_num[i].full()
                outputs['Einv'][i] = Einv_num[i].full()
                outputs['oneover'][i] = oneover_num[i].full()
                outputs['mu'][i] = mu_num[i].full()
                outputs['i_matrix'][i] = i_matrix_num[i].full()
            else:  # Only nodal quantities
                outputs['D'][i] = D_num[i].full()
                outputs['Einv'][i] = Einv_num[i].full()
                outputs['oneover'][i] = oneover_num[i].full()

        total_mass = self.symbolic_functions['mass'](cs_num, self.options['delta_s0'])
        stress_recovery_points = self.symbolic_functions['corner_points'](cs_num)

        outputs['mass'] = total_mass.full()
        outputs['corner_points'] = stress_recovery_points.full()


class StaticStateOutput(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n', types=int)

    def setup(self):
        n = self.options['n']
        # The only input is the converged state:
        self.add_input('x', shape=18 * n)
        # Define the state outputs that are necessary:
        self.add_output('r', shape=(n, 3))
        self.add_output('theta', shape=(n, 3))
        self.add_output('F', shape=(n, 3))
        self.add_output('M', shape=(n, 3))
        self.add_output('u', shape=(n, 3))
        self.add_output('omega', shape=(n, 3))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n = self.options['n']
        x = np.transpose(np.reshape(inputs['x'], (n, 18)))
        outputs['r'] = x[0:3, :]
        outputs['theta'] = x[3:6, :]
        outputs['F'] = x[6:9, :]
        outputs['M'] = x[9:12, :]
        outputs['u'] = x[12:15, :]
        outputs['omega'] = x[15:18, :]


class DynamicStateOutput(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n', types=int)
        self.options.declare('T', types=int)

    def setup(self):
        n = self.options['n']
        T = self.options['T']
        # The only input is the converged state:
        self.add_input('x', shape=(T, 18 * n))
        # Define the state outputs that are necessary:
        self.add_output('r', shape=(T, 3*n))
        self.add_output('theta', shape=(T, 3*n))
        self.add_output('F', shape=(T, 3*n))
        self.add_output('M', shape=(T, 3*n))
        self.add_output('u', shape=(T, 3*n))
        self.add_output('omega', shape=(T, 3*n))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n = self.options['n']
        x = inputs['x']

        for i in range(0, n):
            print(i)
            outputs['r'][:, 3*i:3*i+3] = x[i*18:i*18+3, :]
            outputs['theta'][:, 3*i:3*i+3] = x[i*18+3:i*18+6, :]
            outputs['F'][:, 3*i:3*i+3] = x[i*18+6:i*18+9, :]
            outputs['M'][:, 3*i:3*i+3] = x[i*18+9:i*18+12, :]
            outputs['u'][:, 3*i:3*i+3] = x[i*18+12:i*18+15, :]
            outputs['omega'][:, 3*i:3*i+3] = x[i*18+15:i*18+18, :]


class SymbolicBeam(ABC, om.Group):
    """
        Group that contains the symbolic beam functions that will be used for the structure. It will include
        the definition of the different beam constants, as well as the proper point distribution based on the joints
        and point loads.
    """

    def initialize(self):
        self.options.declare('name', types=str)
        self.options.declare("beam_definition", default=None)
        self.options.declare("constraints", default=[])
        self.options.declare('num_divisions', types=int)
        self.options.declare("applied_loads", default=[])
        self.options.declare("joints", default=[])
        self.options.declare('beam_type', values=('Fuselage', 'Wing'))
        self.options.declare('beam_bc', types=BoundaryType)
        self.options.declare('debug_flag', types=bool, default=False)
        self.options.declare('E', types=float)
        self.options.declare('rho', types=float)
        self.options.declare('G', types=float)
        self.options.declare('sigmaY', types=float)
        self.options.declare('num_timesteps', types=int)
        self.options.declare('beam_shape', types=BeamCS)  # To check what beam type is coming through

        # Optional stress definition, in case stress analysis is of importance
        self.options.declare('stress_definition', default=None)

        # Beam interpolated sections
        self.options.declare('num_interp_sections', default=0)
        self.options.declare('num_DvCs')
        self.options.declare('section_characteristics')

        # Beam axis node locations
        self.options.declare('r0')
        # Beam rotation sequence
        self.options.declare('seq')
        # Beam axis initial angles
        self.options.declare('th0')
        # Beam s0
        self.options.declare('delta_s0')
        # Beam node_lim and inter_node_lim
        self.options.declare('node_lim', types=np.ndarray)
        self.options.declare('inter_node_lim', types=np.ndarray)
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
        # Additional inputs at initialize
        self.declare_additional_beam_inputs()
        return

    def setup(self):
        beam_definition = self.options["beam_definition"]
        applied_loads = self.options["applied_loads"]
        joints = self.options["joints"]

        # Define basic beam parameters from containers:
        self.options["seq"] = beam_definition.orientation
        self.options['beam_bc'] = beam_definition.beam_bc
        self.options['name'] = beam_definition.beam_identifier
        self.options['E'] = beam_definition.E.magnitude
        self.options['G'] = beam_definition.G.magnitude
        self.options['rho'] = beam_definition.rho.magnitude
        self.options['sigmaY'] = beam_definition.sigmaY.magnitude
        self.options['num_timesteps'] = beam_definition.num_timesteps
        # Read sequence of points within the beam

        initial_points = beam_definition.beam_points.magnitude
        span = 0

        self.options['num_DvCs'] = initial_points.shape[1]  # Number of cross-sections with design variables

        # Get basic span value:

        if np.array_equal(self.options["seq"], np.array([3, 1, 2])):  # Fuselage beam
            span = initial_points[0, -1] - initial_points[0, 0]
            self.options['beam_type'] = 'Fuselage'
        if np.array_equal(self.options["seq"], np.array([1, 3, 2])):  # Wing beam
            span = initial_points[1, -1] - initial_points[1, 0]
            self.options['beam_type'] = 'Wing'

        num_total_sections = self.options['num_DvCs'] + (self.options['num_DvCs'] - 1) * self.options['num_interp_sections']

        self.options['section_characteristics'] = []

        for i in range(0, num_total_sections):
            if i % (self.options['num_interp_sections'] + 1) == 0:
                self.options['section_characteristics'].append(SectionType.CS)
            else:
                self.options['section_characteristics'].append(SectionType.INTERP)

        # Modify number of initial points with the specified number of interpolated sections:

        augmented_initial_points = np.empty([3, self.options['num_DvCs'] + (self.options['num_DvCs'] - 1) * self.options['num_interp_sections']])
        for i in range(self.options['num_DvCs'] - 1):
            augmented_initial_points[:, (self.options['num_interp_sections'] + 1) * i:(self.options['num_interp_sections'] + 1) * i + self.options['num_interp_sections'] + 2] = np.transpose(np.linspace(
                start=initial_points[:, i],
                stop=initial_points[:, i + 1],
                num=self.options['num_interp_sections'] + 2))

        initial_points = augmented_initial_points

        recorded_load_points = []

        # Start with loads:

        # These in case of point loads at the tip
        edge_force = np.zeros(3)
        edge_moment = np.zeros(3)

        if len(applied_loads) > 0:
            for a_load in applied_loads:
                found_lower_point = False
                # Check the load location and add a node there
                load_location = span * a_load.eta  # Either x or y, depending on the type of the beam
                if not (1.0 >= a_load.eta > 0.0):
                    raise ValueError('Load eta must be between 0 and 1')
                # Make point easily if eta is 1:
                if a_load.eta == 1.0:
                    # The tip load will not add nodes. Rather, it will add a boundary condition value
                    edge_force += np.squeeze(a_load.vector_force.magnitude)
                    edge_moment += np.squeeze(a_load.vector_moment.magnitude)
                    continue
                for i in range(0, initial_points.shape[1] - 1):
                    if np.array_equal(self.options["seq"], np.array([3, 1, 2])):  # Fuselage beam
                        current_span = abs(initial_points[0, i])
                        next_span = abs(initial_points[0, i + 1])
                    else:  # Wing beam
                        current_span = abs(initial_points[1, i])
                        next_span = abs(initial_points[1, i + 1])
                    if not found_lower_point and (load_location >= current_span and load_location < next_span):
                        found_lower_point = True
                        load_point = np.zeros(3)
                        if np.array_equal(self.options["seq"], np.array([3, 1, 2])):  # Fuselage beam
                            load_point[0] = load_location
                            span_percentage = (load_location - current_span) / (
                                    initial_points[0, i + 1] - initial_points[0, i])
                            load_point[1] = initial_points[1, i] + span_percentage * (
                                    initial_points[1, i + 1] - initial_points[1, i])
                            load_point[2] = initial_points[2, i] + span_percentage * (
                                    initial_points[2, i + 1] - initial_points[2, i])
                        else:  # Wing beam
                            load_point[1] = load_location
                            span_percentage = (load_location - current_span) / (
                                    initial_points[1, i + 1] - initial_points[1, i])
                            load_point[0] = initial_points[0, i] + span_percentage * (
                                    initial_points[0, i + 1] - initial_points[0, i])
                            load_point[2] = initial_points[2, i] + span_percentage * (
                                    initial_points[2, i + 1] - initial_points[2, i])
                        # Prepare interpolated load point
                        recorded_load_points.append(
                            load_point)  # To efficiently consider if the point has been created or not
                        initial_points = np.insert(initial_points, i + 1, load_point, axis=1)
                        self.options['section_characteristics'].insert(i+1, SectionType.FORCE)
                        if span_percentage > 0.0:  # only duplicate point IF that point did not exist before
                            initial_points = np.insert(initial_points, i + 1, load_point, axis=1)
                            self.options['section_characteristics'].insert(i + 1, SectionType.FORCE)
        # Then add the points of the joints
        if len(joints) > 0:
            for a_joint in joints:
                point_exists = False
                if a_joint.parent_beam == beam_definition.beam_identifier:
                    joint_eta = a_joint.parent_eta
                else:
                    if a_joint.child_beam == beam_definition.beam_identifier:
                        joint_eta = a_joint.child_eta
                    else:
                        raise ValueError('Joint not belonging to beam')
                # Check if the joint eta already was included with the loads
                for a_load in applied_loads:
                    if a_load.eta == joint_eta:
                        point_exists = True
                        break
                if point_exists:
                    break
                else:
                    # Time to create the point with regards to the joint
                    found_lower_point = False
                    joint_location = span * joint_eta  # Either x or y, depending on the type of the beam

                    # Check for additional points:
                    if np.array_equal(self.options["seq"], np.array([3, 1, 2])):  # Fuselage beam
                        if(initial_points.shape[1] - np.count_nonzero(initial_points[0, :] - joint_location) > 1):
                            break
                    else:
                        if (initial_points.shape[1] - np.count_nonzero(initial_points[1, :] - joint_location) > 1):
                            break

                    if not (1.0 >= joint_eta >= 0.0):
                        raise ValueError('Load eta must be between 0 and 1')
                    # Make point easily if eta is 1:
                    if joint_eta == 1.0:
                        last_slice = np.copy(initial_points[:, -1])
                        initial_points = np.transpose(np.vstack([np.transpose(initial_points), last_slice]))
                        self.options['section_characteristics'].append(SectionType.JOINT)
                        continue
                    for i in range(0, initial_points.shape[1]):
                        if np.array_equal(self.options["seq"], np.array([3, 1, 2])):  # Fuselage beam
                            current_span = abs(initial_points[0, i])
                            next_span = abs(initial_points[0, i + 1])
                        else:  # Wing beam
                            current_span = abs(initial_points[1, i])
                            next_span = abs(initial_points[1, i + 1])
                        if not found_lower_point and (joint_location >= current_span and joint_location < next_span):
                            found_lower_point = True
                            load_point = np.zeros(3)
                            if np.array_equal(self.options["seq"], np.array([3, 1, 2])):  # Fuselage beam
                                load_point[0] = joint_location
                                span_percentage = (joint_location - current_span) / (
                                        initial_points[0, i + 1] - initial_points[0, i])
                                load_point[1] = initial_points[1, i] + span_percentage * (
                                        initial_points[1, i + 1] - initial_points[1, i])
                                load_point[2] = initial_points[2, i] + span_percentage * (
                                        initial_points[2, i + 1] - initial_points[2, i])
                            else:  # Wing beam
                                load_point[1] = joint_location
                                span_percentage = (joint_location - current_span) / (
                                        initial_points[1, i + 1] - initial_points[1, i])
                                load_point[0] = initial_points[0, i] + span_percentage * (
                                        initial_points[0, i + 1] - initial_points[0, i])
                                load_point[2] = initial_points[2, i] + span_percentage * (
                                        initial_points[2, i + 1] - initial_points[2, i])
                            # Prepare interpolated load point
                            recorded_load_points.append(
                                load_point)  # To efficiently consider if the point has been created or not
                            initial_points = np.insert(initial_points, i + 1, load_point, axis=1)
                            self.options['section_characteristics'].insert(i + 1, SectionType.JOINT)
                            if span_percentage > 0.0:  # only duplicate point IF that point did not exist before
                                initial_points = np.insert(initial_points, i + 1, load_point, axis=1)
                                self.options['section_characteristics'].insert(i + 1, SectionType.JOINT)
        self.options["r0"] = initial_points  # Storing the augmented point structure
        self.options['num_divisions'] = initial_points.shape[1]
        # Calculate th0
        self.options['th0'] = calculate_th0(r0=self.options['r0'], seq=self.options["seq"])
        # Calculate delta_s0
        self.options['delta_s0'] = np.sqrt((self.options["r0"][0, 1:] - self.options["r0"][0, 0:-1]) ** 2 + (self.options["r0"][1, 1:] - self.options["r0"][1, 0:-1]) ** 2 + (
                    self.options["r0"][2, 1:] - self.options["r0"][2, 0:-1]) ** 2)

        # Correct th0 from zero length elements at the beginning:
        for i in range(self.options['delta_s0'].shape[0]):
            ds = self.options['delta_s0'][i]
            if ds == 0:
                self.options['th0'][:, i] = self.options['th0'][:, i+1]
        # Initial guess values
        x0 = np.zeros((18, self.options['num_divisions']))
        xDot0 = np.zeros((18, self.options['num_divisions']))
        x0[0:3, :] = self.options["r0"]
        x0[3:6, :] = self.options['th0']
        self.options['x0'] = np.resize(np.transpose(x0), (18 * self.options['num_divisions']))
        self.options['xDot0'] = np.resize(np.transpose(xDot0), (18 * self.options['num_divisions']))

        # Boundary conditions
        if self.options['beam_bc'] == BoundaryType.CANTILEVER:
            self.BC['root'][0:3] = self.options['r0'][0:3, 0]
            self.BC['root'][3:6] = self.options['th0'][0:3, 0]
        elif self.options['beam_bc'] == BoundaryType.FREEFREE:
            self.BC['root'][6:9] = np.zeros(3)
            self.BC['root'][9:12] = np.zeros(3)
        else:
            raise IOError

        # Adding the tip loads in the beam:
        self.BC['tip'][6:9] = edge_force
        self.BC['tip'][9:12] = edge_moment

        # K0a
        self.options['K0a'] = np.zeros((self.options['num_divisions'] - 1, 3, 3))
        K = np.zeros((self.options['num_divisions'], 3, 3))
        for i in range(self.options['num_divisions']):
            if self.options['beam_type'] == 'Fuselage':  # 312
                K[i, :, :] = np.asarray([
                    [cos(self.options['th0'][1, i]),
                     0,
                     -cos(self.options['th0'][0, i]) * sin(self.options['th0'][1, i])],
                    [0,
                     1,
                     sin(self.options['th0'][0, i])],
                    [sin(self.options['th0'][1, i]),
                     0, cos(self.options['th0'][0, i]) * cos(self.options['th0'][1, i])]
                ])
                if i >= 1:
                    self.options['K0a'][i - 1, :, :] = (K[i, :, :] + K[i - 1, :, :]) / 2
            elif self.options['beam_type'] == 'Wing':  # 132
                K[i, :, :] = np.array([
                    [cos(self.options['th0'][2, i]) * cos(self.options['th0'][1, i]),
                     0,
                     -sin(self.options['th0'][1, i])],
                    [-sin(self.options['th0'][2, i]),
                     1,
                     0],
                    [cos(self.options['th0'][2, i]) * sin(self.options['th0'][1, i]),
                     0,
                     cos(self.options['th0'][1, i])]
                ])
                if i >= 1:
                    self.options['K0a'][i - 1, :, :] = (K[i, :, :] + K[i - 1, :, :]) / 2
            else:
                raise IOError

    def create_symbolic_expressions(self,
                                    n,
                                    n_ea, n_ta, c_ta, c_ea,
                                    GKc, GKn, EA,
                                    mu,
                                    delta_r_CG,
                                    EIxx, EIzz, EIxz, GJ):
        D = SX.sym(self.options['name'] + 'D', 3, 3, n)
        for i in range(n):
            D[i][0, 0] = 0
            D[i][0, 1] = -n_ea[i]
            D[i][0, 2] = 0
            D[i][1, 0] = n_ta[0]
            D[i][1, 1] = 0
            D[i][1, 2] = -c_ta[0]
            D[i][2, 0] = 0
            D[i][2, 1] = c_ea[0]
            D[i][2, 2] = 0
        self.symbolic_expressions['D'] = D

        oneover = SX.sym(self.options['name'] + 'oneover', 3, 3, n)
        for i in range(n):
            oneover[i][0, 0] = 1 / GKc[i]
            oneover[i][0, 1] = 0
            oneover[i][0, 2] = 0
            oneover[i][1, 0] = 0
            oneover[i][1, 1] = 1 / EA[i]
            oneover[i][1, 2] = 0
            oneover[i][2, 0] = 0
            oneover[i][2, 1] = 0
            oneover[i][2, 2] = 1 / GKn[i]
        self.symbolic_expressions['oneover'] = oneover

        i_matrix = SX.sym(self.options['name'] + 'i_matrix', 3, 3, n - 1)
        for i in range(n - 1):
            i_matrix[i][0, 0] = mu[i]
            i_matrix[i][0, 1] = 0
            i_matrix[i][0, 2] = 0
            i_matrix[i][1, 0] = 0
            i_matrix[i][1, 1] = mu[i]
            i_matrix[i][1, 2] = 0
            i_matrix[i][2, 0] = 0
            i_matrix[i][2, 1] = 0
            i_matrix[i][2, 2] = mu[i]
        self.symbolic_expressions['mu'] = mu
        self.symbolic_expressions['i_matrix'] = i_matrix

        delta_r_CG_tilde = SX.sym(self.options['name'] + 'delta_r_CG_tilde', 3, 3, n - 1)
        for i in range(self.options['num_divisions'] - 1):
            drCG = delta_r_CG[i, :]
            delta_r_CG_tilde[i][0, 0] = 0
            delta_r_CG_tilde[i][0, 1] = -drCG[2]
            delta_r_CG_tilde[i][0, 2] = drCG[1]
            delta_r_CG_tilde[i][1, 0] = drCG[2]
            delta_r_CG_tilde[i][1, 1] = 0
            delta_r_CG_tilde[i][1, 2] = -drCG[0]
            delta_r_CG_tilde[i][2, 0] = -drCG[1]
            delta_r_CG_tilde[i][2, 1] = drCG[0]
            delta_r_CG_tilde[i][2, 2] = 0
        self.symbolic_expressions['delta_r_CG_tilde'] = delta_r_CG_tilde

        E = SX.sym(self.options['name'] + 'E', 3, 3, n)
        Einv = SX.sym(self.options['name'] + 'Einv', 3, 3, n)
        E_rot = SX.sym(self.options['name'] + 'Erot', 3, 3, n)
        T, Ta = CalcNodalT(self.options['th0'], self.options['seq'], self.options['num_divisions'])

        # E matrix where each 3x3 corresponds to cross section i at node i
        # E has the following form:
        # E = [EIcc  EIcs     EIcn;
        #      0     GJ       EIsn;
        #      0     GJ       EIsn;
        #      0     0        EInn]
        for i in range(n):
            E[i][0, 0] = EIxx[i]
            E[i][0, 1] = 0
            E[i][0, 2] = EIxz[i]
            E[i][1, 0] = 0
            E[i][1, 1] = GJ[i]
            E[i][1, 2] = 0
            E[i][2, 0] = EIxz[i]
            E[i][2, 1] = 0
            E[i][2, 2] = EIzz[i]

            if self.options['beam_type'] == 'Wing':  # Added in case the ticknesses are given on a wing wrt y axis

                E_rot[i] = mtimes(T[i], mtimes(E[i], transpose(T[i])))

                E[i][0, 0] = E_rot[i][0, 0]
                E[i][0, 1] = 0
                E[i][0, 2] = E_rot[i][0, 2]
                E[i][1, 0] = 0
                E[i][1, 1] = GJ[i]
                E[i][1, 2] = 0
                E[i][2, 0] = E_rot[i][2, 0]
                E[i][2, 1] = 0
                E[i][2, 2] = E_rot[i][2, 2]

            Einv[i] = inv(E[i])
        self.symbolic_expressions['Einv'] = Einv
        self.symbolic_expressions['E'] = E
        self.symbolic_expressions['EA'] = EA

        # region Loads
        self.symbolic_expressions['forces_dist'] = SX.sym(self.options['name'] + 'forces_dist', 3, n - 1)
        self.symbolic_expressions['moments_dist'] = SX.sym(self.options['name'] + 'moments_dist', 3, n - 1)

        self.symbolic_expressions['forces_conc'] = SX.sym(self.options['name'] + 'forces_conc', 3, n - 1)
        self.symbolic_expressions['moments_conc'] = SX.sym(self.options['name'] + 'moments_conc', 3, n - 1)
        # endregion
        return

    @abstractmethod
    def declare_additional_beam_inputs(self):
        return


class StaticDoublySymRectBeamRepresentation(SymbolicBeam):
    def initialize(self):
        # Initializing superclass
        super().initialize()
        pass

    def declare_additional_beam_inputs(self):
        return

    def setup(self):
        # Adding the different common beam properties
        super().setup()

        # Specifying the beam shape so that other sub-components know:
        self.options['beam_shape'] = BeamCS.RECTANGULAR

        # Create the symbolic function relating inputs and outputs
        self.create_symbolic_function()

        # Generating beam interface:
        self.beam_interface = BeamInterface(name='DoubleSymmetricBeamInterface', delta_s0=self.options['delta_s0'],
                                            symbolic_parent=self.symbolic_functions, beam_shape=self.options['beam_shape'],
                                            num_timesteps=self.options["num_timesteps"],
                                            debug_flag=self.options["debug_flag"], num_DvCs=self.options["num_DvCs"])
        # Adding beam interface
        self.add_subsystem('DoubleSymmetricBeamInterface', self.beam_interface)

        # Check whether the beam was passed a stress definition:
        if not (self.options['stress_definition'] is None):
            self.options['stress_definition'].options['num_DvCs'] = self.options["num_DvCs"]
            self.options['stress_definition'].options['num_divisions'] = self.options["num_divisions"]
            self.options['stress_definition'].options['num_timesteps'] = self.options["num_timesteps"]
            self.options['stress_definition'].options['r0'] = self.options["r0"]
            self.options['stress_definition'].options['th0'] = self.options["th0"]
            self.options['stress_definition'].options['seq'] = self.options["seq"]
            self.options['stress_definition'].options['beam_shape'] = self.options["beam_shape"]

            self.options['stress_definition'].options['E'] = self.options["E"]      # TODO: Do Material model
            self.options['stress_definition'].options['G'] = self.options["G"]

            self.options['stress_definition'].options['symbolic_variables'] = self.symbolic_expressions

            # Check if we have a stress constraint that is related to the stress model:
            if len(self.options["constraints"]) > 0:
                for a_constraint in self.options["constraints"]:
                    if isinstance(a_constraint, StrengthAggregatedConstraint):
                        a_constraint.options["num_divisions"] = self.options["num_divisions"]
                        a_constraint.options["beam_shape"] = self.options["beam_shape"]
                        a_constraint.options["stress_computation"] = self.options['stress_definition']
                        break
        return

    # region Common functions

    def create_symbolic_function(self):
        n = self.options['num_divisions']
        n_dvcs = self.options['num_DvCs']
        cs_r0 = self.options["beam_definition"].beam_points.magnitude  # Information belonging to the original beam
        r0 = self.options['r0']

        if np.array_equal(self.options["seq"], np.array([3, 1, 2])):  # Fuselage beam
            span_var = 0
        else:
            span_var = 1

        # Design Variables
        cs = SX.sym(self.options['name'] + 'cs', 2 * n_dvcs)
        self.symbolic_expressions['cs'] = cs

        # Construct augmented set
        h = SX.zeros(n, 1)
        w = SX.zeros(n, 1)

        # Lay up the augmented section set with the available cross-section information:
        J = 0
        for i in range(0, n):
            if self.options['section_characteristics'][i] == SectionType.CS:
                h[i] = cs[J]
                w[i] = cs[J+n_dvcs]
                J += 1
            else:
                if i == n-1 or J == n_dvcs:   # Last element
                    h[i] = cs[n_dvcs-1]
                    w[i] = cs[-1]
                else:
                    h_prev = cs[J-1]
                    w_prev = cs[J+n_dvcs-1]
                    h_next = cs[J]
                    w_next = cs[J + n_dvcs]

                    y_prev = cs_r0[span_var, J-1]
                    y_current = r0[span_var, i]
                    y_next = cs_r0[span_var, J]

                    h[i] = h_next*(y_current-y_prev)/(y_next-y_prev) + h_prev*(1-(y_current-y_prev)/(y_next-y_prev))
                    w[i] = w_next*(y_current-y_prev)/(y_next-y_prev) + w_prev*(1-(y_current-y_prev)/(y_next-y_prev))
                    pass


        # Generating the corner point function:
        corner_points = SX.sym('cpoint', 8, n)

        corner_points[0, :] = -w / 2
        corner_points[1, :] = h / 2
        corner_points[2, :] = w / 2
        corner_points[3, :] = h / 2
        corner_points[4, :] = w / 2
        corner_points[5, :] = -h / 2
        corner_points[6, :] = -w / 2
        corner_points[7, :] = -h / 2

        self.symbolic_expressions['corner_points'] = corner_points

        self.symbolic_expressions['h'] = h
        self.symbolic_expressions['w'] = w

        self.create_symbolic_states()

        # region Offsets from beam axis
        n_ea = SX.zeros(n)
        c_ea = SX.zeros(n)
        n_ta = SX.zeros(n)
        c_ta = SX.zeros(n)
        # endregion

        # region Bending and Torsional Stiffness
        EIxx = SX.sym(self.options['name'] + 'EIxx', n)
        EIzz = SX.sym(self.options['name'] + 'EIzz', n)
        EIxz = SX.sym(self.options['name'] + 'EIxz', n)
        GJ = SX.sym(self.options['name'] + 'GJ', n)
        for i in range(n):
            EIxx[i] = self.options['E'] * (1 / 12) * (w[i] * h[i] ** 3)
            EIzz[i] = self.options['E'] * (1 / 12) * (w[i] ** 3 * h[i])
            EIxz[i] = 0
            J = (w[i] * h[i] ** 3) * (2 / 9) * (1 / (1 + (h[i] / w[i]) ** 2))
            GJ[i] = self.options['G'] * J

        # endregion

        # region Axial and Shear Stiffness
        GKn = (self.options['G']) / 1.2 * np.ones(n)
        GKc = (self.options['G']) / 1.2 * np.ones(n)
        A = SX.sym(self.options['name'] + 'A', n)
        EA = SX.sym(self.options['name'] + 'EA', n)
        for i in range(n):
            A[i] = h[i] * w[i]
            EA[i] = self.options['E'] * A[i]
        # endregion

        # region Mass properties
        mu = SX.sym(self.options['name'] + 'mu', n - 1)
        for i in range(n - 1):
            A1 = A[i]
            A2 = A[i + 1]
            mu[i] = self.options['rho'] * 1 / 3 * (A1 + A2 + sqrt(A1 * A2))
        # endregion

        # region delta_r_CG
        # column i is the position of the CG on the cross-section i, relative to the csn origin, expressed in xyz
        delta_r_CG = np.zeros([self.options['num_divisions'] - 1, 3])
        # endregion

        self.create_symbolic_expressions(n=n,
                                         n_ea=n_ea, n_ta=n_ta, c_ta=c_ta, c_ea=c_ea,
                                         GKc=GKc, GKn=GKn, EA=EA,
                                         mu=mu,
                                         delta_r_CG=delta_r_CG,
                                         EIxx=EIxx, EIzz=EIzz, EIxz=EIxz, GJ=GJ)

        self.symbolic_functions['D'] = Function(self.options['name'] + "D", [cs], self.symbolic_expressions['D'])
        self.symbolic_functions['oneover'] = Function(self.options['name'] + "oneover", [cs],
                                                      self.symbolic_expressions['oneover'])
        self.symbolic_functions['mu'] = Function(self.options['name'] + "mu", [cs], [self.symbolic_expressions['mu']])
        self.symbolic_functions['i_matrix'] = Function(self.options['name'] + "i_matrix", [cs],
                                                       self.symbolic_expressions['i_matrix'])
        self.symbolic_functions['delta_r_CG_tilde'] = Function(self.options['name'] + "mu", [cs],
                                                               self.symbolic_expressions['delta_r_CG_tilde'])
        self.symbolic_functions['Einv'] = Function(self.options['name'] + "Einv", [cs],
                                                   self.symbolic_expressions['Einv'])

        self.symbolic_functions['corner_points'] = Function(self.options['name'] + "stress_rec", [cs],
                                                   [self.symbolic_expressions['corner_points']])
        self.create_mass_function()
        return

    # endregion

    def create_symbolic_states(self):
        n = self.options['num_divisions']
        T = self.options['num_timesteps']
        # State Variables
        self.symbolic_expressions['x'] = SX.sym(self.options['name'] + 'x', 18 * n, T+1)
        self.symbolic_expressions['xDot'] = SX.sym(self.options['name'] + 'xDot', 18 * n, T+1)
        self.symbolic_expressions['x_slice'] = self.symbolic_expressions['x'][:, 0]
        self.symbolic_expressions['xDot_slice'] = self.symbolic_expressions['xDot'][:, 0]
        return

    def create_mass_function(self):
        SCALE_FACT = 100000
        n = self.options['num_divisions']
        cs = self.symbolic_expressions['cs']
        h = self.symbolic_expressions['h']
        w = self.symbolic_expressions['w']
        s_0 = SX.sym(self.options['name'] + 's_0', n - 1)

        A = h * w

        mu = SX.sym(self.options['name'] + 'mu', n - 1)

        for i in range(0, n - 1):
            A1 = A[i]
            A2 = A[i + 1]
            mu[i] = self.options['rho'] * (1 / 3) * (A1 + A2 + sqrt(A1 * A2))

        self.symbolic_expressions['total_mass'] = sum1(mu * s_0) / SCALE_FACT
        self.symbolic_functions['mass'] = Function('mass', [cs, s_0], [self.symbolic_expressions['total_mass']])
        self.symbolic_expressions['mass_jac'] = jacobian(self.symbolic_expressions['total_mass'], cs)
        self.symbolic_functions['mass_jac'] = Function('mass_jac', [cs, s_0], [self.symbolic_expressions['mass_jac']])


class BoxBeamRepresentation(SymbolicBeam):
    def initialize(self):
        # Initializing superclass
        super().initialize()

    def declare_additional_beam_inputs(self):
        self.options.declare('corner_points')
        return

    def setup(self):
        # Adding the different common beam properties
        super().setup()

        # Specifying the beam shape so that other sub-components know:
        self.options['beam_shape'] = BeamCS.BOX

        # Create the symbolic function relating inputs and outputs
        self.create_symbolic_function()

        # Generating beam interface:
        self.beam_interface = BeamInterface(name='BoxBeamInterface', delta_s0=self.options['delta_s0'],
                                            symbolic_parent=self.symbolic_functions, beam_shape=self.options['beam_shape'],
                                            num_timesteps=self.options["num_timesteps"],
                                            debug_flag=self.options["debug_flag"], num_DvCs=self.options["num_DvCs"])
        # Adding beam interface
        self.add_subsystem('BoxBeamInterface', self.beam_interface)

        # Check whether the beam was passed a stress definition:
        if not (self.options['stress_definition'] is None):
            self.options['stress_definition'].options['num_DvCs'] = self.options["num_DvCs"]
            self.options['stress_definition'].options['num_divisions'] = self.options["num_divisions"]
            self.options['stress_definition'].options['num_timesteps'] = self.options["num_timesteps"]
            self.options['stress_definition'].options['r0'] = self.options["r0"]
            self.options['stress_definition'].options['th0'] = self.options["th0"]
            self.options['stress_definition'].options['seq'] = self.options["seq"]
            self.options['stress_definition'].options['beam_shape'] = self.options["beam_shape"]

            self.options['stress_definition'].options['E'] = self.options["E"]      # TODO: Do Material model
            self.options['stress_definition'].options['G'] = self.options["G"]

            self.options['stress_definition'].options['symbolic_variables'] = self.symbolic_expressions

            # Check if we have a stress constraint that is related to the stress model:
            if len(self.options["constraints"]) > 0:
                for a_constraint in self.options["constraints"]:
                    if isinstance(a_constraint, StrengthAggregatedConstraint):
                        a_constraint.options["num_divisions"] = self.options["num_divisions"]
                        a_constraint.options["beam_shape"] = self.options["beam_shape"]
                        a_constraint.options["stress_computation"] = self.options['stress_definition']
                        break
            pass
        return

    # region Common functions

    def create_symbolic_function(self):
        n = self.options['num_divisions']
        n_dvcs = self.options['num_DvCs']

        cs_r0 = self.options["beam_definition"].beam_points.magnitude  # Information belonging to the original beam
        r0 = self.options['r0']

        if np.array_equal(self.options["seq"], np.array([3, 1, 2])):  # Fuselage beam
            span_var = 0
        else:
            span_var = 1

        # Design Variables
        cs = SX.sym(self.options['name'] + 'cs', 6 * n_dvcs)
        self.symbolic_expressions['cs'] = cs

        # Isolate important variables from cs

        h_seed = cs[0:n_dvcs]
        w_seed = cs[n_dvcs:2 * n_dvcs]
        t_left_seed = cs[2 * n_dvcs:3 * n_dvcs]
        t_top_seed = cs[3 * n_dvcs:4 * n_dvcs]
        t_right_seed = cs[4 * n_dvcs:5 * n_dvcs]
        t_bot_seed = cs[5 * n_dvcs:6 * n_dvcs]

        # Construct augmented set
        h = SX.zeros(n, 1)
        w = SX.zeros(n, 1)
        t_left = SX.zeros(n, 1)
        t_top = SX.zeros(n, 1)
        t_right = SX.zeros(n, 1)
        t_bot = SX.zeros(n, 1)

        # Lay up the augmented section set with the available cross-section information:
        J = 0
        for i in range(0, n):
            if self.options['section_characteristics'][i] == SectionType.CS:
                h[i] = h_seed[J]
                w[i] = w_seed[J]
                t_left[i] = t_left_seed[J]
                t_top[i] = t_top_seed[J]
                t_right[i] = t_right_seed[J]
                t_bot[i] = t_bot_seed[J]
                J += 1
            else:
                if i == n - 1 or J == n_dvcs:  # Last element
                    h[i] = h_seed[-1]
                    w[i] = w_seed[-1]
                    t_left[i] = t_left_seed[-1]
                    t_top[i] = t_top_seed[-1]
                    t_right[i] = t_right_seed[-1]
                    t_bot[i] = t_bot_seed[-1]
                else:
                    h_prev = h_seed[J-1]
                    w_prev = w_seed[J-1]
                    t_left_prev = t_left_seed[J-1]
                    t_top_prev = t_top_seed[J-1]
                    t_right_prev = t_right_seed[J-1]
                    t_bot_prev = t_bot_seed[J-1]

                    h_next = h_seed[J]
                    w_next = w_seed[J]
                    t_left_next = t_left_seed[J]
                    t_top_next = t_top_seed[J]
                    t_right_next = t_right_seed[J]
                    t_bot_next = t_bot_seed[J]

                    y_prev = cs_r0[span_var, J - 1]
                    y_current = r0[span_var, i]
                    y_next = cs_r0[span_var, J]

                    h[i] = h_next * (y_current - y_prev) / (y_next - y_prev) + h_prev * (
                                1 - (y_current - y_prev) / (y_next - y_prev))
                    w[i] = w_next * (y_current - y_prev) / (y_next - y_prev) + w_prev * (
                                1 - (y_current - y_prev) / (y_next - y_prev))

                    t_left[i] = t_left_next * (y_current - y_prev) / (y_next - y_prev) + t_left_prev * (
                            1 - (y_current - y_prev) / (y_next - y_prev))
                    t_top[i] = t_top_next * (y_current - y_prev) / (y_next - y_prev) + t_top_prev * (
                            1 - (y_current - y_prev) / (y_next - y_prev))

                    t_right[i] = t_right_next * (y_current - y_prev) / (y_next - y_prev) + t_right_prev * (
                            1 - (y_current - y_prev) / (y_next - y_prev))
                    t_bot[i] = t_bot_next * (y_current - y_prev) / (y_next - y_prev) + t_bot_prev * (
                            1 - (y_current - y_prev) / (y_next - y_prev))
                    pass

        # Generating the corner point function:
        corner_points = SX.sym('cpoint', 8, n)

        corner_points[0, :] = -w / 2
        corner_points[1, :] = h / 2
        corner_points[2, :] = w / 2
        corner_points[3, :] = h / 2
        corner_points[4, :] = w / 2
        corner_points[5, :] = -h / 2
        corner_points[6, :] = -w / 2
        corner_points[7, :] = -h / 2

        self.symbolic_expressions['corner_points'] = corner_points

        self.symbolic_expressions['h'] = h
        self.symbolic_expressions['w'] = w
        self.symbolic_expressions['t_left'] = t_left
        self.symbolic_expressions['t_top'] = t_top
        self.symbolic_expressions['t_right'] = t_right
        self.symbolic_expressions['t_bot'] = t_bot

        self.create_symbolic_states()

        # region Box-beam cross-section 4 parts
        # Segments 1:
        A_sect1 = t_top * w
        rho_sect1 = self.options['rho']
        E_sect1 = self.options['E']
        cg_sect1_y = 0
        cg_sect1_z = (h - t_top) / 2
        Ixx_sect1 = (w * t_top ** 3) / 12
        Izz_sect1 = (t_top * w ** 3) / 12
        # Segments 2:
        A_sect2 = t_left * (h - t_top - t_bot)
        rho_sect2 = self.options['rho']
        E_sect2 = self.options['E']
        cg_sect2_y = (t_left - w) / 2
        cg_sect2_z = h / 2 - t_top - (h - t_top - t_bot) / 2
        Ixx_sect2 = (t_left * (h - t_top - t_bot) ** 3) / 12
        Izz_sect2 = ((h - t_top - t_bot) * t_left ** 3) / 12
        # Segments 3:
        A_sect3 = t_bot * w
        rho_sect3 = self.options['rho']
        E_sect3 = self.options['E']
        cg_sect3_y = 0
        cg_sect3_z = (t_bot - h) / 2
        Ixx_sect3 = (w * t_bot ** 3) / 12
        Izz_sect3 = (t_bot * w ** 3) / 12
        # Segments 4:
        A_sect4 = t_right * (h - t_top - t_bot)
        E_sect4 = self.options['E']
        rho_sect4 = self.options['rho']
        cg_sect4_y = (w - t_right) / 2
        cg_sect4_z = h / 2 - t_top - (h - t_top - t_bot) / 2
        Ixx_sect4 = (t_right * (h - t_top - t_bot) ** 3) / 12
        Izz_sect4 = ((h - t_top - t_bot) * t_right ** 3) / 12

        self.symbolic_expressions['A_segments'] = {'Top': A_sect1,
                                                   'Left': A_sect2,
                                                   'Bot': A_sect3,
                                                   'Right': A_sect4}
        # endregion

        # region Offsets from beam axis
        e_cg_x = (cg_sect1_y * A_sect1 * E_sect1 +
                  cg_sect2_y * A_sect2 * E_sect2 +
                  cg_sect3_y * A_sect3 * E_sect3 +
                  cg_sect4_y * A_sect4 * E_sect4) / \
                 (A_sect1 * E_sect1 +
                  A_sect2 * E_sect2 +
                  A_sect3 * E_sect3 +
                  A_sect4 * E_sect4)
        e_cg_z = (cg_sect1_z * A_sect1 * E_sect1 +
                  cg_sect2_z * A_sect2 * E_sect2 +
                  cg_sect3_z * A_sect3 * E_sect3 +
                  cg_sect4_z * A_sect4 * E_sect4) / \
                 (A_sect1 * E_sect1 +
                  A_sect2 * E_sect2 +
                  A_sect3 * E_sect3 +
                  A_sect4 * E_sect4)
        n_ea = e_cg_z
        c_ea = e_cg_x
        n_ta = SX.zeros(n)
        c_ta = SX.zeros(n)
        # endregion

        # region Bending and Torsional Stiffness
        # Parallel axis theorem
        Ixx = Ixx_sect1 + A_sect1 * (cg_sect1_z - e_cg_z) ** 2 + \
              Ixx_sect2 + A_sect2 * (cg_sect2_z - e_cg_z) ** 2 + \
              Ixx_sect3 + A_sect3 * (cg_sect3_z - e_cg_z) ** 2 + \
              Ixx_sect4 + A_sect4 * (cg_sect4_z - e_cg_z) ** 2
        Izz = Izz_sect1 + A_sect1 * (cg_sect1_y - e_cg_x) ** 2 + \
              Izz_sect2 + A_sect2 * (cg_sect2_y - e_cg_x) ** 2 + \
              Izz_sect3 + A_sect3 * (cg_sect3_y - e_cg_x) ** 2 + \
              Izz_sect4 + A_sect4 * (cg_sect4_y - e_cg_x) ** 2
        Ixz = -(
                A_sect1 * (cg_sect1_y - e_cg_x) * (cg_sect1_z - e_cg_z) +
                A_sect2 * (cg_sect2_y - e_cg_x) * (cg_sect2_z - e_cg_z) +
                A_sect3 * (cg_sect3_y - e_cg_x) * (cg_sect3_z - e_cg_z) +
                A_sect4 * (cg_sect4_y - e_cg_x) * (cg_sect4_z - e_cg_z)
        )
        J = 2 * (
                ((h - t_top / 2 - t_bot / 2) * (w - t_right / 2 - t_left / 2)) ** 2
        ) / \
            (
                    ((w - t_right / 2 - t_left / 2) / (0.5 * t_top + 0.5 * t_bot)) +
                    ((h - t_top / 2 - t_bot / 2) / (0.5 * t_right + 0.5 * t_left))
            )

        EIxx = self.options['E'] * Ixx
        EIzz = self.options['E'] * Izz
        EIxz = self.options['E'] * Ixz
        GJ = self.options['G'] * J
        # endregion

        # region Axial and Shear Stiffness
        GKn = self.options['G'] / 1.2 * np.ones(n)
        GKc = self.options['G'] / 1.2 * np.ones(n)
        EA = E_sect1 * A_sect1 + E_sect2 * A_sect2 + E_sect3 * A_sect3 + E_sect4 * A_sect4
        self.symbolic_expressions['EA'] = EA
        # endregion

        # region Mass properties
        A = A_sect1 + A_sect2 + A_sect3 + A_sect4
        mu = SX.sym('mu', n - 1)
        for i in range(n - 1):
            A1 = A[i]
            A2 = A[i + 1]
            mu[i] = self.options['rho'] * 1 / 3 * (A1 + A2 + sqrt(A1 * A2))
        # endregion

        # region delta_r_CG
        m_cg_x = (cg_sect1_y * A_sect1 * rho_sect1 +
                  cg_sect2_y * A_sect2 * rho_sect2 +
                  cg_sect3_y * A_sect3 * rho_sect3 +
                  cg_sect4_y * A_sect4 * rho_sect4) / \
                 (A_sect1 * rho_sect1 +
                  A_sect2 * rho_sect2 +
                  A_sect3 * rho_sect3 +
                  A_sect4 * rho_sect4)
        m_cg_z = (cg_sect1_z * A_sect1 * rho_sect1 +
                  cg_sect2_z * A_sect2 * rho_sect2 +
                  cg_sect3_z * A_sect3 * rho_sect3 +
                  cg_sect4_z * A_sect4 * rho_sect4) / \
                 (A_sect1 * rho_sect1 +
                  A_sect2 * rho_sect2 +
                  A_sect3 * rho_sect3 +
                  A_sect4 * rho_sect4)
        # column i is the position of the CG on the cross-section i, relative to the csn origin, expressed in xyz
        delta_r_CG = SX.zeros(n - 1, 3)
        for i in range(n - 1):
            delta_r_CG[i, 1] = (m_cg_x[i] + m_cg_x[i + 1]) / 2
            delta_r_CG[i, 2] = (m_cg_z[i] + m_cg_z[i + 1]) / 2
        # endregion

        # region Shear terms
        A_inner = ((h - t_top / 2 - t_bot / 2) * (w - t_right / 2 - t_left / 2))

        Q_max_z = w * t_top * (h / 2 - e_cg_z - t_top / 2) + \
                  0.5 * t_right * (h / 2 - t_top - e_cg_z) ** 2 + \
                  0.5 * t_left * (h / 2 - t_top - e_cg_z) ** 2
        Q_max_x = h * t_left * (w / 2 - e_cg_x - t_left / 2) + \
                  0.5 * t_top * (w / 2 - t_left - e_cg_x) ** 2 + \
                  0.5 * t_bot * (w / 2 - t_left - e_cg_x) ** 2
        self.symbolic_expressions['A_inner'] = A_inner
        self.symbolic_expressions['Q_max_z'] = Q_max_z
        self.symbolic_expressions['Q_max_x'] = Q_max_x
        # endregion

        self.create_symbolic_expressions(n=n,
                                         n_ea=n_ea, n_ta=n_ta, c_ta=c_ta, c_ea=c_ea,
                                         GKc=GKc, GKn=GKn, EA=EA,
                                         mu=mu,
                                         delta_r_CG=delta_r_CG,
                                         EIxx=EIxx, EIzz=EIzz, EIxz=EIxz, GJ=GJ)

        self.symbolic_functions['D'] = Function(self.options['name'] + "D", [cs], self.symbolic_expressions['D'])
        self.symbolic_functions['oneover'] = Function(self.options['name'] + "oneover", [cs],
                                                      self.symbolic_expressions['oneover'])
        self.symbolic_functions['mu'] = Function(self.options['name'] + "mu", [cs], [self.symbolic_expressions['mu']])
        self.symbolic_functions['i_matrix'] = Function(self.options['name'] + "i_matrix", [cs],
                                                       self.symbolic_expressions['i_matrix'])
        self.symbolic_functions['delta_r_CG_tilde'] = Function(self.options['name'] + "mu", [cs],
                                                               self.symbolic_expressions['delta_r_CG_tilde'])
        self.symbolic_functions['Einv'] = Function(self.options['name'] + "Einv", [cs],
                                                   self.symbolic_expressions['Einv'])

        self.symbolic_functions['E'] = Function("E", [cs], self.symbolic_expressions['E'])

        self.symbolic_functions['EA'] = Function("EA", [cs], [EA])

        self.symbolic_functions['corner_points'] = Function(self.options['name'] + "stress_rec", [cs],
                                                   [self.symbolic_expressions['corner_points']])

        self.create_mass_function()
        return

    # endregion

    def create_symbolic_states(self):
        n = self.options['num_divisions']
        T = self.options['num_timesteps']
        # State Variables
        self.symbolic_expressions['x'] = SX.sym(self.options['name'] + 'x', 18 * n, T + 1)
        self.symbolic_expressions['xDot'] = SX.sym(self.options['name'] + 'xDot', 18 * n, T + 1)
        self.symbolic_expressions['x_slice'] = self.symbolic_expressions['x'][:, 0]
        self.symbolic_expressions['xDot_slice'] = self.symbolic_expressions['xDot'][:, 0]
        return

    def create_mass_function(self):
        SCALE_FACT = 100000

        cs = self.symbolic_expressions['cs']

        h = self.symbolic_expressions['h']
        w = self.symbolic_expressions['w']
        t_top = self.symbolic_expressions['t_top']
        t_left = self.symbolic_expressions['t_left']
        t_bot = self.symbolic_expressions['t_bot']
        t_right = self.symbolic_expressions['t_right']

        n_nodes = self.options['num_divisions']

        s_0 = SX.sym('s_0', n_nodes - 1)

        mu = SX.sym('mu', n_nodes - 1)

        sum_mu = SX.zeros(1)
        for i in range(0, n_nodes - 1):
            A1 = (w[i] * h[i] - ((w[i] - t_right[i] - t_left[i]) * (h[i] - t_top[i] - t_bot[i])))
            A2 = (w[i + 1] * h[i + 1] - ((w[i + 1] - t_right[i + 1] - t_left[i + 1]) * (h[i + 1] - t_top[i + 1] - t_bot[i + 1])))
            mu[i] = s_0[i] * self.options['rho'] * (1 / 3) * (A1 + A2 + sqrt(A1 * A2))
            sum_mu = sum_mu + mu[i]
        # TODO: Look at why 2*scale gives proper weight
        self.symbolic_expressions['total_mass'] = sum_mu / SCALE_FACT
        self.symbolic_functions['mass'] = Function('mass', [cs, s_0], [self.symbolic_expressions['total_mass']])
        self.symbolic_expressions['mass_jac'] = jacobian(self.symbolic_expressions['total_mass'], cs)
        self.symbolic_functions['mass_jac'] = Function('mass_jac', [cs, s_0], [self.symbolic_expressions['mass_jac']])