from openasemdao.structures.beam.beam import StaticDoublySymRectBeamRepresentation, BoxBeamRepresentation, MassCombiner
from openasemdao.structures.inputs.inputs import BeamDefinition, PointLoadDefinition, JointDefinition
from openasemdao.structures.utils.utils import unique
import openmdao.api as om
from openasemdao.structures.beam.constraints import StrengthAggregatedConstraint, SpecificStrengthAggregatedConstraint
from openasemdao.structures.beam.stress_models import EulerBernoulliStressModel,  BasicEulerBernoulliStressModel
from openasemdao.structures.beam.stickmodel import BeamStickModel, StickModelFeeder, StickModelDemultiplexer
from openasemdao import Q_
import numpy as np
from openasemdao.structures.utils.beam_categories import BoundaryType


def test_zero_element_generation():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 12
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    beam_points[1, :] = np.linspace(0, 15, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    # Example of a complete input for the beam definition:
    rect_beam = BeamDefinition(identifier='MainWing', points=beam_point_input, orientation=np.array([1, 3, 2]), E=Q_(70e9, 'pascal'),
                               G=Q_(30e9, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), bc=BoundaryType.CANTILEVER, num_timesteps=1)

    # Test loads for the geometry
    loads = []
    etas = []
    label1 = 'load1'
    f1 = Q_(np.array([0, 0, 1000]), 'newton')
    m1 = Q_(np.array([0, 0, 10000]), 'newton*meter')
    eta1 = 0.25
    etas.append(eta1)
    load1 = PointLoadDefinition(label1, eta1, f1, m1)
    loads.append(load1)
    label2 = 'load2'
    f2 = Q_(np.array([0, 0, 2000]), 'newton')
    m2 = Q_(np.array([0, 0, 20000]), 'newton*meter')
    eta2 = 0.5
    etas.append(eta2)
    load2 = PointLoadDefinition(label2, eta2, f2, m2)
    loads.append(load2)
    label3 = 'load3'
    f3 = Q_(np.array([0, 0, 4000]), 'newton')
    m3 = Q_(np.array([0, 0, 50000]), 'newton*meter')
    eta3 = 0.75
    etas.append(eta3)
    load3 = PointLoadDefinition(label3, eta3, f3, m3)
    loads.append(load3)

    # Test joints for the geometry
    joints = []
    eta4 = 0.6
    joint1 = JointDefinition('inner_joint', 'MainWing', eta4, 'FuseWing', 0.3)
    etas.append(eta4)
    joints.append(joint1)
    eta5 = 0.8
    joint2 = JointDefinition('outer_joint', 'MainWing', eta5, 'Strut', 0.8)
    etas.append(eta5)
    joints.append(joint2)

    num_zero_elements = len(unique(etas))

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints, constraints=[str_constraint], stress_definition=stress_model,
                                                        num_interp_sections=0)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

    prob = om.Problem(model)
    prob.setup()

    # Test the actual result of the solution:
    r0 = sample_beam.options['r0']
    augmented_predicted_size = r0.shape[1]

    augmented_actual_size = 2 * num_zero_elements + n_sections_before_joints_loads

    np.testing.assert_equal(augmented_predicted_size, augmented_actual_size)


def test_th0_generation():
    model = om.Group()
    # Generate a sequence of points for the beam (aircraft base from literature)
    beam_points = np.array([[12.5676660000000, 12.7764540000000, 12.9854960000000, 13.1391660000000, 13.2930900000000,
                             13.4470140000000, 13.6006840000000, 13.7353040000000, 13.8696700000000, 14.0482320000000,
                             14.2267940000000, 14.4053560000000, 14.5839180000000, 14.7642580000000, 14.9443440000000,
                             15.1246840000000, 15.3047700000000, 15.4787600000000, 15.6530040000000, 15.8269940000000,
                             16.0009840000000, 16.1927540000000, 16.3842700000000, 16.5757860000000, 16.7675560000000,
                             16.9237660000000, 17.0802300000000, 17.2542200000000, 17.4282100000000, 17.6022000000000,
                             17.7764440000000, 17.9504340000000, 18.1244240000000, 18.2844440000000, 18.4444640000000,
                             18.5521600000000, 18.6596020000000],
                            [0, 0.454660000000000, 0.909320000000000, 1.24409200000000, 1.57886400000000,
                             1.91363600000000, 2.24840800000000, 2.54127000000000, 2.83387800000000, 3.22249800000000,
                             3.61111800000000, 3.99973800000000, 4.38835800000000, 4.78053400000000, 5.17296400000000,
                             5.56514000000000, 5.95731600000000, 6.33603000000000, 6.71499800000000, 7.09371200000000,
                             7.47268000000000, 7.88974800000000, 8.30681600000000, 8.72363000000000, 9.14069800000000,
                             9.48105800000000, 9.82116400000000, 10.1998780000000, 10.5788460000000, 10.9575600000000,
                             11.3365280000000, 11.7152420000000, 12.0942100000000, 12.4424440000000, 12.7906780000000,
                             13.0248660000000, 13.2588000000000],
                            [1.25526800000000, 1.26517400000000, 1.27533400000000, 1.28270000000000, 1.28981200000000,
                             1.29717800000000, 1.30454400000000, 1.31114800000000, 1.31749800000000, 1.32613400000000,
                             1.33451600000000, 1.34315200000000, 1.35153400000000, 1.36093200000000, 1.37033000000000,
                             1.37947400000000, 1.38887200000000, 1.39776200000000, 1.40690600000000, 1.41579600000000,
                             1.42468600000000, 1.43459200000000, 1.44449800000000, 1.45440400000000, 1.46431000000000,
                             1.47243800000000, 1.48056600000000, 1.48945600000000, 1.49860000000000, 1.50749000000000,
                             1.51638000000000, 1.52552400000000, 1.53441400000000, 1.54279600000000, 1.55092400000000,
                             1.55651200000000, 1.56210000000000]])
    beam_point_input = Q_(beam_points, 'meter')
    rect_beam = BeamDefinition('MainWing', beam_point_input, np.array([1, 3, 2]), E=Q_(70e9, 'pascal'),
                               G=Q_(30e9, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1)

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=[], joints=[], constraints=[])

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

    prob = om.Problem(model)
    prob.setup()

    # Test the actual result of the solution:
    th0 = sample_beam.options['th0']

    th0_actual = [[0.0217842629058000, 0.0223426502001360, 0.0219994851300184, 0.0212411144263057, 0.0219994851300184,
                   0.0219994851300184, 0.0225460488926736, 0.0216979830928244, 0.0222185653267193, 0.0215652837681581,
                   0.0222185653267193, 0.0215652837681581, 0.0239591450094641, 0.0239436434004407, 0.0233118383821370,
                   0.0239591450094641, 0.0234698681147238, 0.0241240054405991, 0.0234698681147239, 0.0234541434102709,
                   0.0237470576910110, 0.0237470576910114, 0.0237615233356608, 0.0237470576910114, 0.0238760590021121,
                   0.0238938834800200, 0.0234698681147239, 0.0241240054405991, 0.0234698681147239, 0.0234541434102709,
                   0.0241401789064097, 0.0234541434102705, 0.0240653750466704, 0.0233363901239838, 0.0238566444165834,
                   0.0238825375532485, 0.0238825375532485],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0],
                  [-0.430403034326643, -0.430859555188469, -0.430245138613573, -0.430877762049916, -0.430871538863014,
                   -0.430245138613580, -0.430770239108869, -0.430390131022733, -0.430613452962408, -0.430618878751699,
                   -0.430613452962402, -0.430618878751698, -0.430901492292724, -0.430121543241721, -0.430907299664680,
                   -0.430366841483884, -0.430558095960167, -0.430851263898784, -0.430558095960168, -0.430303993596863,
                   -0.430870859025579, -0.430368108620078, -0.430599008802363, -0.430870859025579, -0.430168030999965,
                   -0.431067338532504, -0.430558095960161, -0.430297951067347, -0.430558095960168, -0.430857312174402,
                   -0.430552042683969, -0.430303993596858, -0.430631477046549, -0.430638032269378, -0.430923292926869,
                   -0.430438954590719, -0.430438954590719]]

    np.testing.assert_almost_equal(th0, th0_actual, decimal=2)


def test_axial_stress_computation():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 10
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    beam_points[1, :] = np.linspace(0, 15, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')
    rect_beam = BeamDefinition('MainWing', beam_point_input, np.array([1, 3, 2]), E=Q_(70e9, 'pascal'),
                               G=Q_(30e9, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1)

    # Test loads for the geometry
    loads = []

    # Test joints for the geometry
    joints = []

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints,
                                                        constraints=[str_constraint], stress_definition=stress_model, debug_flag=True, num_interp_sections=4)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

    # Add the stress definition to the general model
    model.add_subsystem('DoubleSymmetricStressModel', sample_beam.options['stress_definition'])
    # Add the constraint:
    if len(sample_beam.options['constraints']) > 0:
        for a_constraint in sample_beam.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('DoubleSymmetricStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('DoubleSymmetricStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('DoubleSymmetricStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('DoubleSymmetricStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface

    model.connect('RectBeam.DoubleSymmetricBeamInterface.corner_points', 'DoubleSymmetricStressModel.corner_points')
    model.connect('RectBeam.DoubleSymmetricBeamInterface.cs_out', 'DoubleSymmetricStressModel.cs')

    prob = om.Problem(model)
    prob.setup()

    # Test the actual result of the solution:
    r0 = sample_beam.options['r0']
    th0 = sample_beam.options['th0']

    F = np.zeros((3, r0.shape[1]))

    M = np.zeros((3, r0.shape[1]))

    M[0, :] = np.linspace(1000000, 0, r0.shape[1])  # Some triangular moment

    u = np.zeros((3, r0.shape[1]))
    omega = np.zeros((3, r0.shape[1]))

    x_eval = np.transpose(np.vstack((r0, th0, F, M, u, omega)))

    x_eval = np.reshape(x_eval, 18 * r0.shape[1])

    # Keeping in mind that the number of sections and design variables is not the same

    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    cs = np.hstack((h, w))

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.cs', cs)

    prob.set_val('DoubleSymmetricStressModel.x', np.transpose(x_eval))

    prob.run_model()

    # Compute expected value of stress:

    h_expr = 0.5 * np.ones((1, r0.shape[1]))
    w_expr = 3 * np.ones((1, r0.shape[1]))

    I_xx = (w_expr * h_expr ** 3) / 12

    y = h_expr / 2

    sigma_expected = M[0, :] * y / I_xx

    sigma_actual = prob.get_val('DoubleSymmetricStressModel.sigma_axial')

    sigma_actual_tensile = sigma_actual[2 * r0.shape[1]:3 * r0.shape[1]]
    np.testing.assert_equal(np.squeeze(sigma_expected), sigma_actual_tensile)
    pass


def test_fuse_axial_stress_computation():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 10
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    beam_points[0, :] = np.linspace(0, 15, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')
    rect_beam = BeamDefinition('MainWing', beam_point_input, np.array([3, 1, 2]), E=Q_(70e9, 'pascal'),
                               G=Q_(30e9, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1)

    # Test loads for the geometry
    loads = []

    # Test joints for the geometry
    joints = []

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints,
                                                        constraints=[str_constraint], stress_definition=stress_model, debug_flag=True, num_interp_sections=4)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

    # Add the stress definition to the general model
    model.add_subsystem('DoubleSymmetricStressModel', sample_beam.options['stress_definition'])
    # Add the constraint:
    if len(sample_beam.options['constraints']) > 0:
        for a_constraint in sample_beam.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('DoubleSymmetricStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('DoubleSymmetricStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('DoubleSymmetricStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('DoubleSymmetricStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface

    model.connect('RectBeam.DoubleSymmetricBeamInterface.corner_points', 'DoubleSymmetricStressModel.corner_points')
    model.connect('RectBeam.DoubleSymmetricBeamInterface.cs_out', 'DoubleSymmetricStressModel.cs')

    prob = om.Problem(model)
    prob.setup()

    # Test the actual result of the solution:
    r0 = sample_beam.options['r0']
    th0 = sample_beam.options['th0']

    F = np.zeros((3, r0.shape[1]))

    M = np.zeros((3, r0.shape[1]))

    M[1, :] = -np.linspace(1000000, 0, r0.shape[1])  # Some triangular moment

    u = np.zeros((3, r0.shape[1]))
    omega = np.zeros((3, r0.shape[1]))

    x_eval = np.transpose(np.vstack((r0, th0, F, M, u, omega)))

    x_eval = np.reshape(x_eval, 18 * r0.shape[1])

    # Keeping in mind that the number of sections and design variables is not the same

    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    cs = np.hstack((h, w))

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.cs', cs)

    prob.set_val('DoubleSymmetricStressModel.x', np.transpose(x_eval))

    prob.run_model()

    # Compute expected value of stress:

    h_expr = 0.5 * np.ones((1, r0.shape[1]))
    w_expr = 3 * np.ones((1, r0.shape[1]))

    I_xx = (w_expr * h_expr ** 3) / 12

    y = h_expr / 2

    sigma_expected = -M[1, :] * y / I_xx

    sigma_actual = prob.get_val('DoubleSymmetricStressModel.sigma_axial')

    sigma_actual_tensile = sigma_actual[2 * r0.shape[1]:3 * r0.shape[1]]
    np.testing.assert_almost_equal(np.squeeze(sigma_expected), sigma_actual_tensile)
    pass


def test_shear_stress_computation():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 10
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    beam_points[1, :] = np.linspace(0, 15, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')
    rect_beam = BeamDefinition('MainWing', beam_point_input, np.array([1, 3, 2]), E=Q_(70e9, 'pascal'),
                               G=Q_(30e9, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1)

    # Test loads for the geometry
    loads = []

    # Test joints for the geometry
    joints = []

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints,
                                                        constraints=[str_constraint],
                                                        stress_definition=stress_model, num_interp_sections=1)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

    # Add the stress definition to the general model
    model.add_subsystem('DoubleSymmetricStressModel', sample_beam.options['stress_definition'])
    # Add the constraint:
    if len(sample_beam.options['constraints']) > 0:
        for a_constraint in sample_beam.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('DoubleSymmetricStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('DoubleSymmetricStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('DoubleSymmetricStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('DoubleSymmetricStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface

    model.connect('RectBeam.DoubleSymmetricBeamInterface.corner_points', 'DoubleSymmetricStressModel.corner_points')
    model.connect('RectBeam.DoubleSymmetricBeamInterface.cs_out', 'DoubleSymmetricStressModel.cs')

    prob = om.Problem(model)
    prob.setup()

    # Test the actual result of the solution:
    r0 = sample_beam.options['r0']
    th0 = sample_beam.options['th0']

    F = np.zeros((3, r0.shape[1]))

    M = np.zeros((3, r0.shape[1]))

    F[2, :] = np.linspace(1000000, 0, r0.shape[1])  # Directly loaded shear force

    u = np.zeros((3, r0.shape[1]))
    omega = np.zeros((3, r0.shape[1]))

    x_eval = np.transpose(np.vstack((r0, th0, F, M, u, omega)))

    x_eval = np.reshape(x_eval, 18 * r0.shape[1])

    h = 0.65 * np.ones((1, n_sections_before_joints_loads))
    w = 3.25 * np.ones((1, n_sections_before_joints_loads))

    cs = np.hstack((h, w))

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.cs', cs)

    prob.set_val('DoubleSymmetricStressModel.x', np.transpose(x_eval))

    prob.run_model()

    # Compute expected value of stress:

    h_expr = 0.65 * np.ones((1, r0.shape[1]))
    w_expr = 3.25 * np.ones((1, r0.shape[1]))

    sigma_expected = 1.5 * F[2, :] / (h_expr * w_expr)  # For a rectangular section, value of maximum shear
    # stress will be equal to the 1.5 times of mean shear stress.

    sigma_actual = prob.get_val('DoubleSymmetricStressModel.tau_max_n')

    sigma_actual_tensile = sigma_actual[0 * r0.shape[1]:1 * r0.shape[1]]
    np.testing.assert_almost_equal(np.squeeze(sigma_expected), sigma_actual_tensile)
    pass


def test_torsional_stress_computation():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 10
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    beam_points[1, :] = np.linspace(0, 15, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')
    rect_beam = BeamDefinition('MainWing', beam_point_input, np.array([1, 3, 2]), E=Q_(70e9, 'pascal'),
                               G=Q_(30e9, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1)

    # Test loads for the geometry
    loads = []

    # Test joints for the geometry
    joints = []

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints,
                                                        constraints=[str_constraint],
                                                        stress_definition=stress_model, num_interp_sections=3)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

    # Add the stress definition to the general model
    model.add_subsystem('DoubleSymmetricStressModel', sample_beam.options['stress_definition'])
    # Add the constraint:
    if len(sample_beam.options['constraints']) > 0:
        for a_constraint in sample_beam.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('DoubleSymmetricStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('DoubleSymmetricStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('DoubleSymmetricStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('DoubleSymmetricStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface

    model.connect('RectBeam.DoubleSymmetricBeamInterface.corner_points', 'DoubleSymmetricStressModel.corner_points')
    model.connect('RectBeam.DoubleSymmetricBeamInterface.cs_out', 'DoubleSymmetricStressModel.cs')

    prob = om.Problem(model)
    prob.setup()

    # Test the actual result of the solution:
    r0 = sample_beam.options['r0']
    th0 = sample_beam.options['th0']

    F = np.zeros((3, r0.shape[1]))

    M = np.zeros((3, r0.shape[1]))

    M[1, :] = np.linspace(1000000, 0, r0.shape[1])  # Some triangular torsional moment

    u = np.zeros((3, r0.shape[1]))
    omega = np.zeros((3, r0.shape[1]))

    x_eval = np.transpose(np.vstack((r0, th0, F, M, u, omega)))

    x_eval = np.reshape(x_eval, 18 * r0.shape[1])

    h = 0.5 * np.ones(n_sections_before_joints_loads)
    w = 3.0 * np.ones(n_sections_before_joints_loads)

    cs = np.hstack((h, w))

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.cs', cs)

    prob.set_val('DoubleSymmetricStressModel.x', np.transpose(x_eval))

    prob.run_model()

    # Compute expected value of stress:

    h_expr = 0.5 * np.ones(r0.shape[1])
    w_expr = 3.0 * np.ones(r0.shape[1])

    if np.linalg.norm(w_expr) > np.linalg.norm(h_expr):
        tau_torsion = ((3 * M[1, :]) / (w_expr * h_expr ** 2)) * (
                1 + 0.6095 * (h_expr / w_expr) + 0.8865 * (h_expr / w_expr) ** 2 - 1.8023 * (h_expr / w_expr) ** 3 + 0.91 * (h_expr / w_expr) ** 4)
    else:
        tau_torsion = ((3 * M[1, :]) / (h_expr * w_expr ** 2)) * (
                1 + 0.6095 * (w_expr / h_expr) + 0.8865 * (w_expr / h_expr) ** 2 - 1.8023 * (w_expr / h_expr) ** 3 + 0.91 * (w_expr / h_expr) ** 4)

    sigma_actual = prob.get_val('DoubleSymmetricStressModel.sigma_vm')

    sigma_actual_torsion = sigma_actual[0 * r0.shape[1]:1 * r0.shape[1]]
    np.testing.assert_almost_equal(np.squeeze(tau_torsion), sigma_actual_torsion)
    pass


def test_box_axial_stress_computation():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 10
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    beam_points[1, :] = np.linspace(0, 15, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')
    rect_beam = BeamDefinition('MainWing', beam_point_input, np.array([1, 3, 2]), E=Q_(70e9, 'pascal'),
                               G=Q_(30e9, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1)

    # Test loads for the geometry
    loads = []

    # Test joints for the geometry
    joints = []

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = BoxBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints,
                                        constraints=[str_constraint], stress_definition=stress_model, debug_flag=True, num_interp_sections=4)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

    # Add the stress definition to the general model
    model.add_subsystem('BoxBeamStressModel', sample_beam.options['stress_definition'])
    # Add the constraint:
    if len(sample_beam.options['constraints']) > 0:
        for a_constraint in sample_beam.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('BoxBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('BoxBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('BoxBeamStressModel.tau_side', a_constraint.options["name"] + '.tau_side')

    # Connect the stress model with the beam interface

    model.connect('RectBeam.BoxBeamInterface.corner_points', 'BoxBeamStressModel.corner_points')
    model.connect('RectBeam.BoxBeamInterface.cs_out', 'BoxBeamStressModel.cs')

    prob = om.Problem(model)
    prob.setup()

    # Test the actual result of the solution:
    r0 = sample_beam.options['r0']
    th0 = sample_beam.options['th0']

    F = np.zeros((3, r0.shape[1]))

    M = np.zeros((3, r0.shape[1]))

    M[0, :] = np.linspace(1000000, 0, r0.shape[1])  # Some triangular moment

    u = np.zeros((3, r0.shape[1]))
    omega = np.zeros((3, r0.shape[1]))

    x_eval = np.transpose(np.vstack((r0, th0, F, M, u, omega)))

    x_eval = np.reshape(x_eval, 18 * r0.shape[1])

    # Keeping in mind that the number of sections and design variables is not the same

    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    t_left = 0.1 * np.ones((1, n_sections_before_joints_loads))
    t_top = 0.05 * np.ones((1, n_sections_before_joints_loads))

    t_right = 0.1 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.05 * np.ones((1, n_sections_before_joints_loads))

    cs = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    prob.set_val('RectBeam.BoxBeamInterface.cs', cs)

    prob.set_val('BoxBeamStressModel.x', np.transpose(x_eval))

    prob.run_model()

    # Compute expected value of stress:

    h_expr = 0.5 * np.ones((1, r0.shape[1]))
    w_expr = 3 * np.ones((1, r0.shape[1]))

    t_left_expr = 0.1 * np.ones((1, r0.shape[1]))
    t_top_expr = 0.05 * np.ones((1, r0.shape[1]))

    t_right_exprt = 0.1 * np.ones((1, r0.shape[1]))
    t_bot_expr = 0.05 * np.ones((1, r0.shape[1]))

    I_xx = (w_expr * h_expr ** 3) / 12 - ((w_expr - t_left_expr - t_right_exprt) * (h_expr - t_top_expr - t_bot_expr) ** 3) / 12

    y = h_expr / 2

    sigma_expected = M[0, :] * y / I_xx

    sigma_actual = prob.get_val('BoxBeamStressModel.sigma_axial')

    sigma_actual_tensile = sigma_actual[2 * r0.shape[1]:3 * r0.shape[1]]
    np.testing.assert_almost_equal(np.squeeze(sigma_expected), sigma_actual_tensile)
    pass


def test_box_torsion_computation():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 10
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    beam_points[1, :] = np.linspace(0, 1.5, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')
    rect_beam = BeamDefinition('MainWing', beam_point_input, np.array([1, 3, 2]), E=Q_(75e9, 'pascal'),
                               G=Q_(38e9, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1)

    # Test loads for the geometry
    loads = []

    # Test joints for the geometry
    joints = []

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = BoxBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints,
                                        constraints=[str_constraint], stress_definition=stress_model, debug_flag=True, num_interp_sections=4)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

    # Add the stress definition to the general model
    model.add_subsystem('BoxBeamStressModel', sample_beam.options['stress_definition'])
    # Add the constraint:
    if len(sample_beam.options['constraints']) > 0:
        for a_constraint in sample_beam.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('BoxBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('BoxBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('BoxBeamStressModel.tau_side', a_constraint.options["name"] + '.tau_side')

    # Connect the stress model with the beam interface

    model.connect('RectBeam.BoxBeamInterface.corner_points', 'BoxBeamStressModel.corner_points')
    model.connect('RectBeam.BoxBeamInterface.cs_out', 'BoxBeamStressModel.cs')

    prob = om.Problem(model)
    prob.setup()

    # Test the actual result of the solution:
    r0 = sample_beam.options['r0']
    th0 = sample_beam.options['th0']

    F = np.zeros((3, r0.shape[1]))

    M = np.zeros((3, r0.shape[1]))

    Torsion = 100  # N m

    M[1, :] = np.linspace(Torsion, Torsion, r0.shape[1])  # Some constant torsion

    u = np.zeros((3, r0.shape[1]))
    omega = np.zeros((3, r0.shape[1]))

    x_0 = np.transpose(np.vstack((r0, th0, 0 * F, 0 * M, u, omega)))

    x_eval = np.transpose(np.vstack((r0, th0, F, M, u, omega)))

    x_0 = np.reshape(x_0, 18 * r0.shape[1])
    x_eval = np.reshape(x_eval, 18 * r0.shape[1])

    # Keeping in mind that the number of sections and design variables is not the same

    h = 0.084 * np.ones((1, n_sections_before_joints_loads))
    w = 0.053 * np.ones((1, n_sections_before_joints_loads))

    t_left = 0.003 * np.ones((1, n_sections_before_joints_loads))
    t_top = 0.004 * np.ones((1, n_sections_before_joints_loads))

    t_right = 0.003 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.004 * np.ones((1, n_sections_before_joints_loads))

    cs = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    prob.set_val('RectBeam.BoxBeamInterface.cs', cs)

    prob.set_val('BoxBeamStressModel.x', np.transpose(x_eval))

    prob.run_model()

    # Compute expected value of stress:

    h_expr = 0.084 * np.ones((1, r0.shape[1]))
    w_expr = 0.053 * np.ones((1, r0.shape[1]))

    t_left_expr = 0.003 * np.ones((1, r0.shape[1]))
    t_top_expr = 0.004 * np.ones((1, r0.shape[1]))

    t_right_exprt = 0.003 * np.ones((1, r0.shape[1]))
    t_bot_expr = 0.004 * np.ones((1, r0.shape[1]))

    A_inner = (h_expr - t_top_expr) * (w_expr - t_right_exprt)

    tau_right_theoretical = Torsion * np.ones((1, r0.shape[1])) / (2 * t_right_exprt * A_inner)

    tau_top_theoretical = Torsion * np.ones((1, r0.shape[1])) / (2 * t_top_expr * A_inner)

    sigma_actual = prob.get_val('BoxBeamStressModel.tau_side')

    tau_top = sigma_actual[1 * r0.shape[1]:2 * r0.shape[1]]
    tau_right = sigma_actual[2 * r0.shape[1]:3 * r0.shape[1]]
    np.testing.assert_almost_equal(np.squeeze(tau_right_theoretical), tau_right)
    np.testing.assert_almost_equal(np.squeeze(tau_top_theoretical), tau_top)
    pass


def test_lean_rect_beam_computation():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 7
    beam_points = np.zeros((3, n_sections_before_joints_loads))

    L = 10

    beam_points[0, :] = np.linspace(0, L, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    fuse_beam = BeamDefinition('Fuselage', beam_point_input, np.array([3, 1, 2]), E=Q_(69e9, 'pascal'),
                               G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1, bc=BoundaryType.CANTILEVER)

    # Test loads for the geometry
    loads = []
    label1 = 'load1'
    Load = 1000  # Newton
    f1 = Q_(np.array([0, 0, Load]), 'newton')
    m1 = Q_(np.array([0, 0, 0]), 'newton*meter')
    eta1 = 1.0
    load1 = PointLoadDefinition(label1, eta1, f1, m1)
    loads.append(load1)

    joints = []

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint_fuse", debug_flag=True)
    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular_fuse')

    fuselage = StaticDoublySymRectBeamRepresentation(beam_definition=fuse_beam, applied_loads=loads, joints=joints, constraints=[str_constraint], stress_definition=stress_model,
                                                     num_interp_sections=0)

    T_stickmodel = BeamStickModel(load_factor=0.0, beam_list=[fuselage], joint_reference=joints)

    model.add_subsystem(name='Fuselage', subsys=fuselage)
    model.add_subsystem(name='Tstickmodel', subsys=T_stickmodel)

    # Add the stress definition to the general model
    model.add_subsystem('FuselageDoubleSymmetricBeamStressModel', fuselage.options['stress_definition'])
    # Add the constraint:
    if len(fuselage.options['constraints']) > 0:
        for a_constraint in fuselage.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuselageDoubleSymmetricBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuselageDoubleSymmetricBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuselageDoubleSymmetricBeamStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('FuselageDoubleSymmetricBeamStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Setting up connections:
    model.connect('Fuselage.DoubleSymmetricBeamInterface.cs_out', 'Tstickmodel.cs')
    model.connect('Tstickmodel.x', 'FuselageDoubleSymmetricBeamStressModel.x')

    # Connect the stress model with the beam interface
    model.connect('Fuselage.DoubleSymmetricBeamInterface.cs_out', 'FuselageDoubleSymmetricBeamStressModel.cs')
    model.connect('Fuselage.DoubleSymmetricBeamInterface.corner_points', 'FuselageDoubleSymmetricBeamStressModel.corner_points')

    prob = om.Problem(model)
    prob.setup()
    om.n2(prob)
    # Input design variables
    h = 0.1
    w = 0.5
    cs_in = np.hstack(
        (h * np.ones((int(fuselage.options['beam_shape'].value/2) * fuselage.options['num_DvCs'])), w * np.ones((int(fuselage.options['beam_shape'].value/2) * fuselage.options['num_DvCs']))))

    prob.set_val('Fuselage.DoubleSymmetricBeamInterface.cs', cs_in)

    prob.set_val('Tstickmodel.Xac', np.zeros(18))
    prob.set_val('Tstickmodel.forces_dist', np.zeros(3 * T_stickmodel.beam_reference['forces_dist'].shape[1]))
    prob.set_val('Tstickmodel.moments_dist', np.zeros(3 * T_stickmodel.beam_reference['moments_dist'].shape[1]))
    prob.set_val('Tstickmodel.forces_conc', np.zeros(3 * T_stickmodel.beam_reference['forces_conc'].shape[1]))
    prob.set_val('Tstickmodel.moments_conc', np.zeros(3 * T_stickmodel.beam_reference['moments_conc'].shape[1]))

    # Solve Model

    prob.run_model()

    prob.check_partials(method='cs', compact_print=True, form='central')

    # Gather data:
    x_fuselage = prob.get_val('Tstickmodel.x')

    # Sort data:
    x_fuselage_start = np.reshape(T_stickmodel.numeric_storage['x0'], (int(x_fuselage.shape[0] / 18), 18)).T

    x_fuselage_end = np.reshape(np.squeeze(x_fuselage[:, 0]), (int(x_fuselage.shape[0] / 18), 18)).T

    dx = x_fuselage_end - x_fuselage_start

    dr = dx[0:3, :]

    endpoint_deflection = dr[2, -1]

    I_expr = (h ** 3 * w) / 12

    F = Load

    kappa = 5 / 6

    theoretical_deflection = (F * L ** 3) / (3 * 69e9 * I_expr) - (F * L) / (kappa * h * w * 1e20)

    np.testing.assert_almost_equal(theoretical_deflection, endpoint_deflection, decimal=3)


def test_lean_rect_beam_optimization():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 15
    beam_points = np.zeros((3, n_sections_before_joints_loads))

    L = 20

    beam_points[0, :] = np.linspace(0, L, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    fuse_beam = BeamDefinition('Fuselage', beam_point_input, np.array([3, 1, 2]), E=Q_(69e9, 'pascal'),
                               G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1, bc=BoundaryType.CANTILEVER)

    # Test loads for the geometry
    loads = []
    label1 = 'load1'
    Load = 1e6 # Newton
    f1 = Q_(np.array([0, 0, Load]), 'newton')
    m1 = Q_(np.array([0, 0, 0]), 'newton*meter')
    eta1 = 1.0
    load1 = PointLoadDefinition(label1, eta1, f1, m1)
    loads.append(load1)

    joints = []

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint_fuse", debug_flag=False)
    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular_fuse')

    fuselage = StaticDoublySymRectBeamRepresentation(beam_definition=fuse_beam, applied_loads=loads, joints=joints, constraints=[str_constraint], stress_definition=stress_model,
                                                     num_interp_sections=0)

    T_stickmodel = BeamStickModel(load_factor=0.0, beam_list=[fuselage], joint_reference=joints)

    model.add_subsystem(name='Fuselage', subsys=fuselage)
    model.add_subsystem(name='Tstickmodel', subsys=T_stickmodel)

    # # Add the stress definition to the general model
    model.add_subsystem('FuselageDoubleSymmetricBeamStressModel', fuselage.options['stress_definition'])
    # Add the constraint:
    if len(fuselage.options['constraints']) > 0:
        for a_constraint in fuselage.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint) or isinstance(a_constraint, SpecificStrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuselageDoubleSymmetricBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuselageDoubleSymmetricBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuselageDoubleSymmetricBeamStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('FuselageDoubleSymmetricBeamStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Setting up connections:
    model.connect('Fuselage.DoubleSymmetricBeamInterface.cs_out', 'Tstickmodel.cs')
    model.connect('Tstickmodel.x', 'FuselageDoubleSymmetricBeamStressModel.x')

    # Connect the stress model with the beam interface
    model.connect('Fuselage.DoubleSymmetricBeamInterface.cs_out', 'FuselageDoubleSymmetricBeamStressModel.cs')
    model.connect('Fuselage.DoubleSymmetricBeamInterface.corner_points', 'FuselageDoubleSymmetricBeamStressModel.corner_points')

    lb = 0.15
    ub = 2.0

    # model.nonlinear_solver = om.NewtonSolver()
    # model.linear_solver = om.DirectSolver()
    # model.nonlinear_solver.options['iprint'] = 2
    # model.nonlinear_solver.options['maxiter'] = 12
    # model.nonlinear_solver.options['solve_subsystems'] = False
    model.options['assembled_jac_type'] = 'csc'

    model.add_design_var('Fuselage.DoubleSymmetricBeamInterface.cs', lower=lb, upper=ub)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_axial', upper=0.0)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_axial_n', upper=0.0)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_vm', upper=0.0)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_tau_max_c', upper=0.0)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_tau_max_c_n', upper=0.0)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_tau_max_n', upper=0.0)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_tau_max_n_n', upper=0.0)

    model.add_objective('Fuselage.DoubleSymmetricBeamInterface.mass', scaler=1/1300000)
    # 1300000 axial 900000 vm
    prob = om.Problem(model)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 3e-6
    prob.driver.options['disp'] = True
    prob.driver.options['maxiter'] = 500

    prob.setup()

    om.n2(prob)

    # Input design variables
    h = 0.5
    w = 0.5
    cs_in = np.hstack(
        (h * np.ones((int(fuselage.options['beam_shape'].value/2) * fuselage.options['num_DvCs'])), w * np.ones((int(fuselage.options['beam_shape'].value/2) * fuselage.options['num_DvCs']))))

    prob.set_val('Fuselage.DoubleSymmetricBeamInterface.cs', cs_in)

    prob.set_val('Tstickmodel.Xac', np.zeros(18))
    prob.set_val('Tstickmodel.forces_dist', np.zeros(3 * T_stickmodel.beam_reference['forces_dist'].shape[1]))
    prob.set_val('Tstickmodel.moments_dist', np.zeros(3 * T_stickmodel.beam_reference['moments_dist'].shape[1]))
    prob.set_val('Tstickmodel.forces_conc', np.zeros(3 * T_stickmodel.beam_reference['forces_conc'].shape[1]))
    prob.set_val('Tstickmodel.moments_conc', np.zeros(3 * T_stickmodel.beam_reference['moments_conc'].shape[1]))

    # Solve Model

    # Solve Model
    # prob.run_model()

    # prob.check_partials(method='cs', compact_print=True, form='central')
    #
    prob.run_driver()
    print(prob['Fuselage.DoubleSymmetricBeamInterface.cs'])
    pass


def test_lean_box_beam_optimization():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 15
    beam_points = np.zeros((3, n_sections_before_joints_loads))

    L = 20

    beam_points[0, :] = np.linspace(0, L, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    fuse_beam = BeamDefinition('Fuselage', beam_point_input, np.array([3, 1, 2]), E=Q_(69e9, 'pascal'),
                               G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1, bc=BoundaryType.CANTILEVER)

    # Test loads for the geometry
    loads = []
    label1 = 'load1'
    Load = 1e6 # Newton
    f1 = Q_(np.array([0, 0, Load]), 'newton')
    m1 = Q_(np.array([0, 0, 0]), 'newton*meter')
    eta1 = 1.0
    load1 = PointLoadDefinition(label1, eta1, f1, m1)
    loads.append(load1)

    joints = []

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint_fuse", debug_flag=False)
    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular_fuse')

    fuselage = BoxBeamRepresentation(beam_definition=fuse_beam, applied_loads=loads, joints=joints, constraints=[str_constraint], stress_definition=stress_model,
                                                     num_interp_sections=0)

    T_stickmodel = BeamStickModel(load_factor=0.0, beam_list=[fuselage], joint_reference=joints)

    model.add_subsystem(name='Fuselage', subsys=fuselage)
    model.add_subsystem(name='Tstickmodel', subsys=T_stickmodel)

    # # Add the stress definition to the general model
    model.add_subsystem('FuselageBoxBeamStressModel', fuselage.options['stress_definition'])
    # Add the constraint:
    if len(fuselage.options['constraints']) > 0:
        for a_constraint in fuselage.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint) or isinstance(a_constraint, SpecificStrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuselageBoxBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuselageBoxBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuselageBoxBeamStressModel.tau_side', a_constraint.options["name"] + '.tau_side')

    # Setting up connections:
    model.connect('Fuselage.BoxBeamInterface.cs_out', 'Tstickmodel.cs')
    model.connect('Tstickmodel.x', 'FuselageBoxBeamStressModel.x')

    # Connect the stress model with the beam interface
    model.connect('Fuselage.BoxBeamInterface.cs_out', 'FuselageBoxBeamStressModel.cs')
    model.connect('Fuselage.BoxBeamInterface.corner_points', 'FuselageBoxBeamStressModel.corner_points')

    model.options['assembled_jac_type'] = 'csc'

    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    t_left = 0.003 * np.ones((1, n_sections_before_joints_loads))
    t_top = 0.003 * np.ones((1, n_sections_before_joints_loads))

    t_right = 0.003 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.003 * np.ones((1, n_sections_before_joints_loads))

    lb = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    t_left = 1.5 * np.ones((1, n_sections_before_joints_loads))
    t_top = 1.5 * np.ones((1, n_sections_before_joints_loads))

    t_right = 0.25 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.25 * np.ones((1, n_sections_before_joints_loads))

    ub = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    model.add_design_var('Fuselage.BoxBeamInterface.cs', lower=lb, upper=ub)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_axial', upper=0.0)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_axial_n', upper=0.0)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_vm', upper=0.0)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_tau_side', upper=0.0)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_tau_side_n', upper=0.0)

    model.add_objective('Fuselage.BoxBeamInterface.mass', scaler=1/1300000)
    # 1300000 axial 900000 vm
    prob = om.Problem(model)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 3e-6
    prob.driver.options['disp'] = True
    prob.driver.options['maxiter'] = 500

    prob.setup()

    om.n2(prob)

    # Input design variables
    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    t_left = 0.1 * np.ones((1, n_sections_before_joints_loads))
    t_top = 0.15 * np.ones((1, n_sections_before_joints_loads))

    t_right = 0.1 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.15 * np.ones((1, n_sections_before_joints_loads))

    cs = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    prob.set_val('Fuselage.BoxBeamInterface.cs', cs)

    prob.set_val('Tstickmodel.Xac', np.zeros(18))
    prob.set_val('Tstickmodel.forces_dist', np.zeros(3 * T_stickmodel.beam_reference['forces_dist'].shape[1]))
    prob.set_val('Tstickmodel.moments_dist', np.zeros(3 * T_stickmodel.beam_reference['moments_dist'].shape[1]))
    prob.set_val('Tstickmodel.forces_conc', np.zeros(3 * T_stickmodel.beam_reference['forces_conc'].shape[1]))
    prob.set_val('Tstickmodel.moments_conc', np.zeros(3 * T_stickmodel.beam_reference['moments_conc'].shape[1]))

    # Solve Model
    prob.run_driver()
    print('State of cross-sections:')
    print(prob['Fuselage.BoxBeamInterface.cs'])
    print('State of mass:')
    print(prob['Fuselage.BoxBeamInterface.mass'])
    print('State of Constraints:')
    print(prob[fuselage.options['constraints'][0].options["name"] + '.c_axial'])
    print(prob[fuselage.options['constraints'][0].options["name"] + '.c_axial_n'])
    print(prob[fuselage.options['constraints'][0].options["name"] + '.c_vm'])
    print(prob[fuselage.options['constraints'][0].options["name"] + '.c_tau_side'])
    print(prob[fuselage.options['constraints'][0].options["name"] + '.c_tau_side_n'])


def test_rect_lean_beam_dynamic_computation():
    # importing matplotlib package
    import matplotlib.pyplot as plt

    model = om.Group()

    t_initial = 0.0
    t_final = 1.0+1e-2
    time_step = 1e-2

    num_timesteps = int((t_final - t_initial) / time_step)

    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 21
    beam_points = np.zeros((3, n_sections_before_joints_loads))

    L = 20

    beam_points[0, :] = np.linspace(0, L, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    fuse_beam = BeamDefinition('Fuselage', beam_point_input, np.array([3, 1, 2]), E=Q_(70e9, 'pascal'),
                               G=Q_(30e9, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=num_timesteps, bc=BoundaryType.CANTILEVER)

    # Test loads for the geometry
    loads = []

    joints = []

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint_fuse")
    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular_fuse')

    fuselage = StaticDoublySymRectBeamRepresentation(beam_definition=fuse_beam, applied_loads=loads, joints=joints, constraints=[str_constraint], stress_definition=stress_model,
                                                     num_interp_sections=0)

    T_stickmodel = BeamStickModel(load_factor=0.0, beam_list=[fuselage], joint_reference=joints, t_initial=t_initial, t_final=t_final, time_step=time_step, load_function=beam_sinusoidal_load,
                                  t_gamma=0.0, t_epsilon=0.0)

    model.add_subsystem(name='Fuselage', subsys=fuselage)
    model.add_subsystem(name='Tstickmodel', subsys=T_stickmodel)

    # Add the stress definition to the general model
    model.add_subsystem('FuselageDoubleSymmetricBeamStressModel', fuselage.options['stress_definition'])
    # Add the constraint:
    if len(fuselage.options['constraints']) > 0:
        for a_constraint in fuselage.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuselageDoubleSymmetricBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuselageDoubleSymmetricBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuselageDoubleSymmetricBeamStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('FuselageDoubleSymmetricBeamStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Setting up connections:
    model.connect('Fuselage.DoubleSymmetricBeamInterface.cs_out', 'Tstickmodel.cs')
    model.connect('Tstickmodel.x', 'FuselageDoubleSymmetricBeamStressModel.x')

    # Connect the stress model with the beam interface
    model.connect('Fuselage.DoubleSymmetricBeamInterface.cs_out', 'FuselageDoubleSymmetricBeamStressModel.cs')
    model.connect('Fuselage.DoubleSymmetricBeamInterface.corner_points', 'FuselageDoubleSymmetricBeamStressModel.corner_points')

    prob = om.Problem(model)
    prob.setup()

    # Input design variables
    h = 0.1
    w = 0.5
    cs_in = np.hstack(
        (h * np.ones((int(fuselage.options['beam_shape'].value / 2) * fuselage.options['num_DvCs'])), w * np.ones((int(fuselage.options['beam_shape'].value / 2) * fuselage.options['num_DvCs']))))

    prob.set_val('Fuselage.DoubleSymmetricBeamInterface.cs', cs_in)


    prob.set_val('Tstickmodel.Xac', np.zeros(18))
    prob.set_val('Tstickmodel.forces_dist', np.zeros(3 * T_stickmodel.beam_reference['forces_dist'].shape[1]))
    prob.set_val('Tstickmodel.moments_dist', np.zeros(3 * T_stickmodel.beam_reference['moments_dist'].shape[1]))
    prob.set_val('Tstickmodel.forces_conc', np.zeros(3 * T_stickmodel.beam_reference['forces_conc'].shape[1]))
    prob.set_val('Tstickmodel.moments_conc', np.zeros(3 * T_stickmodel.beam_reference['moments_conc'].shape[1]))

    # Solve Model
    prob.run_model()

    # prob.check_partials(method='cs', compact_print=True, form='central')

    # Gather data:
    x_fuselage = prob.get_val('Tstickmodel.x')

    r_fuselage = np.squeeze(x_fuselage[362, :])

    t = T_stickmodel.numeric_storage['time']

    fig1 = plt.figure("Figure 1")

    ax11 = plt.subplot(111)

    ax11.plot(t, r_fuselage, color='r')

    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('Tip deflection (m)')
    ax11.grid(True)

    plt.show()


def test_t_beam_lean_computation():
    # importing matplotlib package
    import matplotlib.pyplot as plt

    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 7
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    L = 10
    beam_points[0, :] = np.linspace(0, L, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    fuse_beam = BeamDefinition('Fuselage', beam_point_input, np.array([3, 1, 2]), E=Q_(69e9, 'pascal'),
                               G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1, bc=BoundaryType.CANTILEVER)

    # Test loads for the geometry
    loads = []
    label1 = 'load1'
    Load = 200  # Newton
    f1 = Q_(np.array([0, 0, Load]), 'newton')
    m1 = Q_(np.array([0, 0, 0]), 'newton*meter')
    eta1 = 1.0
    load1 = PointLoadDefinition(label1, eta1, f1, m1)
    loads.append(load1)

    # Test joints for the geometry
    joints = []
    eta4 = 0.95
    joint1 = JointDefinition('Fuse_to_RHT', 'Fuselage', eta4, 'FuseRHT', 0.0)
    joints.append(joint1)
    joint2 = JointDefinition('Fuse_to_LHT', 'Fuselage', eta4, 'FuseLHT', 0.0)
    joints.append(joint2)

    joint_rhs = []
    joint_rhs.append(joint1)

    joint_lhs = []
    joint_lhs.append(joint2)

    # RHS Wing
    n_sections_before_joints_loads = 6
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    L_tail = 2.5
    beam_points[0, :] = np.linspace(eta4 * L, eta4 * L, n_sections_before_joints_loads)
    beam_points[1, :] = np.linspace(0, L_tail, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    RHS_beam = BeamDefinition('FuseRHT', beam_point_input, np.array([1, 3, 2]), E=Q_(69e9, 'pascal'),
                              G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1, bc=BoundaryType.FREEFREE)

    # LHS Wing

    beam_points = np.zeros((3, n_sections_before_joints_loads))
    L_tail = 2.5
    beam_points[0, :] = np.linspace(eta4 * L, eta4 * L, n_sections_before_joints_loads)
    beam_points[1, :] = np.linspace(0, -L_tail, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    LHS_beam = BeamDefinition('FuseLHT', beam_point_input, np.array([1, 3, 2]), E=Q_(69e9, 'pascal'),
                              G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1, bc=BoundaryType.FREEFREE)

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint_fuse")
    str_constraint_2 = StrengthAggregatedConstraint(name="basic_constraint_rht")
    str_constraint_3 = StrengthAggregatedConstraint(name="basic_constraint_lht")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular_fuse')
    stress_model_2 = EulerBernoulliStressModel(name='EBRectangular_rht')
    stress_model_3 = EulerBernoulliStressModel(name='EBRectangular_lht')

    # Adding Joints to the stickmodel group

    fuselage = StaticDoublySymRectBeamRepresentation(beam_definition=fuse_beam, applied_loads=[], joints=joints, constraints=[str_constraint], stress_definition=stress_model,
                                                     num_interp_sections=0)

    RHS_tail = StaticDoublySymRectBeamRepresentation(beam_definition=RHS_beam, applied_loads=loads, joints=joint_rhs, constraints=[str_constraint_2], stress_definition=stress_model_2,
                                                     num_interp_sections=0)

    LHS_tail = StaticDoublySymRectBeamRepresentation(beam_definition=LHS_beam, applied_loads=loads, joints=joint_lhs, constraints=[str_constraint_3], stress_definition=stress_model_3,
                                                     num_interp_sections=0)

    T_stickmodel = BeamStickModel(load_factor=0.0, beam_list=[fuselage, RHS_tail, LHS_tail], joint_reference=joints)

    input_fuser = StickModelFeeder(beam_list=[fuselage, RHS_tail, LHS_tail])

    input_return = StickModelDemultiplexer(beam_list=[fuselage, RHS_tail, LHS_tail], joint_reference=joints)

    model.add_subsystem(name='Fuselage', subsys=fuselage)
    model.add_subsystem(name='FuseRHT', subsys=RHS_tail)
    model.add_subsystem(name='FuseLHT', subsys=LHS_tail)
    model.add_subsystem(name='inputStickmodel', subsys=input_fuser)
    model.add_subsystem(name='Tstickmodel', subsys=T_stickmodel)
    model.add_subsystem(name='outputStickmodel', subsys=input_return)

    # Setting up connections:
    model.connect('Fuselage.DoubleSymmetricBeamInterface.cs_out', 'inputStickmodel.cs_0')
    model.connect('FuseRHT.DoubleSymmetricBeamInterface.cs_out', 'inputStickmodel.cs_1')
    model.connect('FuseLHT.DoubleSymmetricBeamInterface.cs_out', 'inputStickmodel.cs_2')

    model.connect('inputStickmodel.cs_out', 'Tstickmodel.cs')
    model.connect('Tstickmodel.x', 'outputStickmodel.x_in')

    # Add the stress definition to the general model
    model.add_subsystem('FuselageDoubleSymmetricBeamStressModel', fuselage.options['stress_definition'])

    # Add the constraint:
    if len(fuselage.options['constraints']) > 0:
        for a_constraint in fuselage.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuselageDoubleSymmetricBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuselageDoubleSymmetricBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuselageDoubleSymmetricBeamStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('FuselageDoubleSymmetricBeamStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface
    model.connect('Fuselage.DoubleSymmetricBeamInterface.cs_out', 'FuselageDoubleSymmetricBeamStressModel.cs')
    model.connect('Fuselage.DoubleSymmetricBeamInterface.corner_points', 'FuselageDoubleSymmetricBeamStressModel.corner_points')
    model.connect('outputStickmodel.x_0', 'FuselageDoubleSymmetricBeamStressModel.x')

    model.add_subsystem('FuseRHTDoubleSymmetricBeamStressModel', RHS_tail.options['stress_definition'])

    # Add the constraint:
    if len(RHS_tail.options['constraints']) > 0:
        for a_constraint in RHS_tail.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuseRHTDoubleSymmetricBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuseRHTDoubleSymmetricBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuseRHTDoubleSymmetricBeamStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('FuseRHTDoubleSymmetricBeamStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface
    model.connect('FuseRHT.DoubleSymmetricBeamInterface.cs_out', 'FuseRHTDoubleSymmetricBeamStressModel.cs')
    model.connect('FuseRHT.DoubleSymmetricBeamInterface.corner_points', 'FuseRHTDoubleSymmetricBeamStressModel.corner_points')
    model.connect('outputStickmodel.x_1', 'FuseRHTDoubleSymmetricBeamStressModel.x')

    model.add_subsystem('FuseLHTDoubleSymmetricBeamStressModel', LHS_tail.options['stress_definition'])

    # Add the constraint:
    if len(LHS_tail.options['constraints']) > 0:
        for a_constraint in LHS_tail.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuseLHTDoubleSymmetricBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuseLHTDoubleSymmetricBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuseLHTDoubleSymmetricBeamStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('FuseLHTDoubleSymmetricBeamStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface
    model.connect('FuseLHT.DoubleSymmetricBeamInterface.cs_out', 'FuseLHTDoubleSymmetricBeamStressModel.cs')
    model.connect('FuseLHT.DoubleSymmetricBeamInterface.corner_points', 'FuseLHTDoubleSymmetricBeamStressModel.corner_points')
    model.connect('outputStickmodel.x_2', 'FuseLHTDoubleSymmetricBeamStressModel.x')

    prob = om.Problem(model)
    prob.setup()

    om.n2(prob)

    # Set some initial guesses

    # Input design variables
    prob.set_val('Fuselage.DoubleSymmetricBeamInterface.cs', 0.1 * np.ones((fuselage.options['beam_shape'].value * fuselage.options['num_DvCs'])))
    prob.set_val('FuseRHT.DoubleSymmetricBeamInterface.cs', 0.1 * np.ones((RHS_tail.options['beam_shape'].value * RHS_tail.options['num_DvCs'])))
    prob.set_val('FuseLHT.DoubleSymmetricBeamInterface.cs', 0.1 * np.ones((LHS_tail.options['beam_shape'].value * LHS_tail.options['num_DvCs'])))

    # prob.set_val('Tstickmodel.xDot', np.zeros((x_0.shape[0], x_0.shape[1])))
    prob.set_val('Tstickmodel.Xac', np.zeros(18))
    prob.set_val('Tstickmodel.forces_dist', np.zeros(3 * T_stickmodel.beam_reference['forces_dist'].shape[1]))
    prob.set_val('Tstickmodel.moments_dist', np.zeros(3 * T_stickmodel.beam_reference['moments_dist'].shape[1]))
    prob.set_val('Tstickmodel.forces_conc', np.zeros(3 * T_stickmodel.beam_reference['forces_conc'].shape[1]))
    prob.set_val('Tstickmodel.moments_conc', np.zeros(3 * T_stickmodel.beam_reference['moments_conc'].shape[1]))

    # Solve Model
    prob.run_model()

    # Gather data:
    x_fuselage = prob.get_val('outputStickmodel.x_0')
    x_rht = prob.get_val('outputStickmodel.x_1')
    x_lht = prob.get_val('outputStickmodel.x_2')

    # Sort data:
    x_fuselage_end = np.reshape(np.squeeze(x_fuselage[:, 0]), (int(x_fuselage.shape[0] / 18), 18)).T
    x_rht_end = np.reshape(np.squeeze(x_rht[:, 0]), (int(x_rht.shape[0] / 18), 18)).T
    x_lht_end = np.reshape(np.squeeze(x_lht[:, 0]), (int(x_lht.shape[0] / 18), 18)).T

    x_fuselage_start = np.reshape(fuselage.options['x0'], (int(fuselage.options['x0'].shape[0] / 18), 18)).T
    x_rht_start = np.reshape(RHS_tail.options['x0'], (int(RHS_tail.options['x0'].shape[0] / 18), 18)).T
    x_lht_start = np.reshape(LHS_tail.options['x0'], (int(LHS_tail.options['x0'].shape[0] / 18), 18)).T

    # Plotting initial state:

    # creating an empty canvas
    fig = plt.figure()

    # defining the axes with the projection
    # as 3D so as to plot 3D graphs
    ax = plt.axes(projection="3d")

    r_fuse_start = np.squeeze(x_fuselage_start[0:3, :])
    r_rht_start = np.squeeze(x_rht_start[0:3, :])
    r_lht_start = np.squeeze(x_lht_start[0:3, :])

    ax.plot3D(r_fuse_start[0, :], r_fuse_start[1, :], r_fuse_start[2, :], 'red')
    ax.scatter3D(r_fuse_start[0, :], r_fuse_start[1, :], r_fuse_start[2, :], c=r_fuse_start[2, :], cmap='cividis')

    ax.plot3D(r_rht_start[0, :], r_rht_start[1, :], r_rht_start[2, :], 'red')
    ax.scatter3D(r_rht_start[0, :], r_rht_start[1, :], r_rht_start[2, :], c=r_rht_start[2, :], cmap='cividis')

    ax.plot3D(r_lht_start[0, :], r_lht_start[1, :], r_lht_start[2, :], 'red')
    ax.scatter3D(r_lht_start[0, :], r_lht_start[1, :], r_lht_start[2, :], c=r_lht_start[2, :], cmap='cividis')

    r_fuse_end = np.squeeze(x_fuselage_end[0:3, :])
    r_rht_end = np.squeeze(x_rht_end[0:3, :])
    r_lht_end = np.squeeze(x_lht_end[0:3, :])

    ax.plot3D(r_fuse_end[0, :], r_fuse_end[1, :], r_fuse_end[2, :], 'green')
    ax.scatter3D(r_fuse_end[0, :], r_fuse_end[1, :], r_fuse_end[2, :], c=r_fuse_end[2, :], cmap='cividis')

    ax.plot3D(r_rht_end[0, :], r_rht_end[1, :], r_rht_end[2, :], 'green')
    ax.scatter3D(r_rht_end[0, :], r_rht_end[1, :], r_rht_end[2, :], c=r_rht_end[2, :], cmap='cividis')

    ax.plot3D(r_lht_end[0, :], r_lht_end[1, :], r_lht_end[2, :], 'green')
    ax.scatter3D(r_lht_end[0, :], r_lht_end[1, :], r_lht_end[2, :], c=r_lht_end[2, :], cmap='cividis')

    # Showing the above plot
    plt.show()

    # Compute proper fuselage tip deflection:
    h_expr = 0.1
    w_expr = 0.1

    I_expr = (h_expr ** 3 * w_expr) / 12

    F = 2 * Load

    kappa = 5 / 6

    joint_deflection = (F * (eta4 * L) ** 3) / (3 * 69e9 * I_expr) - (F * eta4 * L) / (kappa * h_expr * w_expr * 1e20)

    actual_deflection = r_fuse_end[2, -2]

    np.testing.assert_almost_equal(joint_deflection, actual_deflection, decimal=2)


def test_t_beam_lean_optimization():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 8
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    L = 20
    beam_points[0, :] = np.linspace(0, L, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    fuse_beam = BeamDefinition('Fuselage', beam_point_input, np.array([3, 1, 2]), E=Q_(69e9, 'pascal'),
                               G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1, bc=BoundaryType.CANTILEVER)

    # Test loads for the geometry
    loads = []
    label1 = 'load1'
    Load = 5e5  # Newton
    f1 = Q_(np.array([0, 0, Load]), 'newton')
    m1 = Q_(np.array([0, 0, 0]), 'newton*meter')
    eta1 = 1.0
    load1 = PointLoadDefinition(label1, eta1, f1, m1)
    loads.append(load1)

    # Test joints for the geometry
    joints = []
    eta4 = 0.95
    joint1 = JointDefinition('Fuse_to_RHT', 'Fuselage', eta4, 'FuseRHT', 0.0)
    joints.append(joint1)
    joint2 = JointDefinition('Fuse_to_LHT', 'Fuselage', eta4, 'FuseLHT', 0.0)
    joints.append(joint2)

    joint_rhs = []
    joint_rhs.append(joint1)

    joint_lhs = []
    joint_lhs.append(joint2)

    # RHS Wing
    n_sections_before_joints_loads = 7
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    L_tail = 5.0
    beam_points[0, :] = np.linspace(eta4 * L, eta4 * L, n_sections_before_joints_loads)
    beam_points[1, :] = np.linspace(0, L_tail, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    RHS_beam = BeamDefinition('FuseRHT', beam_point_input, np.array([1, 3, 2]), E=Q_(69e9, 'pascal'), G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1, bc=BoundaryType.FREEFREE)

    # LHS Wing

    beam_points = np.zeros((3, n_sections_before_joints_loads))
    L_tail = 5.0
    beam_points[0, :] = np.linspace(eta4 * L, eta4 * L, n_sections_before_joints_loads)
    beam_points[1, :] = np.linspace(0, -L_tail, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    LHS_beam = BeamDefinition('FuseLHT', beam_point_input, np.array([1, 3, 2]), E=Q_(69e9, 'pascal'), G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1, bc=BoundaryType.FREEFREE)

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint_fuse")
    str_constraint_2 = StrengthAggregatedConstraint(name="basic_constraint_rht")
    str_constraint_3 = StrengthAggregatedConstraint(name="basic_constraint_lht")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular_fuse')
    stress_model_2 = EulerBernoulliStressModel(name='EBRectangular_rht')
    stress_model_3 = EulerBernoulliStressModel(name='EBRectangular_lht')

    # Adding Joints to the stickmodel group

    fuselage = StaticDoublySymRectBeamRepresentation(beam_definition=fuse_beam, applied_loads=[], joints=joints, constraints=[str_constraint], stress_definition=stress_model,
                                                     num_interp_sections=0)

    RHS_tail = StaticDoublySymRectBeamRepresentation(beam_definition=RHS_beam, applied_loads=loads, joints=joint_rhs, constraints=[str_constraint_2], stress_definition=stress_model_2,
                                                     num_interp_sections=0)

    LHS_tail = StaticDoublySymRectBeamRepresentation(beam_definition=LHS_beam, applied_loads=loads, joints=joint_lhs, constraints=[str_constraint_3], stress_definition=stress_model_3,
                                                     num_interp_sections=0)

    TotalMass = MassCombiner(name="TBeamCombiner", num_beams=3)

    T_stickmodel = BeamStickModel(load_factor=0.0, beam_list=[fuselage, RHS_tail, LHS_tail], joint_reference=joints)

    input_fuser = StickModelFeeder(beam_list=[fuselage, RHS_tail, LHS_tail])

    input_return = StickModelDemultiplexer(beam_list=[fuselage, RHS_tail, LHS_tail], joint_reference=joints)

    model.add_subsystem(name='Fuselage', subsys=fuselage)
    model.add_subsystem(name='FuseRHT', subsys=RHS_tail)
    model.add_subsystem(name='FuseLHT', subsys=LHS_tail)
    model.add_subsystem(name='TBeamCombiner', subsys=TotalMass)
    model.add_subsystem(name='inputStickmodel', subsys=input_fuser)
    model.add_subsystem(name='Tstickmodel', subsys=T_stickmodel)
    model.add_subsystem(name='outputStickmodel', subsys=input_return)

    # Setting up connections:
    model.connect('Fuselage.DoubleSymmetricBeamInterface.cs_out', 'inputStickmodel.cs_0')
    model.connect('FuseRHT.DoubleSymmetricBeamInterface.cs_out', 'inputStickmodel.cs_1')
    model.connect('FuseLHT.DoubleSymmetricBeamInterface.cs_out', 'inputStickmodel.cs_2')

    model.connect('Fuselage.DoubleSymmetricBeamInterface.mass', 'TBeamCombiner.mass_0')
    model.connect('FuseRHT.DoubleSymmetricBeamInterface.mass', 'TBeamCombiner.mass_1')
    model.connect('FuseLHT.DoubleSymmetricBeamInterface.mass', 'TBeamCombiner.mass_2')

    model.connect('inputStickmodel.cs_out', 'Tstickmodel.cs')
    model.connect('Tstickmodel.x', 'outputStickmodel.x_in')

    # Add the stress definition to the general model
    model.add_subsystem('FuselageDoubleSymmetricBeamStressModel', fuselage.options['stress_definition'])

    # Add the constraint:
    if len(fuselage.options['constraints']) > 0:
        for a_constraint in fuselage.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuselageDoubleSymmetricBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuselageDoubleSymmetricBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuselageDoubleSymmetricBeamStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('FuselageDoubleSymmetricBeamStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface
    model.connect('Fuselage.DoubleSymmetricBeamInterface.cs_out', 'FuselageDoubleSymmetricBeamStressModel.cs')
    model.connect('Fuselage.DoubleSymmetricBeamInterface.corner_points', 'FuselageDoubleSymmetricBeamStressModel.corner_points')
    model.connect('outputStickmodel.x_0', 'FuselageDoubleSymmetricBeamStressModel.x')

    model.add_subsystem('FuseRHTDoubleSymmetricBeamStressModel', RHS_tail.options['stress_definition'])

    # Add the constraint:
    if len(RHS_tail.options['constraints']) > 0:
        for a_constraint in RHS_tail.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuseRHTDoubleSymmetricBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuseRHTDoubleSymmetricBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuseRHTDoubleSymmetricBeamStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('FuseRHTDoubleSymmetricBeamStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface
    model.connect('FuseRHT.DoubleSymmetricBeamInterface.cs_out', 'FuseRHTDoubleSymmetricBeamStressModel.cs')
    model.connect('FuseRHT.DoubleSymmetricBeamInterface.corner_points', 'FuseRHTDoubleSymmetricBeamStressModel.corner_points')
    model.connect('outputStickmodel.x_1', 'FuseRHTDoubleSymmetricBeamStressModel.x')

    model.add_subsystem('FuseLHTDoubleSymmetricBeamStressModel', LHS_tail.options['stress_definition'])

    # Add the constraint:
    if len(LHS_tail.options['constraints']) > 0:
        for a_constraint in LHS_tail.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuseLHTDoubleSymmetricBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuseLHTDoubleSymmetricBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuseLHTDoubleSymmetricBeamStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('FuseLHTDoubleSymmetricBeamStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface
    model.connect('FuseLHT.DoubleSymmetricBeamInterface.cs_out', 'FuseLHTDoubleSymmetricBeamStressModel.cs')
    model.connect('FuseLHT.DoubleSymmetricBeamInterface.corner_points', 'FuseLHTDoubleSymmetricBeamStressModel.corner_points')
    model.connect('outputStickmodel.x_2', 'FuseLHTDoubleSymmetricBeamStressModel.x')

    lb = 0.15
    ub = 2.0

    model.options['assembled_jac_type'] = 'csc'

    model.add_design_var('Fuselage.DoubleSymmetricBeamInterface.cs', lower=lb, upper=ub)
    model.add_design_var('FuseRHT.DoubleSymmetricBeamInterface.cs', lower=lb, upper=ub)
    model.add_design_var('FuseLHT.DoubleSymmetricBeamInterface.cs', lower=lb, upper=ub)

    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_vm', upper=0.0)
    model.add_constraint(RHS_tail.options['constraints'][0].options["name"] + '.c_vm', upper=0.0)
    model.add_constraint(LHS_tail.options['constraints'][0].options["name"] + '.c_vm', upper=0.0)
    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_axial', upper=0.0)
    model.add_constraint(RHS_tail.options['constraints'][0].options["name"] + '.c_axial', upper=0.0)
    model.add_constraint(LHS_tail.options['constraints'][0].options["name"] + '.c_axial', upper=0.0)

    model.add_objective('TBeamCombiner.total_mass', scaler=1 / 1500000)

    prob = om.Problem(model)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 3e-6
    prob.driver.options['disp'] = True
    prob.driver.options['maxiter'] = 500

    prob.setup()

    om.n2(prob)

    # Set some initial guesses

    # Input design variables
    prob.set_val('Fuselage.DoubleSymmetricBeamInterface.cs', 0.5 * np.ones((fuselage.options['beam_shape'].value * fuselage.options['num_DvCs'])))
    prob.set_val('FuseRHT.DoubleSymmetricBeamInterface.cs', 0.5 * np.ones((RHS_tail.options['beam_shape'].value * RHS_tail.options['num_DvCs'])))
    prob.set_val('FuseLHT.DoubleSymmetricBeamInterface.cs', 0.5 * np.ones((LHS_tail.options['beam_shape'].value * LHS_tail.options['num_DvCs'])))

    prob.set_val('Tstickmodel.Xac', np.zeros(18))
    prob.set_val('Tstickmodel.forces_dist', np.zeros(3 * T_stickmodel.beam_reference['forces_dist'].shape[1]))
    prob.set_val('Tstickmodel.moments_dist', np.zeros(3 * T_stickmodel.beam_reference['moments_dist'].shape[1]))
    prob.set_val('Tstickmodel.forces_conc', np.zeros(3 * T_stickmodel.beam_reference['forces_conc'].shape[1]))
    prob.set_val('Tstickmodel.moments_conc', np.zeros(3 * T_stickmodel.beam_reference['moments_conc'].shape[1]))

    # Solve Model
    prob.run_driver()

    # Gather data:
    cs_fuselage = prob.get_val('Fuselage.DoubleSymmetricBeamInterface.cs')
    cs_rht = prob.get_val('FuseRHT.DoubleSymmetricBeamInterface.cs')
    cs_lht = prob.get_val('FuseLHT.DoubleSymmetricBeamInterface.cs')

    print('Fuselage Cross-section:')
    print(cs_fuselage)
    print('Right Tail Beam Cross-section:')
    print(cs_rht)
    print('Left Tail Beam Cross-section:')
    print(cs_lht)
    print('Fuselage Mass')
    print(prob.get_val('Fuselage.DoubleSymmetricBeamInterface.mass'))
    print('RHT Mass')
    print(prob.get_val('FuseRHT.DoubleSymmetricBeamInterface.mass'))
    print('LHT Mass')
    print(prob.get_val('FuseLHT.DoubleSymmetricBeamInterface.mass'))


def test_tbox_beam_lean_optimization():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 8
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    L = 20
    beam_points[0, :] = np.linspace(0, L, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    fuse_beam = BeamDefinition('Fuselage', beam_point_input, np.array([3, 1, 2]), E=Q_(69e9, 'pascal'),
                               G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1, bc=BoundaryType.CANTILEVER)

    # Test loads for the geometry
    loads = []
    label1 = 'load1'
    Load = 5e5  # Newton
    f1 = Q_(np.array([0, 0, Load]), 'newton')
    m1 = Q_(np.array([0, 0, 0]), 'newton*meter')
    eta1 = 1.0
    load1 = PointLoadDefinition(label1, eta1, f1, m1)
    loads.append(load1)

    # Test joints for the geometry
    joints = []
    eta4 = 0.95
    joint1 = JointDefinition('Fuse_to_RHT', 'Fuselage', eta4, 'FuseRHT', 0.0)
    joints.append(joint1)
    joint2 = JointDefinition('Fuse_to_LHT', 'Fuselage', eta4, 'FuseLHT', 0.0)
    joints.append(joint2)

    joint_rhs = []
    joint_rhs.append(joint1)

    joint_lhs = []
    joint_lhs.append(joint2)

    # RHS Wing
    n_sections_before_joints_loads = 8
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    L_tail = 5.0
    beam_points[0, :] = np.linspace(eta4 * L, eta4 * L, n_sections_before_joints_loads)
    beam_points[1, :] = np.linspace(0, L_tail, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    RHS_beam = BeamDefinition('FuseRHT', beam_point_input, np.array([1, 3, 2]), E=Q_(69e9, 'pascal'), G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1, bc=BoundaryType.FREEFREE)

    # LHS Wing

    beam_points = np.zeros((3, n_sections_before_joints_loads))
    L_tail = 5.0
    beam_points[0, :] = np.linspace(eta4 * L, eta4 * L, n_sections_before_joints_loads)
    beam_points[1, :] = np.linspace(0, -L_tail, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    LHS_beam = BeamDefinition('FuseLHT', beam_point_input, np.array([1, 3, 2]), E=Q_(69e9, 'pascal'), G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=1, bc=BoundaryType.FREEFREE)

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint_fuse")
    str_constraint_2 = StrengthAggregatedConstraint(name="basic_constraint_rht")
    str_constraint_3 = StrengthAggregatedConstraint(name="basic_constraint_lht")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular_fuse')
    stress_model_2 = EulerBernoulliStressModel(name='EBRectangular_rht')
    stress_model_3 = EulerBernoulliStressModel(name='EBRectangular_lht')

    # Adding Joints to the stickmodel group

    fuselage = BoxBeamRepresentation(beam_definition=fuse_beam, applied_loads=[], joints=joints, constraints=[str_constraint], stress_definition=stress_model,
                                                     num_interp_sections=0)

    RHS_tail = BoxBeamRepresentation(beam_definition=RHS_beam, applied_loads=loads, joints=joint_rhs, constraints=[str_constraint_2], stress_definition=stress_model_2,
                                                     num_interp_sections=0)

    LHS_tail = BoxBeamRepresentation(beam_definition=LHS_beam, applied_loads=loads, joints=joint_lhs, constraints=[str_constraint_3], stress_definition=stress_model_3,
                                                     num_interp_sections=0)

    TotalMass = MassCombiner(name="TBeamCombiner", num_beams=3)

    T_stickmodel = BeamStickModel(load_factor=1.0, beam_list=[fuselage, RHS_tail, LHS_tail], joint_reference=joints)

    input_fuser = StickModelFeeder(beam_list=[fuselage, RHS_tail, LHS_tail])

    input_return = StickModelDemultiplexer(beam_list=[fuselage, RHS_tail, LHS_tail], joint_reference=joints)

    model.add_subsystem(name='Fuselage', subsys=fuselage)
    model.add_subsystem(name='FuseRHT', subsys=RHS_tail)
    model.add_subsystem(name='FuseLHT', subsys=LHS_tail)
    model.add_subsystem(name='TBeamCombiner', subsys=TotalMass)
    model.add_subsystem(name='inputStickmodel', subsys=input_fuser)
    model.add_subsystem(name='Tstickmodel', subsys=T_stickmodel)
    model.add_subsystem(name='outputStickmodel', subsys=input_return)

    # Setting up connections:
    model.connect('Fuselage.BoxBeamInterface.cs_out', 'inputStickmodel.cs_0')
    model.connect('FuseRHT.BoxBeamInterface.cs_out', 'inputStickmodel.cs_1')
    model.connect('FuseLHT.BoxBeamInterface.cs_out', 'inputStickmodel.cs_2')

    model.connect('Fuselage.BoxBeamInterface.mass', 'TBeamCombiner.mass_0')
    model.connect('FuseRHT.BoxBeamInterface.mass', 'TBeamCombiner.mass_1')
    model.connect('FuseLHT.BoxBeamInterface.mass', 'TBeamCombiner.mass_2')

    model.connect('inputStickmodel.cs_out', 'Tstickmodel.cs')
    model.connect('Tstickmodel.x', 'outputStickmodel.x_in')

    # Add the stress definition to the general model
    model.add_subsystem('BoxBeamStressModel', fuselage.options['stress_definition'])

    # Add the constraint:
    if len(fuselage.options['constraints']) > 0:
        for a_constraint in fuselage.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('BoxBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('BoxBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('BoxBeamStressModel.tau_side', a_constraint.options["name"] + '.tau_side')

    # Connect the stress model with the beam interface
    model.connect('Fuselage.BoxBeamInterface.cs_out', 'BoxBeamStressModel.cs')
    model.connect('Fuselage.BoxBeamInterface.corner_points', 'BoxBeamStressModel.corner_points')
    model.connect('outputStickmodel.x_0', 'BoxBeamStressModel.x')

    model.add_subsystem('FuseRHTBoxBeamStressModel', RHS_tail.options['stress_definition'])

    # Add the constraint:
    if len(RHS_tail.options['constraints']) > 0:
        for a_constraint in RHS_tail.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuseRHTBoxBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuseRHTBoxBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuseRHTBoxBeamStressModel.tau_side', a_constraint.options["name"] + '.tau_side')

    # Connect the stress model with the beam interface
    model.connect('FuseRHT.BoxBeamInterface.cs_out', 'FuseRHTBoxBeamStressModel.cs')
    model.connect('FuseRHT.BoxBeamInterface.corner_points', 'FuseRHTBoxBeamStressModel.corner_points')
    model.connect('outputStickmodel.x_1', 'FuseRHTBoxBeamStressModel.x')

    model.add_subsystem('FuseLHTBoxBeamStressModel', LHS_tail.options['stress_definition'])

    # Add the constraint:
    if len(LHS_tail.options['constraints']) > 0:
        for a_constraint in LHS_tail.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuseLHTBoxBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuseLHTBoxBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuseLHTBoxBeamStressModel.tau_side', a_constraint.options["name"] + '.tau_side')

    # Connect the stress model with the beam interface
    model.connect('FuseLHT.BoxBeamInterface.cs_out', 'FuseLHTBoxBeamStressModel.cs')
    model.connect('FuseLHT.BoxBeamInterface.corner_points', 'FuseLHTBoxBeamStressModel.corner_points')
    model.connect('outputStickmodel.x_2', 'FuseLHTBoxBeamStressModel.x')

    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    t_left = 0.002 * np.ones((1, n_sections_before_joints_loads))
    t_top = 0.002 * np.ones((1, n_sections_before_joints_loads))

    t_right = 0.002 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.002 * np.ones((1, n_sections_before_joints_loads))

    lb_fuse = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    t_left = 1.5 * np.ones((1, n_sections_before_joints_loads))
    t_top = 0.25 * np.ones((1, n_sections_before_joints_loads))

    t_right = 1.5 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.25 * np.ones((1, n_sections_before_joints_loads))

    ub_fuse = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    h = 0.25 * np.ones((1, n_sections_before_joints_loads))
    w = 1.5 * np.ones((1, n_sections_before_joints_loads))

    t_left = 0.001 * np.ones((1, n_sections_before_joints_loads))
    t_top = 0.001 * np.ones((1, n_sections_before_joints_loads))

    t_right = 0.001 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.001 * np.ones((1, n_sections_before_joints_loads))

    lb = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    h = 0.25 * np.ones((1, n_sections_before_joints_loads))
    w = 1.5 * np.ones((1, n_sections_before_joints_loads))

    t_left = 0.75 * np.ones((1, n_sections_before_joints_loads))
    t_top = 0.125 * np.ones((1, n_sections_before_joints_loads))

    t_right = 0.75 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.125 * np.ones((1, n_sections_before_joints_loads))

    ub = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    model.options['assembled_jac_type'] = 'csc'

    model.add_design_var('Fuselage.BoxBeamInterface.cs', lower=lb_fuse, upper=ub_fuse)
    model.add_design_var('FuseRHT.BoxBeamInterface.cs', lower=lb, upper=ub)
    model.add_design_var('FuseLHT.BoxBeamInterface.cs', lower=lb, upper=ub)

    model.add_constraint(fuselage.options['constraints'][0].options["name"] + '.c_vm', upper=0.0)
    model.add_constraint(RHS_tail.options['constraints'][0].options["name"] + '.c_vm', upper=0.0)
    model.add_constraint(LHS_tail.options['constraints'][0].options["name"] + '.c_vm', upper=0.0)
    model.add_objective('TBeamCombiner.total_mass', scaler=1 / 1500000)

    prob = om.Problem(model)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 8e-7
    prob.driver.options['disp'] = True
    prob.driver.options['maxiter'] = 500

    prob.setup()

    om.n2(prob)

    # Set some initial guesses

    # Input design variables
    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    t_left = 0.1 * np.ones((1, n_sections_before_joints_loads))
    t_top = 0.15 * np.ones((1, n_sections_before_joints_loads))

    t_right = 0.1 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.15 * np.ones((1, n_sections_before_joints_loads))

    cs_fuse = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    h = 0.5 * np.ones((1,n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    t_left = 0.1 * np.ones((1, n_sections_before_joints_loads))
    t_top = 0.15 * np.ones((1, n_sections_before_joints_loads))

    t_right = 0.1 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.15 * np.ones((1, n_sections_before_joints_loads))

    cs_RHS = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    t_left = 0.1 * np.ones((1, n_sections_before_joints_loads))
    t_top = 0.15 * np.ones((1, n_sections_before_joints_loads))

    t_right = 0.1 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.15 * np.ones((1, n_sections_before_joints_loads))

    cs_LHS = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    # Input design variables
    prob.set_val('Fuselage.BoxBeamInterface.cs', cs_fuse)
    prob.set_val('FuseRHT.BoxBeamInterface.cs', cs_RHS)
    prob.set_val('FuseLHT.BoxBeamInterface.cs', cs_LHS)

    prob.set_val('Tstickmodel.Xac', np.zeros(18))
    prob.set_val('Tstickmodel.forces_dist', np.zeros(3 * T_stickmodel.beam_reference['forces_dist'].shape[1]))
    prob.set_val('Tstickmodel.moments_dist', np.zeros(3 * T_stickmodel.beam_reference['moments_dist'].shape[1]))
    prob.set_val('Tstickmodel.forces_conc', np.zeros(3 * T_stickmodel.beam_reference['forces_conc'].shape[1]))
    prob.set_val('Tstickmodel.moments_conc', np.zeros(3 * T_stickmodel.beam_reference['moments_conc'].shape[1]))

    # Solve Model
    prob.run_driver()

    # Gather data:
    cs_fuselage = prob.get_val('Fuselage.BoxBeamInterface.cs')
    cs_rht = prob.get_val('FuseRHT.BoxBeamInterface.cs')
    cs_lht = prob.get_val('FuseLHT.BoxBeamInterface.cs')

    print('Fuselage Cross-section:')
    print(cs_fuselage)
    print('Right Tail Beam Cross-section:')
    print(cs_rht)
    print('Left Tail Beam Cross-section:')
    print(cs_lht)
    print('Fuselage Mass')
    print(prob.get_val('Fuselage.BoxBeamInterface.mass'))
    print('RHT Mass')
    print(prob.get_val('FuseRHT.BoxBeamInterface.mass'))
    print('LHT Mass')
    print(prob.get_val('FuseLHT.BoxBeamInterface.mass'))


def test_dynamic_t_beam_lean_computation():
    # importing matplotlib package
    import matplotlib.pyplot as plt

    model = om.Group()

    t_initial = 0.0
    t_final = 15.0
    time_step = 1/25

    num_timesteps = int((t_final - t_initial) / time_step)

    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 7
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    L = 10
    beam_points[0, :] = np.linspace(0, L, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    fuse_beam = BeamDefinition('Fuselage', beam_point_input, np.array([3, 1, 2]), E=Q_(69e9, 'pascal'),
                               G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=num_timesteps, bc=BoundaryType.CANTILEVER)

    # Test loads for the geometry
    loads = []
    label1 = 'load1'
    Load = 0 # Newton
    f1 = Q_(np.array([0, 0, Load]), 'newton')
    m1 = Q_(np.array([0, 0, 0]), 'newton*meter')
    eta1 = 1.0
    load1 = PointLoadDefinition(label1, eta1, f1, m1)
    loads.append(load1)

    # Test joints for the geometry
    joints = []
    eta4 = 0.95
    joint1 = JointDefinition('Fuse_to_RHT', 'Fuselage', eta4, 'FuseRHT', 0.0)
    joints.append(joint1)
    joint2 = JointDefinition('Fuse_to_LHT', 'Fuselage', eta4, 'FuseLHT', 0.0)
    joints.append(joint2)

    joint_rhs = []
    joint_rhs.append(joint1)

    joint_lhs = []
    joint_lhs.append(joint2)

    # RHS Wing
    n_sections_before_joints_loads = 6
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    L_tail = 2.5
    beam_points[0, :] = np.linspace(eta4 * L, eta4 * L, n_sections_before_joints_loads)
    beam_points[1, :] = np.linspace(0, L_tail, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    RHS_beam = BeamDefinition('FuseRHT', beam_point_input, np.array([1, 3, 2]), E=Q_(69e9, 'pascal'),
                              G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=num_timesteps, bc=BoundaryType.FREEFREE)

    # LHS Wing

    beam_points = np.zeros((3, n_sections_before_joints_loads))
    L_tail = 2.5
    beam_points[0, :] = np.linspace(eta4 * L, eta4 * L, n_sections_before_joints_loads)
    beam_points[1, :] = np.linspace(0, -L_tail, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')

    LHS_beam = BeamDefinition('FuseLHT', beam_point_input, np.array([1, 3, 2]), E=Q_(69e9, 'pascal'),
                              G=Q_(1e20, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'), num_timesteps=num_timesteps, bc=BoundaryType.FREEFREE)

    # Some constraint
    str_constraint = StrengthAggregatedConstraint(name="basic_constraint_fuse",debug_flag=True)
    str_constraint_2 = StrengthAggregatedConstraint(name="basic_constraint_rht")
    str_constraint_3 = StrengthAggregatedConstraint(name="basic_constraint_lht")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular_fuse')
    stress_model_2 = EulerBernoulliStressModel(name='EBRectangular_rht')
    stress_model_3 = EulerBernoulliStressModel(name='EBRectangular_lht')

    # Adding Joints to the stickmodel group

    fuselage = StaticDoublySymRectBeamRepresentation(beam_definition=fuse_beam, applied_loads=[], joints=joints, constraints=[str_constraint], stress_definition=stress_model,
                                                     num_interp_sections=0)

    RHS_tail = StaticDoublySymRectBeamRepresentation(beam_definition=RHS_beam, applied_loads=loads, joints=joint_rhs, constraints=[str_constraint_2], stress_definition=stress_model_2,
                                                     num_interp_sections=0)

    LHS_tail = StaticDoublySymRectBeamRepresentation(beam_definition=LHS_beam, applied_loads=loads, joints=joint_lhs, constraints=[str_constraint_3], stress_definition=stress_model_3,
                                                     num_interp_sections=0)

    TotalMass = MassCombiner(name="TBeamCombiner", num_beams=3)

    T_stickmodel = BeamStickModel(load_factor=1.0, beam_list=[fuselage, RHS_tail, LHS_tail], joint_reference=joints, t_initial=t_initial, t_final=t_final, time_step=time_step,
                                  load_function=t_beam_sinusoidal_load, t_gamma=0.0, t_epsilon=0.0)

    input_fuser = StickModelFeeder(beam_list=[fuselage, RHS_tail, LHS_tail])

    input_return = StickModelDemultiplexer(beam_list=[fuselage, RHS_tail, LHS_tail], joint_reference=joints)

    model.add_subsystem(name='Fuselage', subsys=fuselage)
    model.add_subsystem(name='FuseRHT', subsys=RHS_tail)
    model.add_subsystem(name='FuseLHT', subsys=LHS_tail)
    model.add_subsystem(name='TBeamCombiner', subsys=TotalMass)
    model.add_subsystem(name='inputStickmodel', subsys=input_fuser)
    model.add_subsystem(name='Tstickmodel', subsys=T_stickmodel)
    model.add_subsystem(name='outputStickmodel', subsys=input_return)

    # Setting up connections:
    model.connect('Fuselage.DoubleSymmetricBeamInterface.cs_out', 'inputStickmodel.cs_0')
    model.connect('FuseRHT.DoubleSymmetricBeamInterface.cs_out', 'inputStickmodel.cs_1')
    model.connect('FuseLHT.DoubleSymmetricBeamInterface.cs_out', 'inputStickmodel.cs_2')

    model.connect('Fuselage.DoubleSymmetricBeamInterface.mass', 'TBeamCombiner.mass_0')
    model.connect('FuseRHT.DoubleSymmetricBeamInterface.mass', 'TBeamCombiner.mass_1')
    model.connect('FuseLHT.DoubleSymmetricBeamInterface.mass', 'TBeamCombiner.mass_2')

    model.connect('inputStickmodel.cs_out', 'Tstickmodel.cs')
    model.connect('Tstickmodel.x', 'outputStickmodel.x_in')

    # Add the stress definition to the general model
    model.add_subsystem('FuselageDoubleSymmetricBeamStressModel', fuselage.options['stress_definition'])

    # Add the constraint:
    if len(fuselage.options['constraints']) > 0:
        for a_constraint in fuselage.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuselageDoubleSymmetricBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuselageDoubleSymmetricBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuselageDoubleSymmetricBeamStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('FuselageDoubleSymmetricBeamStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface
    model.connect('Fuselage.DoubleSymmetricBeamInterface.cs_out', 'FuselageDoubleSymmetricBeamStressModel.cs')
    model.connect('Fuselage.DoubleSymmetricBeamInterface.corner_points', 'FuselageDoubleSymmetricBeamStressModel.corner_points')
    model.connect('outputStickmodel.x_0', 'FuselageDoubleSymmetricBeamStressModel.x')

    model.add_subsystem('FuseRHTDoubleSymmetricBeamStressModel', RHS_tail.options['stress_definition'])

    # Add the constraint:
    if len(RHS_tail.options['constraints']) > 0:
        for a_constraint in RHS_tail.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuseRHTDoubleSymmetricBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuseRHTDoubleSymmetricBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuseRHTDoubleSymmetricBeamStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('FuseRHTDoubleSymmetricBeamStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface
    model.connect('FuseRHT.DoubleSymmetricBeamInterface.cs_out', 'FuseRHTDoubleSymmetricBeamStressModel.cs')
    model.connect('FuseRHT.DoubleSymmetricBeamInterface.corner_points', 'FuseRHTDoubleSymmetricBeamStressModel.corner_points')
    model.connect('outputStickmodel.x_1', 'FuseRHTDoubleSymmetricBeamStressModel.x')

    model.add_subsystem('FuseLHTDoubleSymmetricBeamStressModel', LHS_tail.options['stress_definition'])

    # Add the constraint:
    if len(LHS_tail.options['constraints']) > 0:
        for a_constraint in LHS_tail.options['constraints']:
            if isinstance(a_constraint, StrengthAggregatedConstraint):
                model.add_subsystem(a_constraint.options["name"], a_constraint)
                model.connect('FuseLHTDoubleSymmetricBeamStressModel.sigma_axial', a_constraint.options["name"] + '.sigma_axial')
                model.connect('FuseLHTDoubleSymmetricBeamStressModel.sigma_vm', a_constraint.options["name"] + '.sigma_vm')
                model.connect('FuseLHTDoubleSymmetricBeamStressModel.tau_max_c', a_constraint.options["name"] + '.tau_max_c')
                model.connect('FuseLHTDoubleSymmetricBeamStressModel.tau_max_n', a_constraint.options["name"] + '.tau_max_n')

    # Connect the stress model with the beam interface
    model.connect('FuseLHT.DoubleSymmetricBeamInterface.cs_out', 'FuseLHTDoubleSymmetricBeamStressModel.cs')
    model.connect('FuseLHT.DoubleSymmetricBeamInterface.corner_points', 'FuseLHTDoubleSymmetricBeamStressModel.corner_points')
    model.connect('outputStickmodel.x_2', 'FuseLHTDoubleSymmetricBeamStressModel.x')

    prob = om.Problem(model)
    prob.setup()

    # Input design variables
    prob.set_val('Fuselage.DoubleSymmetricBeamInterface.cs', 0.1 * np.ones((fuselage.options['beam_shape'].value * fuselage.options['num_DvCs'])))
    prob.set_val('FuseRHT.DoubleSymmetricBeamInterface.cs', 0.1 * np.ones((RHS_tail.options['beam_shape'].value * RHS_tail.options['num_DvCs'])))
    prob.set_val('FuseLHT.DoubleSymmetricBeamInterface.cs', 0.1 * np.ones((LHS_tail.options['beam_shape'].value * LHS_tail.options['num_DvCs'])))

    prob.set_val('Tstickmodel.Xac', np.zeros(18))
    prob.set_val('Tstickmodel.forces_dist', np.zeros(3 * T_stickmodel.beam_reference['forces_dist'].shape[1]))
    prob.set_val('Tstickmodel.moments_dist', np.zeros(3 * T_stickmodel.beam_reference['moments_dist'].shape[1]))
    prob.set_val('Tstickmodel.forces_conc', np.zeros(3 * T_stickmodel.beam_reference['forces_conc'].shape[1]))
    prob.set_val('Tstickmodel.moments_conc', np.zeros(3 * T_stickmodel.beam_reference['moments_conc'].shape[1]))

    # Solve Model
    prob.run_model()

    # Gather data:
    x_fuselage = prob.get_val('outputStickmodel.x_0')
    x_rht = prob.get_val('outputStickmodel.x_1')
    x_lht = prob.get_val('outputStickmodel.x_2')

    r_fuselage = np.squeeze(x_fuselage[146, :])
    r_rht = np.squeeze(x_rht[110, :])
    r_lht = np.squeeze(x_lht[110, :])

    t = T_stickmodel.numeric_storage['time']

    fig1 = plt.figure("Figure 1")

    ax11 = plt.subplot(111)

    ax11.plot(t, r_fuselage, color='r')

    ax11.plot(t, r_rht, color='b')

    ax11.plot(t, r_lht, color='g')


    ax11.set_xlabel('Time (s) 1-fuse 2-rht 3-lht')
    ax11.set_ylabel('Tip deflection (m)')
    ax11.grid(True)

    plt.show()



def t_beam_sinusoidal_load(x, xDot, Xac, forces_dist, moments_dist, forces_conc, moments_conc, time_step, i):
    # Load does not vary with time and is zero
    import math
    # forces_conc[38] = 0 * math.sin(30 * time_step * i)
    # forces_conc[59] = 0 * math.sin(30 * time_step * i)
    return Xac, forces_dist, moments_dist, forces_conc, moments_conc


def beam_sinusoidal_load(x, xDot, Xac, forces_dist, moments_dist, forces_conc, moments_conc, time_step, i):
    # Load does not vary with time and is zero
    import math
    forces_conc[-1] = 1e4*math.sin(20*time_step*i)
    return Xac, forces_dist, moments_dist, forces_conc, moments_conc
