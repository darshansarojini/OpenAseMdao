from openasemdao.structures.beam.beam import StaticDoublySymRectBeamRepresentation, BoxBeamRepresentation
from openasemdao.structures.inputs.inputs import BeamDefinition, PointLoadDefinition, JointDefinition
from openasemdao.structures.utils.utils import unique
import openmdao.api as om
from openasemdao.structures.beam.constraints import StrenghtAggregatedConstraint
from openasemdao.structures.beam.stress_models import EulerBernoulliStressModel
from openasemdao.structures.beam.stickmodel import StaticBeamStickModel
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
    str_constraint = StrenghtAggregatedConstraint(name="basic_constraint")

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
    str_constraint = StrenghtAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints,
                                                        constraints=[str_constraint], stress_definition=stress_model, debug_flag=True, num_interp_sections=4)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

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

    x_0 = np.transpose(np.vstack((r0, th0, 0 * F, 0 * M, u, omega)))

    x_eval = np.transpose(np.vstack((r0, th0, F, M, u, omega)))

    x_0 = np.reshape(x_0, 18 * r0.shape[1])
    x_eval = np.reshape(x_eval, 18 * r0.shape[1])

    # Keeping in mind that the number of sections and design variables is not the same

    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    cs = np.hstack((h, w))

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.cs', cs)

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.x', np.transpose(np.vstack((x_0, x_eval))))

    prob.run_model()

    # Compute expected value of stress:

    h_expr = 0.5 * np.ones((1, r0.shape[1]))
    w_expr = 3 * np.ones((1, r0.shape[1]))

    I_xx = (w_expr * h_expr ** 3) / 12

    y = h_expr / 2

    sigma_expected = M[0, :] * y / I_xx

    sigma_actual = prob.get_val('RectBeam.DoubleSymmetricBeamStressModel.sigma')

    sigma_actual_tensile = sigma_actual[2 * r0.shape[1]:3 * r0.shape[1], 1]
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
    str_constraint = StrenghtAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints,
                                                        constraints=[str_constraint], stress_definition=stress_model, debug_flag=True, num_interp_sections=4)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

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

    x_0 = np.transpose(np.vstack((r0, th0, 0 * F, 0 * M, u, omega)))

    x_eval = np.transpose(np.vstack((r0, th0, F, M, u, omega)))

    x_0 = np.reshape(x_0, 18 * r0.shape[1])
    x_eval = np.reshape(x_eval, 18 * r0.shape[1])

    # Keeping in mind that the number of sections and design variables is not the same

    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    cs = np.hstack((h, w))

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.cs', cs)

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.x', np.transpose(np.vstack((x_0, x_eval))))

    prob.run_model()

    # Compute expected value of stress:

    h_expr = 0.5 * np.ones((1, r0.shape[1]))
    w_expr = 3 * np.ones((1, r0.shape[1]))

    I_xx = (w_expr * h_expr ** 3) / 12

    y = h_expr / 2

    sigma_expected = -M[1, :] * y / I_xx

    sigma_actual = prob.get_val('RectBeam.DoubleSymmetricBeamStressModel.sigma')

    sigma_actual_tensile = sigma_actual[2 * r0.shape[1]:3 * r0.shape[1], 1]
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
    str_constraint = StrenghtAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints,
                                                        constraints=[str_constraint],
                                                        stress_definition=stress_model, num_interp_sections=1)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

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

    x_0 = np.transpose(np.vstack((r0, th0, 0 * F, 0 * M, u, omega)))

    x_eval = np.transpose(np.vstack((r0, th0, F, M, u, omega)))

    x_0 = np.reshape(x_0, 18 * r0.shape[1])
    x_eval = np.reshape(x_eval, 18 * r0.shape[1])

    h = 0.65 * np.ones((1, n_sections_before_joints_loads))
    w = 3.25 * np.ones((1, n_sections_before_joints_loads))

    cs = np.hstack((h, w))

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.cs', cs)

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.x', np.transpose(np.vstack((x_0, x_eval))))

    prob.run_model()

    # Compute expected value of stress:

    h_expr = 0.65 * np.ones((1, r0.shape[1]))
    w_expr = 3.25 * np.ones((1, r0.shape[1]))

    sigma_expected = 1.5 * F[2, :] / (h_expr * w_expr)  # For a rectangular section, value of maximum shear
    # stress will be equal to the 1.5 times of mean shear stress.

    sigma_actual = prob.get_val('RectBeam.DoubleSymmetricBeamStressModel.sigma')

    sigma_actual_tensile = sigma_actual[9 * r0.shape[1]:10 * r0.shape[1], 1]
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
    str_constraint = StrenghtAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints,
                                                        constraints=[str_constraint],
                                                        stress_definition=stress_model, num_interp_sections=3)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

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

    x_0 = np.transpose(np.vstack((r0, th0, 0 * F, 0 * M, u, omega)))

    x_eval = np.transpose(np.vstack((r0, th0, F, M, u, omega)))

    x_0 = np.reshape(x_0, 18 * r0.shape[1])
    x_eval = np.reshape(x_eval, 18 * r0.shape[1])

    h = 0.5 * np.ones(n_sections_before_joints_loads)
    w = 3.0 * np.ones(n_sections_before_joints_loads)

    cs = np.hstack((h, w))

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.cs', cs)

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.x', np.transpose(np.vstack((x_0, x_eval))))

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

    sigma_actual = prob.get_val('RectBeam.DoubleSymmetricBeamStressModel.sigma')

    sigma_actual_torsion = sigma_actual[4 * r0.shape[1]:5 * r0.shape[1], 1]
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
    str_constraint = StrenghtAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = BoxBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints,
                                        constraints=[str_constraint], stress_definition=stress_model, debug_flag=True, num_interp_sections=4)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

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

    x_0 = np.transpose(np.vstack((r0, th0, 0 * F, 0 * M, u, omega)))

    x_eval = np.transpose(np.vstack((r0, th0, F, M, u, omega)))

    x_0 = np.reshape(x_0, 18 * r0.shape[1])
    x_eval = np.reshape(x_eval, 18 * r0.shape[1])

    # Keeping in mind that the number of sections and design variables is not the same

    h = 0.5 * np.ones((1, n_sections_before_joints_loads))
    w = 3 * np.ones((1, n_sections_before_joints_loads))

    t_left = 0.1 * np.ones((1, n_sections_before_joints_loads))
    t_top = 0.05 * np.ones((1, n_sections_before_joints_loads))

    t_right = 0.1 * np.ones((1, n_sections_before_joints_loads))
    t_bot = 0.05 * np.ones((1, n_sections_before_joints_loads))

    cs = np.hstack((h, w, t_left, t_top, t_right, t_bot))

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.cs', cs)

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.x', np.transpose(np.vstack((x_0, x_eval))))

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

    sigma_actual = prob.get_val('RectBeam.DoubleSymmetricBeamStressModel.sigma')

    sigma_actual_tensile = sigma_actual[6 * r0.shape[1]:7 * r0.shape[1], 1]
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
    str_constraint = StrenghtAggregatedConstraint(name="basic_constraint")

    # EB stress model
    stress_model = EulerBernoulliStressModel(name='EBRectangular')

    sample_beam = BoxBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints,
                                        constraints=[str_constraint], stress_definition=stress_model, debug_flag=True, num_interp_sections=4)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

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

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.cs', cs)

    prob.set_val('RectBeam.DoubleSymmetricBeamInterface.x', np.transpose(np.vstack((x_0, x_eval))))

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

    sigma_actual = prob.get_val('RectBeam.DoubleSymmetricBeamStressModel.sigma')

    tau_top = sigma_actual[9 * r0.shape[1]:10 * r0.shape[1], 1]
    tau_right = sigma_actual[10 * r0.shape[1]:11 * r0.shape[1], 1]
    np.testing.assert_almost_equal(np.squeeze(tau_right_theoretical), tau_right)
    np.testing.assert_almost_equal(np.squeeze(tau_top_theoretical), tau_top)
    pass


def test_t_beam_computation():
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
    f1 = Q_(np.array([0, 0, 1000]), 'newton')
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
    str_constraint = StrenghtAggregatedConstraint(name="basic_constraint_fuse")
    str_constraint_2 = StrenghtAggregatedConstraint(name="basic_constraint_rht")
    str_constraint_3 = StrenghtAggregatedConstraint(name="basic_constraint_lht")

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

    T_stickmodel = StaticBeamStickModel(load_factor=1.0, beam_list=[fuselage, RHS_tail, LHS_tail], joint_reference=joints)

    model.add_subsystem(name='Fuselage', subsys=fuselage)
    model.add_subsystem(name='FuseRHT', subsys=RHS_tail)
    model.add_subsystem(name='FuseLHT', subsys=LHS_tail)
    model.add_subsystem(name='Tstickmodel', subsys=T_stickmodel)

    prob = om.Problem(model)
    prob.setup()

    pass
