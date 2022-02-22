from openasemdao.structures.beam.beam import StaticDoublySymRectBeamRepresentation
from openasemdao.structures.inputs.inputs import BeamDefinition, PointLoadDefinition, JointDefinition
from openasemdao.structures.utils.utils import unique
import openmdao.api as om
from openasemdao.structures.beam.constraints import StrenghtAggregatedConstraint
import math
from openasemdao import Q_
import numpy as np


def test_zero_element_generation():
    model = om.Group()
    # Generate a sequence of points for the beam
    n_sections_before_joints_loads = 12
    beam_points = np.zeros((3, n_sections_before_joints_loads))
    beam_points[1, :] = np.linspace(0, 15, n_sections_before_joints_loads)
    beam_point_input = Q_(beam_points, 'meter')
    rect_beam = BeamDefinition('MainWing', beam_point_input, np.array([1, 3, 2]), E=Q_(70e9, 'pascal'),
                               G=Q_(30e9, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'))

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

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints, constraints=[str_constraint])

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

    prob = om.Problem(model)
    prob.setup()

    # Test the actual result of the solution:
    r0 = sample_beam.options['r0']
    augmented_predicted_size = r0.shape[1]

    augmented_actual_size = 2 * num_zero_elements + n_sections_before_joints_loads

    np.testing.assert_equal(augmented_predicted_size, augmented_actual_size)

    pass


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
                               G=Q_(30e9, 'pascal'), rho=Q_(2700., 'kg/meter**3'), sigmaY=Q_(176e6, 'pascal'))

    # Some constraint
    str_constraint = StrenghtAggregatedConstraint(name="basic_constraint")

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=[], joints=[], constraints=[str_constraint])

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
