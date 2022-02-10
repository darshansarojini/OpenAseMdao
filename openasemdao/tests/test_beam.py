from openasemdao.structures.beam.beam import StaticDoublySymRectBeamRepresentation
from openasemdao.structures.inputs.inputs import BeamDefinition, PointLoadDefinition, JointDefinition
from openasemdao.structures.utils.utils import unique
import openmdao.api as om
import math
from openasemdao import Q_, ureg
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

    sample_beam = StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam, applied_loads=loads, joints=joints)

    model.add_subsystem(name='RectBeam', subsys=sample_beam)

    prob = om.Problem(model)
    prob.setup()

    # Test the actual result of the solution:
    r0 = sample_beam.options['r0']
    augmented_predicted_size = r0.shape[1]

    augmented_actual_size = 2 * num_zero_elements + n_sections_before_joints_loads

    np.testing.assert_equal(augmented_predicted_size, augmented_actual_size)
    
    pass