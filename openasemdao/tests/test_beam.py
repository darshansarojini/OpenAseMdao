from openasemdao.structures.beam.beam import StaticDoublySymRectBeamRepresentation
from openasemdao.structures.inputs.inputs import BeamDefinition, PointLoadDefinition, JointDefinition
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
                               G=Q_(30e9, 'pascal'), rho=Q_(2700., 'kg/meter**3'))

    # Test loads for the geometry
    loads = []
    label1 = 'load1'
    f1 = Q_(np.array([0, 0, 1000]), 'newton')
    m1 = Q_(np.array([0, 0, 10000]), 'newton*meter')
    eta1 = 0.25
    load1 = PointLoadDefinition(label1, eta1, f1, m1)
    loads.append(load1)
    label2 = 'load2'
    f2 = Q_(np.array([0, 0, 2000]), 'newton')
    m2 = Q_(np.array([0, 0, 20000]), 'newton*meter')
    eta2 = 0.5
    load2 = PointLoadDefinition(label2, eta2, f2, m2)
    loads.append(load2)
    label3 = 'load3'
    f3 = Q_(np.array([0, 0, 4000]), 'newton')
    m3 = Q_(np.array([0, 0, 50000]), 'newton*meter')
    eta3 = 0.75
    load3 = PointLoadDefinition(label3, eta3, f3, m3)
    loads.append(load3)

    # Test joints for the geometry
    joints = []
    joint1 = JointDefinition('inner_joint', 'MainWing', 0.6, 'FuseWing', 0.3)
    joints.append(joint1)
    joint2 = JointDefinition('outer_joint', 'MainWing', 0.85, 'Strut', 0.8)
    joints.append(joint2)

    # self.options.declare("beam_definition", default=None)
    # self.options.declare('num_divisions', types=int)
    # self.options.declare("applied_loads", default=[])
    # self.options.declare("joints", default=[])
    model.add_subsystem(name='RectBeam', subsys=StaticDoublySymRectBeamRepresentation(beam_definition=rect_beam,
                                                              applied_loads=loads, joints=joints))


    prob = om.Problem(model)
    prob.setup()