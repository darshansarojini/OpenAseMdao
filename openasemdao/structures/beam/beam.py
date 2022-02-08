from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core import group
from abc import ABC, abstractmethod
import numpy as np
from openasemdao.structures.utils import calculate_th0
from casadi import *


class SymbolicBeam(group, ABC):
    """
        Group that contains the symbolic beam functions that will be used for the structure. It will include
        the definition of the different beam constants, as well as the proper point distribution based on the joints
        and point loads.
    """
    def initialize(self):
        self.options.declare("beam_definition", default=None)
        self.options.declare('num_divisions', types=int)
        self.options.declare("applied_loads", default=[])
        self.options.declare("joints", default=[])

        # Beam axis node locations
        self.options.declare('r0')
        # Beam rotation sequence
        self.options.declare('seq')
        # Beam axis initial angles
        self.options.declare('th0')
        # Beam s0
        self.options.declare('delta_s0')
        # Beam node_lim and inter_node_lim
        self.options.declare('node_lim')
        self.options.declare('inter_node_lim')
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
        self.symbolics = {}
        # Additional inputs at initialize
        self.declare_additional_beam_inputs()
        return

    @abstractmethod
    def declare_additional_beam_inputs(self):
        return

    def setup(self):
        beam_definition = self.options["beam_definition"]
        applied_loads = self.options["applied_loads"]
        joints = self.options["joints"]

        # Define basic beam parameters from containers:
        self.options["seq"] = beam_definition.orientation

        # Read sequence of points within the beam

        initial_points = beam_definition.beam_points.magnitude
        span = 0

        # Get basic span value:

        if self.options["seq"] == np.array([3, 1, 2]): # Fuselage beam
            span = initial_points[0, -1] - initial_points[0, 0]
        if self.options["seq"] == np.array([1, 3, 2]): # Wing beam
            span = initial_points[1, -1] - initial_points[1, 0]

        recorded_load_points = []

        # Start with loads:
        if len(applied_loads) > 0:
            for a_load in applied_loads:
                found_lower_point = False
                # Check the load location and add a node there
                load_location = span * a_load.eta  # Either x or y, depending on the type of the beam
                if not(1.0 >= a_load.eta > 0.0):
                    raise ValueError('Load eta must be between 0 and 1')
                # Make point easily if eta is 1:
                if a_load.eta == 1.0:
                    last_slice = np.copy(initial_points[:, -1])
                    initial_points = np.transpose(np.vstack([np.transpose(initial_points), last_slice]))
                    # Finally add the load subsystem to the beam:
                    self.add_subsystem(a_load.load_label, a_load.component)
                    continue
                for i in range(0, initial_points.shape[1]):
                    if self.options["seq"] == np.array([3, 1, 2]): # Fuselage beam
                        current_span = initial_points[0, i]
                    else:  # Wing beam
                        current_span = initial_points[1, i]
                    if not found_lower_point and (load_location >= current_span):
                        found_lower_point = True
                        load_point = np.zeros(3)
                        if self.options["seq"] == np.array([3, 1, 2]):  # Fuselage beam
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
                        recorded_load_points.append(load_point) # To efficiently consider if the point has been created or not
                        initial_points = np.insert(initial_points, i+1, load_point, axis=1)
                        if span_percentage > 0.0:   # only duplicate point IF that point did not exist before
                            initial_points = np.insert(initial_points, i + 1, load_point, axis=1)
                # Finally add the load subsystem to the beam:
                self.add_subsystem(a_load.load_label, a_load.component)
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
                        # Finally add the load subsystem to the beam:
                        self.add_subsystem(a_joint.joint_label, a_joint.component)
                        break
                if point_exists:
                    break
                else:
                    # Time to create the point with regards to the joint
                    found_lower_point = False
                    joint_location = span * joint_eta   # Either x or y, depending on the type of the beam
                    if not (1.0 >= joint_eta > 0.0):
                        raise ValueError('Load eta must be between 0 and 1')
                    # Make point easily if eta is 1:
                    if joint_eta == 1.0:
                        last_slice = np.copy(initial_points[:, -1])
                        initial_points = np.transpose(np.vstack([np.transpose(initial_points), last_slice]))
                        continue
                    for i in range(0, initial_points.shape[1]):
                        if self.options["seq"] == np.array([3, 1, 2]):  # Fuselage beam
                            current_span = initial_points[0, i]
                        else:  # Wing beam
                            current_span = initial_points[1, i]
                        if not found_lower_point and (joint_location >= current_span):
                            found_lower_point = True
                            load_point = np.zeros(3)
                            if self.options["seq"] == np.array([3, 1, 2]):  # Fuselage beam
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
                            if span_percentage > 0.0:  # only duplicate point IF that point did not exist before
                                initial_points = np.insert(initial_points, i + 1, load_point, axis=1)
                    # Finally add the load subsystem to the beam:
                    self.add_subsystem(a_joint.joint_label, a_joint.component)
        self.options["r0"] = initial_points  # Storing the augmented point structure
        self.options['num_divisions'] = initial_points.shape[1]
        # Calculate th0
        self.options['th0'] = calculate_th0(r0=self.options['r0'], seq=self.options["seq"])
        # Calculate delta_s0
        self.options['delta_s0'] = np.zeros(self.options['num_divisions'] - 1)
        for i in range(0, self.options['num_divisions'] - 1):
            self.options['delta_s0'][i] = sqrt(
                (self.options["r0"][0, i] - self.options["r0"][0, i + 1]) ** 2 + (
                            self.options["r0"][1, i] - self.options["r0"][1, i + 1]) ** 2 + (
                            self.options["r0"][2, i] - self.options["r0"][2, i + 1]) ** 2)

        # Initial guess values
        x0 = np.zeros((18, self.options['num_divisions']))
        xDot0 = np.zeros((18, self.options['num_divisions']))
        x0[0:3, :] = self.options["r0"]
        x0[3:6, :] = self.options['th0']
        self.options['x0'] = np.resize(np.transpose(x0), (18 * self.options['num_divisions']))
        self.options['xDot0'] = np.resize(np.transpose(xDot0), (18 * self.options['num_divisions']))

        # Boundary conditions
        if self.options['beam_bc'] == 'Cantilever':
            self.BC['root'][0:3] = self.options['r0'][0:3, 0]
            self.BC['root'][3:6] = self.options['th0'][0:3, 0]
        elif self.options['beam_bc'] == 'Free-Free':
            raise NotImplementedError
        else:
            raise IOError
