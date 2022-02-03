import openmdao.api as om
from pint import UnitRegistry as ureg
import numpy as np

class BeamDefinition:
    def __init__(self, identifier, points, orientation, E, G, rho):
        # To provide parameters to the constructor WITH units please use the following format:
        # E = Q_(E_val, 'pascal') -> beam1 = inputs.BeamDefinition(...,...,E,...)

        self.beam_identifier = identifier  # UNIQUE identifier that allows reference by name when joints are used
        self.beam_points = points  # array of points with units, 3 x number of nodes defining the shape
        self.E = E  # young's modulus of the material in the beam
        self.G = G  # shear modulus of the material in the beam
        self.rho = rho  # material's density of the beam
        self.orientation = orientation  # This defines whether the beam is a fuselage beam or a wing beam

        beam_component = om.IndepVarComp(name=identifier)

        # Parameters with no units
        beam_component.add_output("orientation", val=self.orientation)

        # Parameters with units
        beam_component.add_output("points", val=self.beam_points.magnitude, units="meter")
        beam_component.add_output("E", val=self.E.magnitude, units=self.E.units)
        beam_component.add_output("G", val=self.G.magnitude, units=self.G.units)
        beam_component.add_output("rho", val=self.rho.magnitude, units=self.rho.units)

        self.component = beam_component


class PointLoadDefinition:
    def __init__(self, name, eta, vector_force=np.array([0.0, 0.0, 0.0]) * ureg['newton'],
                 vector_moment=np.array([0.0, 0.0, 0.0]) * ureg['newton*meter']):
        # To provide vector load to the constructor WITH units please use the following format:
        # F = np.array([1000,0,30000]) * ureg['newton']

        self.load_label = name
        self.eta = eta
        self.vector_force = vector_force
        self.vector_moment = vector_moment

        load_component = om.IndepVarComp(name=name)

        # Unit-less parameters
        load_component.add_output("eta", val=self.eta)
        # Parameters with units
        load_component.add_output("point_force", val=self.vector_force.magnitude, units=self.vector_force.units)
        load_component.add_output("point_moment", val=self.vector_moment.magnitude, units=self.vector_moment.units)

        self.component = load_component


class JointDefinition:
    def __init__(self, name, parent_beam, parent_eta, child_beam, child_eta):

        self.joint_label = name         # Identifier that allows the user distinguish different joints
        self.parent_beam = parent_beam  # Beam from which the joint originates from
        self.parent_eta = parent_eta  # Eta span in the parent beam (including loads/joints) where the joint is located)
        self.child_beam = child_beam  # Beam that the joint extends to
        self.child_eta = child_eta    # Eta span in the child beam (including loads/joints) where the joint is located)