from openasemdao import Q_
import openmdao.api as om


class BeamDefinition:
    def __init__(self, beam_identifier, n_nodes, beam_length, E, G, rho):
        # To provide parameters to the constructor WITH units please use the following format:
        # my_length = Q_(beam_length, 'meter') -> beam1 = inputs.BeamDefinition(...,...,my_length,...)

        self.n_nodes = n_nodes  # Number of nodes (not counting point loads) that the beam must have
        self.beam_identifier = beam_identifier  # UNIQUE identifier that allows reference by name when joints are used
        self.beam_length = beam_length  # length of the beam in Meters
        self.E = E  # young's modulus of the material in the beam
        self.G = G  # shear modulus of the material in the beam
        self.rho = rho  # material's density of the beam

        beam_component = om.IndepVarComp(name=beam_identifier)
        # Unit-less parameters
        beam_component.add_output("n_nodes", val=self.n_nodes)
        # Parameters with units
        beam_component.add_output("length", val=self.beam_length.magnitude, units="meter")
        beam_component.add_output("E", val=self.E.magnitude, units="pascal")
        beam_component.add_output("G", val=self.G.magnitude, units="pascal")
        beam_component.add_output("rho", val=self.rho.magnitude, units="kg/meter**3")

        self.component = beam_component

class PointLoadDefinition:
    def __init__(self, name, eta, vector_load):
        # To provide vector load to the constructor WITH units please use the following format:
        # F = np.array([1000,0,30000]) * ureg['newton']

        self.load_label = name
        self.eta = eta
        self.vector_load = vector_load

        load_component = om.IndepVarComp(name=name)

        # Unit-less parameters
        load_component.add_output("eta", val=self.eta)
        # Parameters with units
        load_component.add_output("load", val=self.vector_load.magnitude, units=self.vector_load.units)

        self.component = load_component