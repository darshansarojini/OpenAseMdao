import openmdao.api as om
from abc import ABC
from casadi import *
import math
import numpy as np

class AdjointSolver(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('order', types=int, default=2)
        self.options.declare('tolerance', types=float, default=1e-8)
        self.options.declare('num_timesteps')
        self.options.declare('time_step')
        self.symbolic_functions_state = {}
        self.symbolic_functions_constraint = {}

    def setup(self):
        self.add_input('x', shape=(self.symbolic_functions_state['dRdx'].size_out(0)[0], self.options['num_timesteps']))
        self.add_input('xDot', shape=(self.symbolic_functions_state['dRdxDot'].size_out(0)[0], self.options['num_timesteps']))

        self.add_input('fh_dRdx', shape=(self.symbolic_functions_state['dRdx'].size_out(0)[0], self.symbolic_functions_state['dRdx'].size_out(0)[1]))
        self.add_input('fh_dRdxDot', shape=(self.symbolic_functions_state['dRdxDot'].size_out(0)[0], self.symbolic_functions_state['dRdxDot'].size_out(0)[1]))
        self.add_input('fh_dFdx', shape=(self.symbolic_functions_state['fh_dFdx'].size_out(0)[0], self.symbolic_functions_state['fh_dFdx'].size_out(0)[1]))
        self.add_input('fh_dRdxDot', shape=(self.symbolic_functions_state['fh_dRdxDot'].size_out(0)[0], self.symbolic_functions_state['fh_dRdxDot'].size_out(0)[1]))

        self.add_output('adjoint', shape=(self.options['num_timesteps'], self.symbolic_functions_state['dRdx'].size_out(0)[0]))

    def solve_nonlinear(self, inputs, outputs):
        x = inputs['x'].T
        dDot = inputs['xDot'].T

        # Lets integrate backwards:
        for i in range(self.options['num_timesteps'], -1, -1):
            alpha = AdjointSolver.BFD_coefficients(i, self.options['order'], self.options['time_step'])
            dRdxDot = self.symbolic_functions_state['dRdxDot'](inputs['x'][:, i], inputs['xDot'][:, i]).sparse()

            if i == self.options['num_timesteps']:
                #TODO: REQUIRES MUCH MORE THAN JUST MEDIOCRE INPUTS, PERHAPS CONNECT WITH STICKMODEL, OR SUMMON FANCY LAMBDA FUNCTION WITH HISTORY INCLUDED
                dFdx = self.symbolic_functions_state['fh_dFdx'](inputs['x'][:, i], inputs['xDot'][:, i])
                dFdxDot = self.symbolic_functions_state['fh_dRdxDot'](inputs['x'][:, i], inputs['xDot'][:, i])

    @staticmethod
    def BFD_coefficients(i, order, delta_t):
        k0 = 1 / delta_t
        k1 = -k0
        alpha = np.asarray([k0, k1])

        if order == 2 or order > 2:
            if i >= 3:
                k0 = 1.5 * (1 / delta_t)
                k1 = -2 * (1 / delta_t)
                k2 = 0.5 * (1 / delta_t)
                alpha = np.asarray([k0, k1, k2])
            else:
                k0 = 1 / delta_t
                k1 = -k0
                alpha = np.asarray([k0, k1])

        return alpha


