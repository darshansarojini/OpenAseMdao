import openmdao.api as om
from abc import ABC
from casadi import *
import math
import numpy as np

from openasemdao.structures.utils.utils import CalcNodalT

class SymbolicStickModel(ABC):

    def AddJointConcLoads(self, x, JointProp, c_Forces, c_Moments):
        n = (x.shape[0] - 12*JointProp['Parent'].shape[1]) / 18
        for k in range(0, JointProp['Parent'].shape[1]):
            F_J = x[18 * (n) + 6 + 12 * (k - 1): 18 * (n) + 9 + 12 * (k - 1)]
            M_J = x[18 * (n) + 9 + 12 * (k - 1): 18 * (n) + 12 + 12 * (k - 1)]
            curr_low_bound = c_Forces['inter_node_lim'][JointProp['Parent'][k], 0]    # Get the location of parent start in forces
            c_Forces['delta_Fapplied'][:, curr_low_bound + JointProp['Parent_NodeNum'][k] - 1] = c_Forces['delta_Fapplied'][:, curr_low_bound + JointProp['Parent_NodeNum'][k] - 1] + F_J
            c_Moments['delta_Mapplied'][:, curr_low_bound + JointProp['Parent_NodeNum'][k] - 1] = c_Moments['delta_Mapplied'][:, curr_low_bound + JointProp['Parent_NodeNum'][k] - 1] + M_J
        return c_Forces, c_Moments

    def ResJac(self, beam_list, X, Xd, X_AC, Forces, Moments, BC, g, element, R_prec):
        num_variables = 18
        num_nodes = X.size(2)
        Res = SX.sym('Fty', num_variables, num_nodes)

        n = num_nodes

        # section Read Inputs

        # read x
        r = X[0:3, :]
        theta = X[3:6, :]
        F = X[6:9, :]
        M = X[9:12, :]
        u = X[12:15, :]
        omega = X[15:18, :]

        # read xDot
        rDot = Xd[0:3, :]
        thetaDot = Xd[3:6, :]
        FDot = Xd[6:9, :]
        MDot = Xd[9:12, :]
        uDot = Xd[12:15, :]
        omegaDot = Xd[15:18, :]

        # read X_ac (aircraft states)
        R = X_AC[0:3]
        U = X_AC[3:6]
        A0 = X_AC[6:9]
        THETA = X_AC[9:12]
        OMEGA = X_AC[12:15]
        ALPHA0 = X_AC[15:18]

        # Read forces and moments
        f_aero = Forces['f_aero'][:, Forces['node_lim'][element, 0]:Forces['node_lim'][element, 1]]
        m_aero = Moments['m_aero'][:, Moments['node_lim'][element, 0]:Moments['node_lim'][element, 1]]
        delta_Fapplied = Forces['delta_Fapplied'][:,
                         Forces['inter_node_lim'][element, 0]:Forces['inter_node_lim'][element, 1]]
        delta_Mapplied = Moments['delta_Mapplied'][:,
                         Moments['inter_node_lim'][element, 0]:Moments['inter_node_lim'][element, 1]]

        # Read Stick Model
        mu = beam_list.symbolic_expressions['mu'][
             beam_list.options['inter_node_lim'][element, 0]:beam_list.options['inter_node_lim'][
                 element, 1]]  # 1xn vector of mass/length
        seq = beam_list.options['seq'][3 * element: 3 + 3 * element]
        theta0 = beam_list.options['th0'][
                 beam_list.options['node_lim'][element, 0]:beam_list.options['node_lim'][element, 1], :]
        K0a = beam_list.options['K0a'][
              beam_list.options['inter_node_lim'][element, 0]:
              beam_list.options['inter_node_lim'][element, 1], :, :]
        delta_s0 = beam_list.options['delta_s0'][
                   beam_list.options['inter_node_lim'][element, 0]:beam_list.options['inter_node_lim'][element, 1]]

        i_matrix = SX.sym('i_s', 3, 3, n - 1)
        delta_rCG_tilde = SX.sym('d_rCG_tilde', 3, 3, n - 1)
        Einv = SX.sym('Ei', 3, 3, n)
        D = SX.sym('D', 3, 3, n)
        oneover = SX.sym('oo', 3, 3, n)

        # Do nodal quantities of symbolic pieces in 3D matrices:
        j = 0

        for i in range(beam_list.options['node_lim'][element, 0], beam_list.options['node_lim'][element, 1]):
            Einv[j][:, :] = beam_list.symbolic_expressions['Einv'][i][:, :]
            D[j][:, :] = beam_list.symbolic_expressions['D'][i][:, :]
            oneover[j][:, :] = beam_list.symbolic_expressions['oneover'][i][:, :]
            j = j + 1

        # Do element quatities of symbolic pieces in 3D matrices:
        j = 0

        for i in range(beam_list.options['inter_node_lim'][element, 0],
                       beam_list.options['inter_node_lim'][element, 1]):
            delta_rCG_tilde[j][:, :] = beam_list.symbolic_expressions['delta_r_CG_tilde'][j][:, :]
            i_matrix[j][:, :] = beam_list.symbolic_expressions['i_matrix'][j][:, :]
            j = j + 1

        # Get a cg:
        a_cg = self.calc_a_cg(r, u, uDot, omega, omegaDot, delta_rCG_tilde, A0, OMEGA, ALPHA0)

        # Get T and K matrices:
        T, Ta = CalcNodalT(theta.T, seq, n=n)
        K, Ka = self.CalcNodalK(theta, seq)

        #  Gravity in body fixed axes (STILL NON SIMBOLIC!!!)
        T_E = self.calcT_ac(THETA)  # UNS, Eq. 6, Page 5
        g_xyz = mtimes(transpose(T_E), g)
        f_acc = SX.sym('f_acc', 3, n - 1)
        m_acc = SX.sym('m_acc', 3, n - 1)

        for ind in range(0, n - 1):
            f_acc[:, ind] = mtimes(mu[ind], (g_xyz - a_cg[:, ind]))
            TiT = mtimes((0.5 * (transpose(T[ind][:, :]) + transpose(T[ind + 1][:, :]))),
                         mtimes(i_matrix[ind][:, :], (0.5 * (T[ind][:, :] + T[ind + 1][:, :]))))
            m_acc[:, ind] = mtimes(delta_rCG_tilde[ind][:, :], f_acc[:, ind]) - mtimes(TiT, ALPHA0 + (
                    0.5 * (omegaDot[:, ind] + omegaDot[:, ind + 1]))) - cross(
                (OMEGA + 0.5 * (omega[:, ind] + omega[:, ind + 1])),
                mtimes(TiT, (OMEGA + 0.5 * (omega[:, ind] + omega[:, ind + 1]))))

        Mcsn = SX.sym('Mcsn', 3, n)
        Fcsn = SX.sym('Fcsn', 3, n)
        Mcsnp = SX.sym('Mcsnp', 3, n)
        strainsCSN = SX.sym('strainsCSN', 3, n)
        damp_MK = SX.sym('damp_MK', 3, n)

        for ind in range(0, n):
            # Transform xyz -> csn (ASW, Eq. 14, page 6);
            Mcsn[:, ind] = mtimes(T[ind][:, :], M[:, ind])
            Fcsn[:, ind] = mtimes(T[ind][:, :], F[:, ind])

            # Get Mcsn_prime (ASW, Eq. 18, page 8)
            Mcsnp[:, ind] = Mcsn[:, ind] + mtimes(transpose(D[ind][:, :]), Fcsn[:, ind])

            # Get strains (ASW, Eq. 19, page 8)
            strainsCSN[:, ind] = mtimes(oneover[ind][:, :], Fcsn[:, ind]) + mtimes(D[ind][:, :], mtimes(Einv[ind][:, :],
                                                                                                        Mcsnp[:, ind]))

            # Get damping vector for moment-curvature relationship
            damp_MK[:, ind] = mtimes(inv(K[ind][:, :]), mtimes(T[ind][:, :], omega[:,ind]))

        damp = SX.zeros(3, 3)
        damp[0, 0] = self.t_kappa_c
        damp[1, 1] = self.t_kappa_s
        damp[2, 2] = self.t_kappa_n
        # get total distributed force
        f = f_aero + f_acc
        m = m_aero + m_acc

        # Get average nodal reaction forces
        Fav = SX.sym('Fav', 3, n - 1)

        for i in range(0, n - 1):
            Fav[:, i] = 0.5 * (F[:, i] + F[:, i + 1])
        # endsection

        eps = 1e-19
        # get delta_s and delta_r
        delta_r = SX.sym('delta_r', 3, n - 1)
        delta_s = SX.sym('delta_s', 3, n - 1)
        # moment-curvature debugger:
        mc_static = SX.sym('mc', 3, n)
        for i in range(0, n - 1):
            delta_r[:, i] = (r[:, i + 1] - r[:,i] + eps)  # Added a non zero number to avoid the 1/sqrt(dx) singularity at the zero length nodes
            delta_s[i] = sqrt(
                (delta_r[0, i]) ** 2 + (delta_r[1, i]) ** 2 + (delta_r[2, i]) ** 2)  # based on ASW, Eq. 49, Page 12

        # section Residual
        for i in range(0, n):
            if i <= n - 2:
                # Rows 1-3: force equilibrium (ASW, Eq. 56, page 13)
                Res[0:3, i] = R_prec[0] * (F[:, i + 1] - F[:, i] + mtimes(f[:, i], delta_s[i]) + delta_Fapplied[:, i])

                # Rows 4-6: moment equilibrium (ASW, Eq. 55, page 13)
                Res[3:6, i] = R_prec[1] * (
                        M[:, i + 1] - M[:, i] + m[:, i] * delta_s[i] + delta_Mapplied[:, i] + cross(delta_r[:, i],
                                                                                                    Fav[:, i]))

                # Rows 7-9: moment-curvature relationship (ASW, Eq. 54, page 13)
                Res[6:9, i] = R_prec[2] * (mtimes(Ka[i][:, :], (theta[:, i + 1] - theta[:, i])) - mtimes(K0a[i, :, :], (
                        theta0[i + 1, :] - theta0[i, :])) - 0.25 * mtimes((Einv[i][:, :] + Einv[i + 1][:, :]), (
                                                   Mcsnp[:, i] + Mcsnp[:, i + 1])) * delta_s[i] + mtimes(damp, (
                        mtimes(Ka[i][:, :], (damp_MK[:, i + 1] - damp_MK[:, i])) + 0.5*mtimes((
                        K[i + 1][:, :] - K[i][:, :]), (damp_MK[:, i + 1] + damp_MK[:, i])))))
                # Rows 10-12: strain-displacement (ASW, Eq. 48, page 12)
                s_vec = SX.zeros(3, 1)
                s_vec[1] = 1
                tempVector = s_vec + 0.5 * (strainsCSN[:, i] + strainsCSN[:, i + 1])

                # ------------------There is some serious leackage here------------------
                Res[9:12, i] = R_prec[3] * (
                        r[:, i + 1] - r[:, i] - delta_s0[i] * mtimes(Ta[i][:, :].T, tempVector) + mtimes(
                    damp, ((u[:, i + 1] - u[:, i]) - cross((0.5 * (omega[:, i + 1] + omega[:, i])),
                                                           (r[:, i + 1] - r[:, i])))))

                mc_static[0:3, i] = r[:, i + 1] - r[:, i] - delta_s0[i] * mtimes(Ta[i][:, :].T, tempVector)
                # Rows 13-16
                Res[12:15, i] = R_prec[4] * (u[:, i] - rDot[:, i])

                # Rows 17-19 (UNS, Eq. 2, page 4);
                Res[15:18, i] = R_prec[5] * (omega[:, i] - mtimes(mtimes(T[i][:, :].T, K[i][:, :]), thetaDot[:, i]))
            else:
                Res[12:15, i] = R_prec[10] * (u[:, i] - rDot[:, i])
                Res[15:18, i] = R_prec[11] * (
                        omega[:, i] - mtimes(T[i][:, :].T, mtimes(K[i][:, :], thetaDot[:, i])))
                # BOUNDARY CONDITIONS *****************************************
                # ****potential source of issues, should we include rDot and thetaDot or u and omega in the options?
                BCroot = BC['root']
                BCtip = BC['tip']
                # potential variables to be set as bc
                varRoot = SX.sym('vr', 12, 1)
                varTip = SX.sym('vt', 12, 1)

                varRoot[0:3] = r[:, 0]
                varRoot[3:6] = theta[:, 0]
                varRoot[6:9] = F[:, 0]
                varRoot[9:12] = M[:, 0]

                varTip[0:3] = r[:, i]
                varTip[3:6] = theta[:, i]
                varTip[6:9] = F[:, i]
                varTip[9:12] = M[:, i]

                # indices that show which variables are to be set as bc (each will return 6 indices)
                indicesRoot_ = (~(BCroot == 8888))
                indicesTip_ = (~(BCtip == 8888))

                indicesRoot = []
                indicesTip = []
                for k in range(0,len(indicesRoot_)):
                    if indicesRoot_[k]:
                        indicesRoot.append(k)
                    if indicesTip_[k]:
                        indicesTip.append(k)
                # root
                Res[0:3, i] = mtimes(R_prec[6], (varRoot[indicesRoot[0:3]] - BCroot[indicesRoot[0:3]]))
                Res[3:6, i] = mtimes(R_prec[7], (varRoot[indicesRoot[3:6]] - BCroot[indicesRoot[3:6]]))
                # tip
                Res[6:9, i] = mtimes(R_prec[8], (varTip[indicesTip[0:3]] - BCtip[indicesTip[0:3]]))
                Res[9:12, i] = mtimes(R_prec[9], (varTip[indicesTip[3:6]] - BCtip[indicesTip[3:6]]))

                mc_static[0:3, i] = mtimes(R_prec[9], (varTip[indicesTip[3:6]] - BCtip[indicesTip[3:6]]))
        # Debugging functions
        self.symbolics['mc_static'] = mc_static
        # endsection
        return Res

    def JoinResJac(self, beam_list, Residuals, JointProp, X):
        # ********** Use for Jacobian
        # function [Res, Jac] = JointResiduals(stickModel_1,stickModel_2,JointProp, ...
        #     Static_x_1, Static_x_2, r_J, th_J, F_J, M_J)
        # ************************
        #
        # It is assumed that the variables are organized such that the Static_x_1
        # (Joint Point 1) comes first and then Static_x_2 (Joint Point 2)

        # JointProp = {}  # Properties:
        # JointProp['Parent_NodeNum'] = [13, 3, 3, 6, 6]  # Index on the parent where the joint is located
        # JointProp['Parent_r0'] = [3 x 5 double]  # r_0 where the joint is at the parent level
        # JointProp['Parent_th0'] = [3 x 5 double]  # th_0 where the joint is at the parent level
        # JointProp['Parent'] = [0, 0, 0, 1, 1]  # Part index that shows what the parent is, based on the insertion order (starting from 0)
        # JointProp['Child'] = [1, 2, 3, 4, 5]  # Part index that shows what the child is for every joint (starting from 0)
        # JointProp['Child_NodeNum'] = [0, 0, 0, 0, 0]  # Node index that shows what the child is for every joint (starting from 0)
        # JointProp['Child_r0'] = [3 x 5 double]  # r_0 where the joint is at the child level per part
        # JointProp['Child_th0'] = [3 x 5 double]  # th_0 where the joint is at the child level per part

        # Pre-declaring variables
        r_J = SX.zeros(3, 1)
        th_J = SX.zeros(3, 1)
        n = X.size(2)
        Res = SX.sym('Fty', JointProp['Child'].size(2) * 12, 1)

        for k in range(0, JointProp['Child'].size(2)):
            r_J = X[18 * (n) + 0 + 12 * (k - 1): 18 * (n) + 3 + 12 * (k - 1)]
            th_J = X[18 * (n) + 3 + 12 * (k - 1): 18 * (n) + 6 + 12 * (k - 1)]
            F_J = X[18 * (n) + 6 + 12 * (k - 1): 18 * (n) + 9 + 12 * (k - 1)]
            M_J = X[18 * (n) + 9 + 12 * (k - 1): 18 * (n) + 12 + 12 * (k - 1)]

            parent_index = JointProp['Parent'][k]
            child_index = JointProp['Child'][k]

            # Node numbers of joint 1 and joint 2
            NodeNum_JP1 = JointProp['Parent_NodeNum'][k]
            NodeNum_JP2 = JointProp['Child_NodeNum'][k]

            # region Residual of kinematic constraints
            starting_node_parent = 18 * (beam_list['node_lim'][parent_index, 0] - 1)  # ZERO entry. Where the part residual entry starts
            starting_node_child = 18 * (beam_list['node_lim'][child_index, 0] - 1)    # ZERO entry. Where the part residual entry starts

            r10 = JointProp['Parent_r0'][:, k]  # stickModel_1.r0(:,NodeNum_JP1)
            r20 = JointProp['Child_r0'][:, k]  # stickModel_2.r0(:,NodeNum_JP2)
            th10 = JointProp['Parent_th0'][:, k]  # stickModel_1.th0(:,NodeNum_JP1)
            th20 = JointProp['Child_th0'][:, k]  # stickModel_2.th0(:,NodeNum_JP2)

            r1 = X[starting_node_parent + 18 * (NodeNum_JP1 - 1) + 0:starting_node_parent + 18 * (NodeNum_JP1 - 1) + 3]
            th1 = X[starting_node_parent + 18 * (NodeNum_JP1 - 1) + 3:starting_node_parent + 18 * (NodeNum_JP1 - 1) + 6]

            r2 = r20 + r_J
            th2 = th20 + th_J

            # Create the matrices

            T1 = SymbolicStickModel.RotationMatrix(beam_list['seq'][1 + 3 * (parent_index - 1):3 + 3 * (parent_index - 1)], th1)
            T10 = SymbolicStickModel.RotationMatrix(beam_list['seq'][1 + 3 * (parent_index - 1):3 + 3 * (parent_index - 1)], th10)
            T2 = SymbolicStickModel.RotationMatrix(beam_list['seq'][1 + 3 * (child_index - 1):3 + 3 * (child_index - 1)], th2)
            T20 = SymbolicStickModel.RotationMatrix(beam_list['seq'][1 + 3 * (child_index - 1):3 + 3 * (child_index - 1)], th20)

            Res[12 * (k - 1) + 0: 12 * (k - 1) + 3] = r2 - r1 - mtimes(transpose(T1), mtimes(T10, (r20-r10)))

            temp1 = mtimes(transpose(T1), T10)
            temp2 = mtimes(transpose(T2), T20)

            Res[12 * (k - 1) + 3] = dot(temp1[:, 1], temp2[:, 2]) - dot(temp1[:, 2], temp2[:, 1])
            Res[12 * (k - 1) + 4] = dot(temp1[:, 2], temp2[:, 0]) - dot(temp1[:, 0], temp2[:, 2])
            Res[12 * (k - 1) + 5] = dot(temp1[:, 0], temp2[:, 1]) - dot(temp1[:, 1], temp2[:, 0])

            # endregion

            # region Residual of F&M equations

            # Form the required variables
            Mi1 = X[starting_node_child + 18 * (NodeNum_JP2) + 9:starting_node_child + 18 * (NodeNum_JP2) + 12]
            Mi = X[starting_node_child + 18 * (NodeNum_JP2 - 1) + 9:starting_node_child + 18 * (NodeNum_JP2 - 1) + 12]
            Fi1 = X[starting_node_child + 18 * (NodeNum_JP2) + 6:starting_node_child + 18 * (NodeNum_JP2) + 9]
            Fi = X[starting_node_child + 18 * (NodeNum_JP2 - 1) + 6:starting_node_child + 18 * (NodeNum_JP2 - 1) + 9]


            # Assumptions (for now)
            deltaF = np.zeros((3, 1))
            deltaM = np.zeros((3, 1))

            # Force equation residual
            Res[12 * (k - 1) + 6: 12 * (k - 1) + 9] = Fi1 - Fi + deltaF - F_J

            # Moment equation residual
            temp3 = (r2 - r1)
            temp3_tilde = np.zeros((3, 3))
            temp3_tilde[0, 0] = 0
            temp3_tilde[0, 1] = -temp3[2]
            temp3_tilde[0, 2] = temp3[1]
            temp3_tilde[1, 0] = temp3[2]
            temp3_tilde[1, 1] = 0
            temp3_tilde[1, 2] = -temp3[0]
            temp3_tilde[2, 0] = -temp3[1]
            temp3_tilde[2, 1] = temp3[0]
            temp3_tilde[2, 2] = 0

            Res[12 * (k - 1) + 9: 12 * (k - 1) + 12] = Mi1 - Mi + deltaM - M_J + mtimes(temp3_tilde, F_J)

            # endregion
        Residuals[18 * n: 18 * n + 12 * JointProp['Child'].size(2)] = Res

    @staticmethod
    def RotationMatrix(seq, th):
        a1 = th[0]
        a2 = th[1]
        a3 = th[2]
        R = SX.sym('R', 3, 3, 3)

        # rotation tensor for "phi" rotation (angle a1)
        R[0][0, 0] = 1
        R[0][0, 1] = 0
        R[0][0, 2] = 0
        R[0][1, 0] = 0
        R[0][1, 1] = cos(a1)
        R[0][1, 2] = sin(a1)
        R[0][2, 0] = 0
        R[0][2, 1] = -sin(a1)
        R[0][2, 2] = cos(a1)

        # rotation tensor for "theta" rotation (angle a2)
        R[1][0, 0] = cos(a2)
        R[1][0, 1] = 0
        R[1][0, 2] = -sin(a2)
        R[1][1, 0] = 0
        R[1][1, 1] = 1
        R[1][1, 2] = 0
        R[1][2, 0] = sin(a2)
        R[1][2, 1] = 0
        R[1][2, 2] = cos(a2)

        # rotation tensor for "psi" rotation (angle a3)
        R[2][0, 0] = cos(a3)
        R[2][0, 1] = sin(a3)
        R[2][0, 2] = 0
        R[2][1, 0] = -sin(a3)
        R[2][1, 1] = cos(a3)
        R[2][1, 2] = 0
        R[2][2, 0] = 0
        R[2][2, 1] = 0
        R[2][2, 2] = 1

        # multiply single-axis rotation tensors in reverse order
        T = mtimes(R[seq[2]], mtimes(R[seq[1]], R[seq[0]]))

        return T

    def ModifyJointPoint2Residuals(self, beam_list, Residuals, X, JointProp):
        # ************ When the joints jacobian is being done, this function is used
        # [Res, Jac] = ModifyJointPoint2Residuals(Res,Jac,Static_x,JointProp,r_J,th_J)
        # *************************************

        # The manual says to introduce the kinematic constraint
        # across the structural interval where the joint is located. The
        # residual matrix is of size 18 X n. The first 6 rows and n-1 columns are
        # the force and moment equilibrium equations written for an element.
        # Element 1 is for node 1 and node 2. Element 2 is for node 2 and node 3.
        # and so on...
        # JointProp.Point2.NodeNum has the index of the node where the joint is
        # present. Hence, the element where the equations need to be applied is on
        # that element
        # The interval has been chosen to have zero length

        n = beam_list['r0'].size(2)
        for k in range(0, JointProp['Child'].shape[2]):
            r_J = X[18 * (n) + 0 + 12 * (k - 1): 18 * (n) + 3 + 12 * (k - 1)]
            th_J = X[18 * (n) + 3 + 12 * (k - 1): 18 * (n) + 6 + 12 * (k - 1)]
            child_joint = JointProp['Child'][k]
            starting_node = 18 * (beam_list.options['node_lim'][child_joint, 0] - 1)        # ZERO entry. Where the part residual entry starts
            Residuals[starting_node+18*(JointProp['Child_NodeNum'][k]-0)+0:starting_node+18*(JointProp['Child_NodeNum'][k]-1)+3] = X[starting_node+18*(JointProp['Child_NodeNum'][k]-1)+0:starting_node+18*(JointProp['Child_NodeNum'][k]-1)+3]-JointProp['Child_r0'][:, k]-r_J
            Residuals[starting_node + 18 * (JointProp['Child_NodeNum'][k] - 0) + 3:starting_node + 18 * (JointProp['Child_NodeNum'][k] - 1) + 6] = X[starting_node+18*(JointProp['Child_NodeNum'][k]-1)+3:starting_node+18*(JointProp['Child_NodeNum'][k]-1)+6]-JointProp['Child_th0'][:, k]-th_J
        return Residuals

    def calc_a_cg(self, r, u, uDot, omega, omegaDot, delta_rCG_tilde, A0, OMEGA, ALPHA0):
        # Initialize variables
        n = max(r.size())
        aCG = SX.sym('aCG', 3, n - 1)
        a_i = SX.sym('a_i', 3, n)
        for i in range(0, n):
            # current node quantities:
            ri = r[:, i]
            ui = u[:, i]
            uDoti = uDot[:, i]
            riT = SX.sym('riT', 3, 3)
            uiT = SX.sym('uiT', 3, 3)
            inner3T = SX.sym('inner3T', 3, 3)
            # for OMEGA X (OMEGA X ri):
            riT[0, 0] = 0
            riT[0, 1] = -ri[2]
            riT[0, 2] = ri[1]
            riT[1, 0] = ri[2]
            riT[1, 1] = 0
            riT[1, 2] = -ri[0]
            riT[2, 0] = -ri[1]
            riT[2, 1] = ri[0]
            riT[2, 2] = 0

            inner3 = mtimes(riT, OMEGA)

            inner3T[0, 0] = 0
            inner3T[0, 1] = -inner3[2]
            inner3T[0, 2] = inner3[1]
            inner3T[1, 0] = inner3[2]
            inner3T[1, 1] = 0
            inner3T[1, 2] = -inner3[0]
            inner3T[2, 0] = -inner3[1]
            inner3T[2, 1] = inner3[0]
            inner3T[2, 2] = 0

            # for OMEGA X ui
            uiT[0, 0] = 0
            uiT[0, 1] = -ri[2]
            uiT[0, 2] = ri[1]
            uiT[1, 0] = ri[2]
            uiT[1, 1] = 0
            uiT[1, 2] = -ri[0]
            uiT[2, 0] = -ri[1]
            uiT[2, 1] = ri[0]
            uiT[2, 2] = 0

            # a_i (UNS, Eq. 23, Page 7)
            a_i[:, i] = A0 + uDoti + mtimes(riT, ALPHA0) + mtimes(inner3T, OMEGA) + 2 * mtimes(uiT, OMEGA)

        # Acceleration of the element
        for i in range(0, n - 1):
            innerT = SX.sym('innerT', 3, 3)
            inner2T = SX.sym('inner3T', 3, 3)
            # current element quantities
            drCG = delta_rCG_tilde[i][:, :]
            # current node quantities
            omi = omega[:, i]
            omDoti = omegaDot[:, i]
            ai = a_i[:, i]
            # next node quantities
            omi1 = omega[:, i + 1];
            omDoti1 = omegaDot[:, i + 1]
            ai1 = a_i[:, i + 1]

            # for OMEGA X (OMEGA X delta_rCG)
            inner = mtimes(drCG, OMEGA)
            innerT[0, 0] = 0
            innerT[0, 1] = -inner[2]
            innerT[0, 2] = inner[1]
            innerT[1, 0] = inner[2]
            innerT[1, 1] = 0
            innerT[1, 2] = -inner[0]
            innerT[2, 0] = -inner[1]
            innerT[2, 1] = inner[0]
            innerT[2, 2] = 0
            # for omega_i X (omega_i X delta_rCG)
            inner2 = mtimes(drCG, (0.5 * (omi + omi1)))
            inner2T[0, 0] = 0
            inner2T[0, 1] = -inner2[2]
            inner2T[0, 2] = inner2[1]
            inner2T[1, 0] = inner2[2]
            inner2T[1, 1] = 0
            inner2T[1, 2] = -inner2[0]
            inner2T[2, 0] = -inner2[1]
            inner2T[2, 1] = inner2[0]
            inner2T[2, 2] = 0
            # nodal a_cg (UNS, eq. 38, Page 9)
            aCG[:, i] = 0.5 * (ai + ai1) + mtimes(drCG, (ALPHA0 + (0.5 * (omDoti + omDoti1)))) + mtimes(innerT,
                                                                                                        OMEGA) + mtimes(
                inner2T, (
                        0.5 * (omi + omi1))) + 2 * mtimes(inner2T, OMEGA)
        return aCG

    def CalcNodalK(self, th, seq):
        n = max(th.size())
        K = SX.sym('K', 3, 3, n)
        Ka = SX.sym('Ka', 3, 3, n - 1)
        for i in range(0, n):
            # surface beam
            if not (seq - [1, 3, 2]).any():
                K[i][0, 0] = cos(th[2, i]) * cos(th[1, i])
                K[i][0, 1] = 0
                K[i][0, 2] = -sin(th[1, i])
                K[i][1, 0] = -sin(th[2, i])
                K[i][1, 1] = 1
                K[i][1, 2] = 0
                K[i][2, 0] = cos(th[2, i]) * sin(th[1, i])
                K[i][2, 1] = 0
                K[i][2, 2] = cos(th[1, i])
            # fuselage beam
            if not (seq - [3, 1, 2]).any():
                K[i][0, 0] = cos(th[1, i])
                K[i][0, 1] = 0
                K[i][0, 2] = -cos(th[0, i]) * sin(th[1, i])
                K[i][1, 0] = 0
                K[i][1, 1] = 1
                K[i][1, 2] = sin(th[0, i])
                K[i][2, 0] = sin(th[1, i])
                K[i][2, 1] = 0
                K[i][2, 2] = cos(th[0, i]) * cos(th[1, i])
            if i >= 1:
                Ka[i - 1][:, :] = (K[i - 1][:, :] + K[i][:, :]) / 2
        return K, Ka

    def calcT_ac(self, TH):
        # Rotation matrix (3x3) that rotates a vector in xyz to XYZ as follows:
        # A_XYZ = T_E * A_xyz
        R_psi = SX.sym('R_psi', 3, 3)
        R_th = SX.sym('R_th', 3, 3)
        R_phi = SX.sym('R_th', 3, 3)
        # Read aircraft states (rad)
        PHI = TH[0]
        THETA = TH[1]
        PSI = TH[2]

        # Calc 3 matrices
        R_psi[0, 0] = cos(PSI)
        R_psi[0, 1] = sin(PSI)
        R_psi[0, 2] = 0
        R_psi[1, 0] = -sin(PSI)
        R_psi[1, 1] = cos(PSI)
        R_psi[1, 2] = 0
        R_psi[2, 0] = 0
        R_psi[2, 1] = 0
        R_psi[2, 2] = 1

        R_th[0, 0] = cos(THETA)
        R_th[0, 1] = 0
        R_th[0, 2] = sin(THETA)
        R_th[1, 0] = 0
        R_th[1, 1] = 1
        R_th[1, 2] = 0
        R_th[2, 0] = -sin(THETA)
        R_th[2, 1] = 0
        R_th[2, 2] = cos(THETA)

        R_phi[0, 0] = 1
        R_phi[0, 1] = 0
        R_phi[0, 2] = 0
        R_phi[1, 0] = 0
        R_phi[1, 1] = cos(PHI)
        R_phi[1, 2] = sin(PHI)
        R_phi[2, 0] = 0
        R_phi[2, 1] = -sin(PHI)
        R_phi[2, 2] = cos(PHI)

        # Calc rotation matrix
        T_E = mtimes(R_psi, mtimes(R_th, R_phi))

        return T_E

class StaticBeamStickModel(SymbolicStickModel, om.ImplicitComponent):
    def initialize(self):
        self.options.declare('load_factor', types=float)
        self.options.declare('beam_list')
        self.options.declare('joint_reference')
        self.options.declare('beam_reference')
        self.symbolics = {}
        self.symbolic_functions = {}
        self.symbolic_expressions = {}
        self.seq = []

        t_gamma = 0.1  # both values were 0.03
        t_epsilon = 0.1

        self.t_gamma_c = t_gamma
        self.t_epsilon_s = t_epsilon
        self.t_gamma_n = t_gamma
        self.t_kappa_c = t_epsilon
        self.t_kappa_s = t_gamma
        self.t_kappa_n = t_epsilon

    def setup(self):
        # First, Its necessary to generate all the symbolics necessary for the jointed system to work:

        self.create_symbolic_function(self.options['beam_list'], self.options['joint_list'])

        # Generate state and force numerical connections:
        n = math.floor(self.symbolic_expressions['Residual'].shape[0] / 18)
        self.add_input('xDot', shape=18 * n)
        self.add_input('Xac', shape=18)
        self.add_input('forces_dist', shape=3 * (n - 1))
        self.add_input('moments_dist', shape=3 * (n - 1))
        self.add_input('forces_conc', shape=3 * (n - 1))
        self.add_input('moments_conc', shape=3 * (n - 1))

        self.add_input('cs', shape=self.options['beam_reference'].symbolics['cs'].shape[0])

        # Add the outputs from this stickmodel class (just a single beam)
        self.add_output('x', shape=self.symbolic_expressions['Residual'].shape[0])
        self.declare_partials('x', 'x')

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None,
                        discrete_outputs=None):
        # self.symbolic_functions['mc_static']
        if 'x' in residuals:
            residuals['x'] = self.symbolic_functions['Residuals'](xDot=inputs['xDot'], Xac=inputs['Xac'],
                                                                  fDist=inputs['forces_dist'], mDist=inputs['moments_dist'],
                                                                  fConc=inputs['forces_conc'], mConc=inputs['moments_conc'],
                                                                  csDVs=inputs['cs'],
                                                                  x=outputs['x'])['r']
            print(np.linalg.norm(residuals['x']))

        pass

    def linearize(self, inputs, outputs, partials, discrete_inputs=None, discrete_outputs=None):

        Jac = self.symbolic_functions['Jacobian'](outputs['x'], inputs['xDot'], inputs['Xac'], inputs['forces_dist'],
                               inputs['moments_dist'], inputs['forces_conc'], inputs['moments_conc'], inputs['cs'])


        partials['x', 'x'] = Jac[0:self.symbolic_expressions['Residual'].shape[0], 0:self.symbolic_expressions['Residual'].shape[0]]
        # J = Jac[0:self.symbolic_expressions['Residual'].shape[0], 0:self.symbolic_expressions['Residual'].shape[0]]
        # R, C, R_pp, C_pp =  calculate_preconditioner(J)
        # B_J = R_pp* R *coo_matrix(J.toarray())* C *C_pp

        pass

    def solve_nonlinear(self, inputs, outputs, residuals):
        pass
        # Integrator = StaticIntegrator(tolerance=1e-9, model=self, inputs=inputs, outputs=outputs)
        # self.apply_nonlinear(inputs, outputs, residuals, None, None)

    def create_symbolic_function(self, beam, joints):
        # num_steps = math.floor((self.options['t_final'] - self.options['t_initial']) / self.options['time_step'])
        g = np.zeros(3)
        g[2] = 9.81
        R_prec = np.ones(12)
        R_prec[0] = 1e-3
        R_prec[1] = 1e-5
        R_prec[2] = 1e5
        R_prec[3] = 1
        R_prec[4] = 1e0
        R_prec[5] = 1e0
        R_prec[6] = 1e-4
        R_prec[7] = 3 * 1e3
        R_prec[8] = 1e2
        R_prec[9] = 1e0
        R_prec[10] = 1e0
        R_prec[11] = 1e0
        self.g = -self.options['load_factor'] * g

        nodes = beam.options['num_divisions']

        self.symbolics['Xac'] = SX.sym('Xac', 18)

        # Generate force/moment dictionaries for resjac:
        Forces = {}
        Forces['inter_node_lim'] = np.asarray([[0, nodes - 1]])
        Forces['node_lim'] = np.asarray([[0, nodes - 1]])
        Forces['delta_Fapplied'] = beam.symbolics['forces_conc']
        Forces['f_aero'] = beam.symbolics['forces_dist']
        Moments = {}
        Moments['inter_node_lim'] = np.asarray([[0, nodes - 1]])
        Moments['node_lim'] = np.asarray([[0, nodes - 1]])
        Moments['delta_Mapplied'] = beam.symbolics['moments_conc']
        Moments['m_aero'] = beam.symbolics['moments_dist']

        # Gather Boundary Conditions:
        self.symbolic_expressions['Residual'] = self.ResJac(beam, reshape(beam.symbolics['x_slice'], (18, nodes)),
                                                            reshape(beam.symbolics['xDot_slice'], (18, nodes)),
                                                            self.symbolics['Xac'], Forces, Moments, beam.BC, self.g, 0,
                                                            R_prec)
        self.symbolic_expressions['Residual'] = reshape(self.symbolic_expressions['Residual'], (18*nodes, 1))
        self.symbolic_functions['Residuals'] = Function("Residuals", [beam.symbolics['x_slice'], beam.symbolics['xDot_slice'],
                                                                      self.symbolics['Xac'],
                                                                      reshape(beam.symbolics['forces_dist'].T, (3*(nodes-1), 1)),
                                                                      reshape(beam.symbolics['moments_dist'].T, (3*(nodes-1), 1)),
                                                                      reshape(beam.symbolics['forces_conc'].T, (3*(nodes-1), 1)),
                                                                      reshape(beam.symbolics['moments_conc'].T, (3*(nodes-1), 1)),
                                                                      beam.symbolics['cs']],
                                                        [self.symbolic_expressions['Residual']],
                                                        ['x', 'xDot', 'Xac', 'fDist', 'mDist', 'fConc', 'mConc', 'csDVs'],
                                                        ['r'])
        self.symbolic_expressions['Jacobian'] = jacobian(self.symbolic_expressions['Residual'],
                                                         vertcat(beam.symbolics['x_slice'], beam.symbolics['xDot_slice']))

        self.symbolic_functions['Jacobian'] = Function('Jac', [beam.symbolics['x_slice'], beam.symbolics['xDot_slice'],
                                                               self.symbolics['Xac'],
                                                               reshape(beam.symbolics['forces_dist'].T, (3*(nodes-1), 1)),
                                                               reshape(beam.symbolics['moments_dist'].T, (3*(nodes-1), 1)),
                                                               reshape(beam.symbolics['forces_conc'].T, (3*(nodes-1), 1)),
                                                               reshape(beam.symbolics['moments_conc'].T, (3*(nodes-1), 1)),
                                                               beam.symbolics['cs']],
                                                       [self.symbolic_expressions['Jacobian']])
        # Debugging functions

        self.symbolic_functions['mc_static'] = Function("Residuals", [beam.symbolics['x_slice'], beam.symbolics['xDot_slice'],
                                                                      self.symbolics['Xac'],
                                                                      reshape(beam.symbolics['forces_dist'].T, (3*(nodes-1), 1)),
                                                                      reshape(beam.symbolics['moments_dist'].T, (3*(nodes-1), 1)),
                                                                      reshape(beam.symbolics['forces_conc'].T, (3*(nodes-1), 1)),
                                                                      reshape(beam.symbolics['moments_conc'].T, (3*(nodes-1), 1)),
                                                                      beam.symbolics['cs']],
                                                        [self.symbolics['mc_static']],
                                                        ['x', 'xDot', 'Xac', 'fDist', 'mDist', 'fConc', 'mConc', 'csDVs'],
                                                        ['r'])


    def ResJac_Multipart(self, Xd, X_AC, Forces, Moments, BC, g, JointProp, X, Rprec):
        # Initialize Macro-Structure
        Residuals = SX.sym('Res', Xd.shape[0])

        # Generate holders for the forces that will get modified due to the Joint Residuals:
        c_Forces = {}
        c_Forces['f_aero'] = Forces['f_aero']
        c_Forces['delta_Fapplied'] = SX.zeros(3, Forces['delta_Fapplied'].shape[1])

        c_Moments = {}
        c_Moments['m_aero'] = Moments['m_aero']
        c_Moments['delta_Mapplied'] = SX.zeros(3, Moments['delta_Mapplied'].shape[1])

        # Pass constant values into the holders from the point loads, different from the aero loads:
        c_Forces['delta_Fapplied'] = c_Forces['delta_Fapplied'] + Forces['delta_Fapplied']
        c_Forces['node_lim'] = Forces['node_lim']
        c_Forces['inter_node_lim'] = Forces['inter_node_lim']

        c_Moments['delta_Mapplied'] = c_Moments['delta_Mapplied'] + Moments['delta_Mapplied']
        c_Moments['node_lim'] = Moments['node_lim']
        c_Moments['inter_node_lim'] = Moments['inter_node_lim']

        if len(JointProp) > 0:
            #  Now lets run through each joint (all at once) and add the forces that correspond to the parent joint part
            c_Forces, c_Moments = self.AddJointConcLoads(X, JointProp, c_Forces, c_Moments)

        # Running individual elements
        n = 0
        for i in range(0, int(len(self.seq)/3)):
            n_element = self.options['beam_reference']['node_lim'][i, 1] - self.options['beam_reference']['node_lim'][i, 0] + 1
            Residuals[18 * n + 1: 18 * (n + n_element)] = self.ResJac(self.options['beam_reference'], np.reshape(X[18*n+1:18*(n+n_element)], [18,n_element]), np.reshape(Xd[18*n+1:18*(n+n_element)],[18,n_element]), X_AC, c_Forces, c_Moments, BC, g, i, Rprec)
            n = n + n_element

        if len(JointProp) > 0:
            # Modify Joint 2 Equations
            Residuals = self.ModifyJointPoint2Residuals(self.options['beam_reference'], Residuals, X, JointProp)
            # Run joint Residuals:
            Residuals = self.JoinResJac(Residuals, self.options['beam_reference'], JointProp, X)

        return Residuals