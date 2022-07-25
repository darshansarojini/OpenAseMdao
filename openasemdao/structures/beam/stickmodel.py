import openmdao.api as om
from abc import ABC
from casadi import *
import math
import numpy as np
from openasemdao.integration.integrator import Integrator
from openasemdao.structures.utils.utils import CalcNodalT


class SymbolicStickModel(ABC):

    @staticmethod
    def AddJointConcLoads(X, JointProp, c_Forces, c_Moments):
        n = int((X.shape[0] - 12 * len(JointProp['Parent'])) / 18)
        for k in range(0, len(JointProp['Parent'])):
            F_J = X[18 * n + 6 + 12 * k: 18 * n + 9 + 12 * k]
            M_J = X[18 * n + 9 + 12 * k: 18 * n + 12 + 12 * k]
            curr_low_bound = c_Forces['inter_node_lim'][JointProp['Parent'][k], 0]  # Get the location of parent start in forces
            c_Forces['delta_Fapplied'][:, curr_low_bound + JointProp['Parent_NodeNum'][k]] = c_Forces['delta_Fapplied'][:, curr_low_bound + JointProp['Parent_NodeNum'][k]] + F_J
            c_Moments['delta_Mapplied'][:, curr_low_bound + JointProp['Parent_NodeNum'][k]] = c_Moments['delta_Mapplied'][:, curr_low_bound + JointProp['Parent_NodeNum'][k]] + M_J
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
        f_aero = Forces['f_aero'][:, Forces['inter_node_lim'][element, 0]:Forces['inter_node_lim'][element, 1]]
        m_aero = Moments['m_aero'][:, Moments['inter_node_lim'][element, 0]:Moments['inter_node_lim'][element, 1]]
        delta_Fapplied = Forces['delta_Fapplied'][:, Forces['inter_node_lim'][element, 0]:Forces['inter_node_lim'][element, 1]]
        delta_Mapplied = Moments['delta_Mapplied'][:, Moments['inter_node_lim'][element, 0]:Moments['inter_node_lim'][element, 1]]

        # Read Stick Model
        mu = beam_list['mu'][beam_list['inter_node_lim'][element, 0]:beam_list['inter_node_lim'][element, 1]]  # 1xn vector of mass/length
        seq = beam_list['seq'][3 * element: 3 + 3 * element]
        theta0 = beam_list['th0'][:, beam_list['node_lim'][element, 0]:beam_list['node_lim'][element, 1]]
        K0a = beam_list['K0a'][beam_list['inter_node_lim'][element, 0]:beam_list['inter_node_lim'][element, 1], :, :]
        delta_s0 = beam_list['delta_s0'][beam_list['inter_node_lim'][element, 0]:beam_list['inter_node_lim'][element, 1]]

        i_matrix = SX.sym('i_s', 3, 3, n - 1)
        delta_rCG_tilde = SX.sym('d_rCG_tilde', 3, 3, n - 1)
        Einv = SX.sym('Ei', 3, 3, n)
        D = SX.sym('D', 3, 3, n)
        oneover = SX.sym('oo', 3, 3, n)

        # Do nodal quantities of symbolic pieces in 3D matrices:
        j = 0

        for i in range(beam_list['node_lim'][element, 0], beam_list['node_lim'][element, 1]):
            Einv[j][:, :] = beam_list['Einv'][i][:, :]
            D[j][:, :] = beam_list['D'][i][:, :]
            oneover[j][:, :] = beam_list['oneover'][i][:, :]
            j = j + 1

        # Do element quatities of symbolic pieces in 3D matrices:
        j = 0

        for i in range(beam_list['inter_node_lim'][element, 0],
                       beam_list['inter_node_lim'][element, 1]):
            delta_rCG_tilde[j][:, :] = beam_list['delta_r_CG_tilde'][j][:, :]
            i_matrix[j][:, :] = beam_list['i_matrix'][j][:, :]
            j = j + 1

        # Get a cg:
        a_cg = self.calc_a_cg(r, u, uDot, omega, omegaDot, delta_rCG_tilde, A0, OMEGA, ALPHA0)

        # Get T and K matrices:
        T, Ta = CalcNodalT(theta, seq, n=n)
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
            damp_MK[:, ind] = mtimes(inv(K[ind][:, :]), mtimes(T[ind][:, :], omega[:, ind]))

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
            delta_r[:, i] = (r[:, i + 1] - r[:, i] + eps)  # Added a non zero number to avoid the 1/sqrt(dx) singularity at the zero length nodes
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
                        theta0[:, i + 1] - theta0[:, i])) - 0.25 * mtimes((Einv[i][:, :] + Einv[i + 1][:, :]), (
                        Mcsnp[:, i] + Mcsnp[:, i + 1])) * delta_s[i] + mtimes(damp, (
                        mtimes(Ka[i][:, :], (damp_MK[:, i + 1] - damp_MK[:, i])) + 0.5 * mtimes((
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
                BCroot = BC[element]['root']
                BCtip = BC[element]['tip']
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
                for k in range(0, len(indicesRoot_)):
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
        return reshape(Res, (18 * n, 1))

    def JointResJac(self, Residuals, beam_list, JointProp, X):
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
        n = beam_list['r0'].shape[1]
        Res = SX.sym('Fty', len(JointProp['Child']) * 12, 1)

        for k in range(0, len(JointProp['Child'])):
            r_J = X[18 * n + 0 + 12 * k: 18 * n + 3 + 12 * k]
            th_J = X[18 * n + 3 + 12 * k: 18 * n + 6 + 12 * k]
            F_J = X[18 * n + 6 + 12 * k: 18 * n + 9 + 12 * k]
            M_J = X[18 * n + 9 + 12 * k: 18 * n + 12 + 12 * k]

            parent_index = JointProp['Parent'][k]
            child_index = JointProp['Child'][k]

            # Node numbers of joint 1 and joint 2
            NodeNum_JP1 = JointProp['Parent_NodeNum'][k]
            NodeNum_JP2 = JointProp['Child_NodeNum'][k]

            # region Residual of kinematic constraints
            starting_node_parent = 18 * beam_list['node_lim'][parent_index, 0]  # ZERO entry. Where the part residual entry starts
            starting_node_child = 18 * beam_list['node_lim'][child_index, 0]  # ZERO entry. Where the part residual entry starts

            r10 = JointProp['Parent_r0'][:, k]  # stickModel_1.r0(:,NodeNum_JP1)
            r20 = JointProp['Child_r0'][:, k]  # stickModel_2.r0(:,NodeNum_JP2)
            th10 = JointProp['Parent_th0'][:, k]  # stickModel_1.th0(:,NodeNum_JP1)
            th20 = JointProp['Child_th0'][:, k]  # stickModel_2.th0(:,NodeNum_JP2)

            r1 = X[starting_node_parent + 18 * NodeNum_JP1 + 0:starting_node_parent + 18 * NodeNum_JP1 + 3]
            th1 = X[starting_node_parent + 18 * NodeNum_JP1 + 3:starting_node_parent + 18 * NodeNum_JP1 + 6]

            r2 = r20 + r_J
            th2 = th20 + th_J

            # Create the matrices

            T1 = SymbolicStickModel.RotationMatrix(beam_list['seq'][3 * parent_index:3 + 3 * parent_index], th1)
            T10 = SymbolicStickModel.RotationMatrix(beam_list['seq'][3 * parent_index:3 + 3 * parent_index], th10)
            T2 = SymbolicStickModel.RotationMatrix(beam_list['seq'][3 * child_index:3 + 3 * child_index], th2)
            T20 = SymbolicStickModel.RotationMatrix(beam_list['seq'][3 * child_index:3 + 3 * child_index], th20)

            Res[12 * k + 0: 12 * k + 3] = r2 - r1 - mtimes(transpose(T1), mtimes(T10, (r20 - r10)))

            temp1 = mtimes(transpose(T1), T10)
            temp2 = mtimes(transpose(T2), T20)

            Res[12 * k + 3] = dot(temp1[:, 1], temp2[:, 2]) - dot(temp1[:, 2], temp2[:, 1])
            Res[12 * k + 4] = dot(temp1[:, 2], temp2[:, 0]) - dot(temp1[:, 0], temp2[:, 2])
            Res[12 * k + 5] = dot(temp1[:, 0], temp2[:, 1]) - dot(temp1[:, 1], temp2[:, 0])

            # endregion

            # region Residual of F&M equations

            # Form the required variables
            Mi1 = X[starting_node_child + 18 * (NodeNum_JP2 + 1) + 9:starting_node_child + 18 * (NodeNum_JP2 + 1) + 12]
            Mi = X[starting_node_child + 18 * (NodeNum_JP2) + 9:starting_node_child + 18 * (NodeNum_JP2) + 12]
            Fi1 = X[starting_node_child + 18 * (NodeNum_JP2 + 1) + 6:starting_node_child + 18 * (NodeNum_JP2 + 1) + 9]
            Fi = X[starting_node_child + 18 * (NodeNum_JP2) + 6:starting_node_child + 18 * (NodeNum_JP2) + 9]

            # Assumptions (for now)
            deltaF = np.zeros((3, 1))
            deltaM = np.zeros((3, 1))

            # Force equation residual
            Res[12 * k + 6: 12 * k + 9] = Fi1 - Fi + deltaF - F_J

            # Moment equation residual
            temp3 = (r2 - r1)
            temp3_tilde = SX.zeros((3, 3))
            temp3_tilde[0, 0] = 0
            temp3_tilde[0, 1] = -temp3[2]
            temp3_tilde[0, 2] = temp3[1]
            temp3_tilde[1, 0] = temp3[2]
            temp3_tilde[1, 1] = 0
            temp3_tilde[1, 2] = -temp3[0]
            temp3_tilde[2, 0] = -temp3[1]
            temp3_tilde[2, 1] = temp3[0]
            temp3_tilde[2, 2] = 0

            Res[12 * k + 9: 12 * k + 12] = Mi1 - Mi + deltaM - M_J + mtimes(temp3_tilde, F_J)

            # endregion
        Residuals[18 * n: 18 * n + 12 * len(JointProp['Child'])] = Res

        return Residuals

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
        T = mtimes(R[seq[2] - 1], mtimes(R[seq[1] - 1], R[seq[0] - 1]))

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

        n = beam_list['r0'].shape[1]
        for k in range(0, len(JointProp['Child'])):
            r_J = X[18 * n + 0 + 12 * k: 18 * n + 3 + 12 * k]
            th_J = X[18 * n + 3 + 12 * k: 18 * n + 6 + 12 * k]
            child_joint = JointProp['Child'][k]
            starting_node = 18 * beam_list['node_lim'][child_joint, 0]  # ZERO entry. Where the part residual entry starts
            Residuals[starting_node + 18 * JointProp['Child_NodeNum'][k] + 0:starting_node + 18 * JointProp['Child_NodeNum'][k] + 3] = X[starting_node + 18 *
                                                                                                                                         JointProp['Child_NodeNum'][k] + 0:starting_node + 18 *
                                                                                                                                                                           JointProp['Child_NodeNum'][
                                                                                                                                                                               k] + 3] - JointProp[
                                                                                                                                                                                             'Child_r0'][
                                                                                                                                                                                         :, k] - r_J
            Residuals[starting_node + 18 * JointProp['Child_NodeNum'][k] + 3:starting_node + 18 * JointProp['Child_NodeNum'][k] + 6] = X[starting_node + 18 *
                                                                                                                                         JointProp['Child_NodeNum'][k] + 3:starting_node + 18 *
                                                                                                                                                                           JointProp['Child_NodeNum'][
                                                                                                                                                                               k] + 6] - JointProp[
                                                                                                                                                                                             'Child_th0'][
                                                                                                                                                                                         :, k] - th_J
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


class StickModelFeeder(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('beam_list')

    def setup(self):
        beam_number = 0
        n_variables = 0
        for a_beam in self.options['beam_list']:
            self.add_input('cs_' + str(beam_number), shape=a_beam.symbolic_expressions['cs'].shape[0])
            n_variables += a_beam.symbolic_expressions['cs'].shape[0]
            beam_number += 1
        self.add_output('cs_out', shape=n_variables)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cs_fused = []
        for beam_number in range(len(self.options['beam_list'])):
            if not isinstance(cs_fused, np.ndarray):
                cs_fused = inputs['cs_' + str(beam_number)]
            else:
                cs_fused = np.hstack((cs_fused, inputs['cs_' + str(beam_number)]))
        outputs['cs_out'] = cs_fused


class StickModelDemultiplexer(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('beam_list')
        self.options.declare('joint_reference')

    def setup(self):
        beam_number = 0
        n_variables = 0

        for a_beam in self.options['beam_list']:
            self.add_output('x_' + str(beam_number), shape=(a_beam.symbolic_expressions['x'].shape[0], a_beam.symbolic_expressions['x'].shape[1]))
            n_variables += a_beam.symbolic_expressions['x'].shape[0]
            beam_number += 1
        self.add_input('x_in', shape=(n_variables+12*len(self.options['joint_reference']), self.options['beam_list'][0].symbolic_expressions['x'].shape[1]))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x_fused = inputs['x_in']
        n_variables = 0
        for beam_number in range(len(self.options['beam_list'])):
            outputs['x_' + str(beam_number)] = x_fused[n_variables:n_variables+self.options['beam_list'][beam_number].symbolic_expressions['x'].shape[0], :]
            n_variables += self.options['beam_list'][beam_number].symbolic_expressions['x'].shape[0]


class BeamStickModel(SymbolicStickModel, om.ImplicitComponent):
    def initialize(self):
        # Dynamic Variables:
        self.options.declare('t_initial', types=float, default=0.0)
        self.options.declare('t_final', types=float, default=1.0)
        self.options.declare('time_step', types=float, default=1.0)    # default is only 1 timestep
        self.options.declare('order', types=int, default=2)
        self.options.declare('tolerance', types=float, default=1e-8)
        self.options.declare('load_function', default=[])
        # General Variables
        self.options.declare('load_factor', types=float)
        self.options.declare('beam_list')
        self.options.declare('joint_reference')
        self.options.declare('num_timesteps')
        self.options.declare('t_gamma', default=0.03)
        self.options.declare('t_epsilon', default=0.03)
        # Generated structures:
        self.beam_reference = {}
        self.JointProp = {}
        self.symbolics = {}
        self.symbolic_functions = {}
        self.symbolic_expressions = {}
        self.numeric_storage = {}
        self.seq = []

    def setup(self):
        self.t_gamma_c = self.options['t_gamma']
        self.t_epsilon_s = self.options['t_epsilon']
        self.t_gamma_n = self.options['t_gamma']
        self.t_kappa_c = self.options['t_epsilon']
        self.t_kappa_s = self.options['t_gamma']
        self.t_kappa_n = self.options['t_epsilon']

        # First, It's necessary to generate all the symbolics necessary for the jointed system to work:

        self.create_symbolic_function(self.options['beam_list'], self.options['joint_reference'])

        self.options['num_timesteps'] = math.floor((self.options['t_final'] - self.options['t_initial']) / self.options['time_step'])

        # Generate state and force numerical connections:

        self.add_input('xDot', shape=(self.symbolic_expressions['Residual'].shape[0], self.options['num_timesteps']+1))
        self.add_input('Xac', shape=18)
        self.add_input('forces_dist', shape=3 * self.beam_reference['forces_conc'].shape[1])
        self.add_input('moments_dist', shape=3 * self.beam_reference['forces_conc'].shape[1])
        self.add_input('forces_conc', shape=3 * self.beam_reference['forces_conc'].shape[1])
        self.add_input('moments_conc', shape=3 * self.beam_reference['forces_conc'].shape[1])

        self.add_input('cs', shape=self.beam_reference['cs'].shape[0])

        # Add the outputs from this stickmodel class (just a single beam)
        self.add_output('x', shape=(self.symbolic_expressions['Residual'].shape[0], self.options['num_timesteps']+1))

        # Initialize numerical integration variables:
        if len(self.options['joint_reference']) > 0:
            self.numeric_storage['x0'] = np.hstack((self.beam_reference['x0'], np.zeros((12 * len(self.options['joint_reference'])))))
        else:
            self.numeric_storage['x0'] = self.beam_reference['x0']

        if self.options['num_timesteps'] > 1:
            # Forces updated via dynamic function
            self.numeric_storage['Xac'] = np.zeros(18)
            self.numeric_storage['forces_dist'] = np.zeros(3 * self.beam_reference['forces_dist'].shape[1])
            self.numeric_storage['moments_dist'] = np.zeros(3 * self.beam_reference['moments_dist'].shape[1])
            self.numeric_storage['forces_conc'] = np.zeros(3 * self.beam_reference['forces_conc'].shape[1])
            self.numeric_storage['moments_conc'] = np.zeros(3 * self.beam_reference['moments_conc'].shape[1])

            self.numeric_storage['max_step'] = self.options['num_timesteps']        # Arrays in Python are zero-based
            self.numeric_storage['time'] = np.zeros(self.options['num_timesteps'] + 1)  # Initially all are zeros
            self.numeric_storage['current_step'] = 1
            self.numeric_storage['xDot'] = np.zeros((self.symbolic_expressions['Residual'].shape[0], self.options['num_timesteps'] + 1))
            self.numeric_storage['xDot_local'] = np.zeros(self.symbolic_expressions['Residual'].shape[0])
            self.numeric_storage['x_local'] = np.zeros(self.symbolic_expressions['Residual'].shape[0])
            self.numeric_storage['x_minus_1'] = np.zeros(self.symbolic_expressions['Residual'].shape[0])
            self.numeric_storage['x_minus_2'] = np.zeros(self.symbolic_expressions['Residual'].shape[0])
            self.numeric_storage['k0'] = 0
            self.numeric_storage['k1'] = 0
            self.numeric_storage['k2'] = 0

    def solve_nonlinear(self, inputs, outputs):
        Integrator(tolerance=self.options['tolerance'], model=self, bdf_order=self.options['order'], inputs=inputs, outputs=outputs)
    def create_symbolic_function(self, beams, joints):

        # Procedure for itemization of beam:

        # Relevant Expressions: cs, x, xDot, x_slice, xDot_slice, D, oneover, mu, i_matrix, delta_r_CG_tilde, Einv, E, EA, forces_dist, moments_dist, forces_conc, moments_conc
        variable_categories = ['cs', 'x', 'xDot', 'x_slice', 'xDot_slice', 'D', 'oneover', 'mu', 'i_matrix', 'delta_r_CG_tilde', 'Einv', 'E', 'EA', 'forces_dist', 'moments_dist',
                               'forces_conc', 'moments_conc', 'r0', 'th0', 'delta_s0', 'x0', 'K0a', 'BC', 'seq']
        variable_types = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 2]  # 0: SX, 1: list

        self.beam_reference['cs'] = {}
        self.beam_reference['x'] = {}
        self.beam_reference['xDot'] = {}
        self.beam_reference['x_slice'] = {}
        self.beam_reference['xDot_slice'] = {}
        self.beam_reference['D'] = []
        self.beam_reference['oneover'] = []
        self.beam_reference['mu'] = {}
        self.beam_reference['i_matrix'] = []
        self.beam_reference['delta_r_CG_tilde'] = []
        self.beam_reference['Einv'] = []
        self.beam_reference['E'] = []
        self.beam_reference['EA'] = {}
        self.beam_reference['forces_dist'] = {}
        self.beam_reference['moments_dist'] = {}
        self.beam_reference['forces_conc'] = {}
        self.beam_reference['moments_conc'] = {}
        self.beam_reference['r0'] = []
        self.beam_reference['th0'] = []
        self.beam_reference['x0'] = []
        self.beam_reference['delta_s0'] = []
        self.beam_reference['K0a'] = []
        self.beam_reference['BC'] = {}
        self.beam_reference['seq'] = []

        # Additional Elements:
        self.beam_reference['node_lim'] = np.zeros((len(beams), 2), dtype=int)
        self.beam_reference['inter_node_lim'] = np.zeros((len(beams), 2), dtype=int)
        self.beam_reference['n'] = 0
        self.beam_reference['n_inter'] = 0

        beam_number = 0
        for a_beam in beams:
            # Accumulated quantities:
            self.beam_reference['node_lim'][beam_number, 0] = self.beam_reference['n']
            self.beam_reference['inter_node_lim'][beam_number, 0] = self.beam_reference['n_inter']
            self.beam_reference['n'] += a_beam.symbolic_expressions['EA'].size()[0]
            self.beam_reference['n_inter'] += a_beam.symbolic_expressions['mu'].size()[0]
            self.beam_reference['node_lim'][beam_number, 1] = self.beam_reference['n']
            self.beam_reference['inter_node_lim'][beam_number, 1] = self.beam_reference['n_inter']
            # Inherited quantities:
            for a_variable_index in range(len(variable_categories)):
                if variable_types[a_variable_index] == 0:
                    if not isinstance(self.beam_reference[variable_categories[a_variable_index]], SX):
                        self.beam_reference[variable_categories[a_variable_index]] = a_beam.symbolic_expressions[variable_categories[a_variable_index]]
                    else:
                        if (a_beam.symbolic_expressions[variable_categories[a_variable_index]].size()[0] > a_beam.symbolic_expressions[variable_categories[a_variable_index]].size()[1]) or (
                                variable_categories[a_variable_index] == 'x' or variable_categories[a_variable_index] == 'xDot'):
                            # Vertically oriented vector
                            self.beam_reference[variable_categories[a_variable_index]] = vertcat(self.beam_reference[variable_categories[a_variable_index]],
                                                                                                 a_beam.symbolic_expressions[variable_categories[a_variable_index]])
                        else:
                            # Horizontally oriented vector
                            self.beam_reference[variable_categories[a_variable_index]] = horzcat(self.beam_reference[variable_categories[a_variable_index]],
                                                                                                 a_beam.symbolic_expressions[variable_categories[a_variable_index]])
                else:
                    if variable_types[a_variable_index] == 1:
                        # List
                        self.beam_reference[variable_categories[a_variable_index]] += a_beam.symbolic_expressions[variable_categories[a_variable_index]]
                    else:
                        # Boundary Condition
                        if variable_categories[a_variable_index] == 'BC':
                            self.beam_reference[variable_categories[a_variable_index]][beam_number] = {}
                            self.beam_reference[variable_categories[a_variable_index]][beam_number]['root'] = a_beam.BC['root']
                            self.beam_reference[variable_categories[a_variable_index]][beam_number]['tip'] = a_beam.BC['tip']
                        else:
                            # Numpy Array
                            if not isinstance(self.beam_reference[variable_categories[a_variable_index]], np.ndarray):
                                self.beam_reference[variable_categories[a_variable_index]] = a_beam.options[variable_categories[a_variable_index]]
                            else:
                                if len(a_beam.options[variable_categories[a_variable_index]].shape) < 3:
                                    self.beam_reference[variable_categories[a_variable_index]] = np.hstack(
                                        (self.beam_reference[variable_categories[a_variable_index]], a_beam.options[variable_categories[a_variable_index]]))
                                else:
                                    self.beam_reference[variable_categories[a_variable_index]] = np.vstack(
                                        (self.beam_reference[variable_categories[a_variable_index]], a_beam.options[variable_categories[a_variable_index]]))
            beam_number += 1

        # Procedure for Itemization of Joints:
        if len(joints) > 0:
            self.JointProp['Parent_NodeNum'] = []
            self.JointProp['Parent_r0'] = np.zeros((3, len(joints)))
            self.JointProp['Parent_th0'] = np.zeros((3, len(joints)))
            self.JointProp['Parent'] = []
            self.JointProp['Child'] = []
            self.JointProp['Child_NodeNum'] = []
            self.JointProp['Child_r0'] = np.zeros((3, len(joints)))
            self.JointProp['Child_th0'] = np.zeros((3, len(joints)))

            joint_number = 0
            for a_joint in joints:
                # The originating joint element is a much simpler object with just parent name, child name, and eta
                beam_number = 0
                for a_beam in beams:
                    if a_joint.parent_beam == a_beam.name:
                        self.JointProp['Parent'].append(beam_number)

                        main_direction = 1
                        if np.array_equal(a_beam.options["seq"], np.array([3, 1, 2])):  # Fuselage beam
                            main_direction = 0
                        span_eta = a_joint.parent_eta * (a_beam.options['r0'][main_direction, -1] - a_beam.options['r0'][main_direction, 0])
                        search_span = a_beam.options['r0'][main_direction, :]
                        result = np.where(abs(search_span - span_eta) < 1e-5)
                        self.JointProp['Parent_NodeNum'].append(result[0][0])
                        self.JointProp['Parent_r0'][:, joint_number] = a_beam.options['r0'][:, result[0][0]]
                        self.JointProp['Parent_th0'][:, joint_number] = a_beam.options['th0'][:, result[0][0]]

                    if a_joint.child_beam == a_beam.name:
                        self.JointProp['Child'].append(beam_number)

                        main_direction = 1
                        if np.array_equal(a_beam.options["seq"], np.array([3, 1, 2])):  # Fuselage beam
                            main_direction = 0
                        span_eta = a_joint.child_eta * (a_beam.options['r0'][main_direction, -1] - a_beam.options['r0'][main_direction, 0])
                        search_span = a_beam.options['r0'][main_direction, :]
                        result = np.where(abs(search_span - span_eta) < 1e-5)
                        self.JointProp['Child_NodeNum'].append(result[0][0])
                        self.JointProp['Child_r0'][:, joint_number] = a_beam.options['r0'][:, result[0][0]]
                        self.JointProp['Child_th0'][:, joint_number] = a_beam.options['th0'][:, result[0][0]]

                    beam_number += 1

                # Add new columns for the symbolic state vectors
                self.beam_reference['x'] = vertcat(self.beam_reference['x'], SX.sym(a_joint.joint_label + 'state', 12, beams[0].options['num_timesteps'] + 1))
                self.beam_reference['xDot'] = vertcat(self.beam_reference['xDot'], SX.sym(a_joint.joint_label + 'stateD', 12, beams[0].options['num_timesteps'] + 1))
                self.beam_reference['x_slice'] = vertcat(self.beam_reference['x_slice'], SX.sym(a_joint.joint_label + 'state_s', 12, 1))
                self.beam_reference['xDot_slice'] = vertcat(self.beam_reference['xDot_slice'], SX.sym(a_joint.joint_label + 'state_s_D', 12, 1))

                joint_number += 1
        g = np.zeros(3)
        g[2] = 9.81

        R_prec = np.ones(12)
        # R_prec[0] = 1e-3
        # R_prec[1] = 1e-5
        # R_prec[2] = 1e5
        # R_prec[3] = 1
        # R_prec[4] = 1e0
        # R_prec[5] = 1e0
        # R_prec[6] = 1e-4
        # R_prec[7] = 3 * 1e3
        # R_prec[8] = 1e2
        # R_prec[9] = 1e0
        # R_prec[10] = 1e0
        # R_prec[11] = 1e0
        self.g = -self.options['load_factor'] * g

        nodes = self.beam_reference['n']

        self.symbolics['Xac'] = SX.sym('Xac', 18)

        # Starting resjac multipart

        # Generate force/moment dictionaries for resjac:
        Forces = {}
        Forces['inter_node_lim'] = self.beam_reference['inter_node_lim']
        Forces['node_lim'] = self.beam_reference['node_lim']
        Forces['delta_Fapplied'] = SX.zeros((self.beam_reference['forces_conc'].shape[0], self.beam_reference['forces_conc'].shape[1])) + self.beam_reference['forces_conc']
        Forces['f_aero'] = self.beam_reference['forces_dist']
        Moments = {}
        Moments['inter_node_lim'] = self.beam_reference['inter_node_lim']
        Moments['node_lim'] = self.beam_reference['node_lim']
        Moments['delta_Mapplied'] = SX.zeros((self.beam_reference['moments_conc'].shape[0], self.beam_reference['moments_conc'].shape[1])) + self.beam_reference['moments_conc']
        Moments['m_aero'] = self.beam_reference['moments_dist']

        self.symbolic_expressions['Residual'] = self.ResJac_Multipart(stickModel=self.beam_reference, Xd=self.beam_reference['xDot_slice'], X_AC=self.symbolics['Xac'],
                                                                      Forces=Forces, Moments=Moments,
                                                                      BC=self.beam_reference['BC'], g=self.g, JointProp=self.JointProp, X=self.beam_reference['x_slice'],
                                                                      Rprec=R_prec)

        # Gather Functions:

        self.symbolic_functions['Residuals'] = Function("Residuals", [self.beam_reference['x_slice'], self.beam_reference['xDot_slice'],
                                                                      self.symbolics['Xac'],
                                                                      reshape(self.beam_reference['forces_dist'].T, (3 * self.beam_reference['n_inter'], 1)),
                                                                      reshape(self.beam_reference['moments_dist'].T, (3 * self.beam_reference['n_inter'], 1)),
                                                                      reshape(self.beam_reference['forces_conc'].T, (3 * self.beam_reference['n_inter'], 1)),
                                                                      reshape(self.beam_reference['moments_conc'].T, (3 * self.beam_reference['n_inter'], 1)),
                                                                      self.beam_reference['cs']],
                                                        [self.symbolic_expressions['Residual']],
                                                        ['x', 'xDot', 'Xac', 'fDist', 'mDist', 'fConc', 'mConc', 'csDVs'],
                                                        ['r'])

        self.symbolic_expressions['Jacobian'] = jacobian(self.symbolic_expressions['Residual'],
                                                         vertcat(self.beam_reference['x_slice'], self.beam_reference['xDot_slice']))

        self.symbolic_functions['Jacobian'] = Function('Jac', [self.beam_reference['x_slice'], self.beam_reference['xDot_slice'],
                                                               self.symbolics['Xac'],
                                                               reshape(self.beam_reference['forces_dist'].T, (3 * self.beam_reference['n_inter'], 1)),
                                                               reshape(self.beam_reference['moments_dist'].T, (3 * self.beam_reference['n_inter'], 1)),
                                                               reshape(self.beam_reference['forces_conc'].T, (3 * self.beam_reference['n_inter'], 1)),
                                                               reshape(self.beam_reference['moments_conc'].T, (3 * self.beam_reference['n_inter'], 1)),
                                                               self.beam_reference['cs']],
                                                       [self.symbolic_expressions['Jacobian']])

    def ResJac_Multipart(self, stickModel, Xd, X_AC, Forces, Moments, BC, g, JointProp, X, Rprec):
        # Initialize Macro-Structure
        Residuals = SX.sym('Res', Xd.shape[0])

        # Generate holders for the forces that will get modified due to the Joint Residuals:
        c_Forces = Forces

        c_Moments = Moments

        if len(JointProp) > 0:
            #  Now lets run through each joint (all at once) and add the forces that correspond to the parent joint part
            c_Forces, c_Moments = BeamStickModel.AddJointConcLoads(X, JointProp, c_Forces, c_Moments)

        # Running individual elements
        n = 0
        for i in range(0, int(stickModel['seq'].size / 3)):
            n_element = stickModel['node_lim'][i, 1] - stickModel['node_lim'][i, 0]
            Residuals[18 * n: 18 * (n + n_element)] = self.ResJac(stickModel, reshape(X[18 * n:18 * (n + n_element)], (18, n_element)),
                                                                  reshape(Xd[18 * n:18 * (n + n_element)], (18, n_element)), X_AC, c_Forces, c_Moments, BC, g, i, Rprec)
            n = n + n_element
        if len(JointProp) > 0:
            # Modify Joint 2 Equations
            Residuals = self.ModifyJointPoint2Residuals(stickModel, Residuals, X, JointProp)
            # Run joint Residuals:
            Residuals = self.JointResJac(Residuals, stickModel, JointProp, X)

        return Residuals

    def update_time_varying_quantities(self, x, xDot, i):
        self.numeric_storage['Xac'], self.numeric_storage['forces_dist'], self.numeric_storage['moments_dist'], self.numeric_storage['forces_conc'],  self.numeric_storage['moments_conc'] = self.options['load_function'](x, xDot, self.numeric_storage['Xac'], self.numeric_storage['forces_dist'], self.numeric_storage['moments_dist'], self.numeric_storage['forces_conc'],  self.numeric_storage['moments_conc'], self.options['time_step'], i)