from casadi import *
from scipy.optimize import minimize

def CalcNodalT(th, seq, n):
    assert th.shape[1] == n
    T = SX.sym('T', 3, 3, n)
    Ta = SX.sym('Ta', 3, 3, n - 1)
    R = SX.sym('R', 3, 3, 3)
    for i in range(0, n):
        a1 = th[0, i]
        a2 = th[1, i]
        a3 = th[2, i]
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
        T[i][:, :] = mtimes(R[seq[2] - 1][:, :], mtimes(R[seq[1] - 1][:, :], R[seq[0] - 1][:, :]))
        if i >= 1:
            Ta[i - 1][:, :] = (T[i - 1][:, :] + T[i][:, :]) / 2
    return T, Ta


def CalcNodalT_singleNode_numeric(th, seq):
    R = np.empty([3, 3, 3])
    a1 = th[0]
    a2 = th[1]
    a3 = th[2]
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
    T = np.dot(R[seq[2] - 1][:, :], np.dot(R[seq[1] - 1][:, :], R[seq[0] - 1][:, :]))
    return T

def centeroidnp(arr):
    """
    Compute the centroid of 2-D points given as a numpy array
    """
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


def angle_calc(angles, r_vec, sequence):
    twist = 0
    sweep = np.deg2rad(angles[0])
    dihedral = np.deg2rad(angles[1])
    r_len = np.linalg.norm(r_vec)
    r_untransformed = np.array([0, r_len, 0])
    th0_vec = np.array([dihedral, twist, -sweep])
    T = CalcNodalT_singleNode_numeric(th=th0_vec, seq=sequence)
    new_dr = np.dot(np.transpose(T), r_untransformed)
    err = np.linalg.norm(np.abs(new_dr - r_vec))
    return err


def calculate_th0(r0, seq):
    """
    r0 is given such that the beam axis nodes are along the y axis
    :param r0: nX3 numpy array
    :param seq: 3x1 numpy array
    :return:

    Parameters
    ----------
    r0
    seq
    """
    n = r0.shape[1]
    th0 = np.empty([3, n])

    for i in range(n):
        # start with a guess for the angles
        sweep = 25
        dihedral = 3
        if i < n-1:
            r_reference = np.array([
                [r0[0, i], r0[0, i + 1]],
                [r0[1, i], r0[1, i + 1]],
                [r0[2, i], r0[2, i + 1]]
            ]
            )
            r_vec = r_reference[:, 1] - r_reference[:, 0]
            if np.linalg.norm(np.array(r_vec)) > 0.0:
                x = minimize(fun=angle_calc,
                             x0=np.array([sweep, dihedral]),
                             args=(r_vec, seq),
                             method='BFGS',
                             options={'disp': False})
                th0[0, i] = np.deg2rad(x.x[1])
                if np.abs(th0[0, i]) < 1e-5:
                    th0[0, i] = 0.
                th0[1, i] = 0.
                th0[2, i] = -np.deg2rad(x.x[0])
                if np.abs(th0[2, i]) < 1e-5:
                    th0[2, i] = 0.
            else:
                if i > 0:
                    # Inherit angle from previous section
                    th0[:, i] = th0[:, i - 1]
                else:
                    th0[:, i] = np.array([0.0, 0.0, 0.0])
        else:
            # Inherit angle from previous section
            th0[:, i] = th0[:, i - 1]
    return th0

# function to get unique values
def unique(list1):
    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list