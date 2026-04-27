#!/usr/bin/env python

"""
Kinematic function skeleton code originally from UC Berkeley EE 106A

Originally written by: Aaron Bestick, 9/10/14
Adapted for SJTU Spring 2026 by: Yanwen Zou, 19/03/26

You should fill in the body of the five empty methods below so that they implement the kinematic
functions described in the assignment.

When you think you have the methods implemented correctly, you can test your
code by running "python kin_func_skeleton.py at the command line.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# -----------------------------2D Examples---------------------------------------
# --(you don't need to modify anything here but you should take a look at them)--


def rotation_2d(theta):
    """
    Computes a 2D rotation matrix given the angle of rotation.

    Args:
    theta: the angle of rotation

    Returns:
    rot - (2,2) ndarray: the resulting rotation matrix
    """

    rot = np.zeros((2, 2))
    rot[0, 0] = np.cos(theta)
    rot[1, 1] = np.cos(theta)
    rot[0, 1] = -np.sin(theta)
    rot[1, 0] = np.sin(theta)

    return rot


def hat_2d(xi):
    """
    Converts a 2D twist to its corresponding 3x3 matrix representation

    Args:
    xi - (3,) ndarray: the 2D twist

    Returns:
    xi_hat - (3,3) ndarray: the resulting 3x3 matrix
    """
    if not xi.shape == (3,):
        raise TypeError("omega must be a 3-vector")

    xi_hat = np.zeros((3, 3))
    xi_hat[0, 1] = -xi[2]
    xi_hat[1, 0] = xi[2]
    # xi_hat[0:2,2] = xi[0:2]
    xi_hat[0, 2] = xi[1]
    xi_hat[1, 2] = -xi[0]
    xi_hat[2, 0] = -xi[1]
    xi_hat[2, 1] = xi[0]

    return xi_hat


def homog_2d(xi, theta):
    """
    Computes a 3x3 homogeneous transformation matrix given a 2D twist and a
    joint displacement

    Args:
    xi - (3,) ndarray: the 2D twist
    theta: the joint displacement

    Returns:
    g - (3,3) ndarray: the resulting homogeneous transformation matrix
    """
    if not xi.shape == (3,):
        raise TypeError("xi must be a 3-vector")

    g = np.zeros((3, 3))
    wtheta = xi[2] * theta
    R = rotation_2d(wtheta)
    p = np.dot(
        np.dot(
            [
                [1 - np.cos(wtheta), np.sin(wtheta)],
                [-np.sin(wtheta), 1 - np.cos(wtheta)],
            ],
            [[0, -1], [1, 0]],
        ),
        [[xi[0] / xi[2]], [xi[1] / xi[2]]],
    )

    g[0:2, 0:2] = R
    g[0:2, 2:3] = p[0:2]
    g[2, 2] = 1

    return g


# -----------------------------3D Functions--------------------------------------
# -------------(These are the functions you need to complete)--------------------


def skew_3d(omega):
    """
    Converts a rotation vector in 3D to its corresponding skew-symmetric matrix.

    Args:
    omega - (3,) ndarray: the rotation vector

    Returns:
    omega_hat - (3,3) ndarray: the corresponding skew symmetric matrix
    """
    omega = np.asarray(omega, dtype=np.float64)
    if omega.shape != (3,):
        raise TypeError("omega must be a 3-vector")

    omega_hat = np.array(
        [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ],
        dtype=np.float64,
    )

    return omega_hat


def rotation_3d(omega, theta):
    """
    Computes a 3D rotation matrix given a rotation axis and angle of rotation.

    Args:
    omega - (3,) ndarray: the axis of rotation
    theta: the angle of rotation

    Returns:
    rot - (3,3) ndarray: the resulting rotation matrix

    R = I + sin(theta) * omega_hat + (1 - cos(theta)) * np.dot(omega_hat, omega_hat)
    """
    omega = np.asarray(omega, dtype=np.float64)
    if omega.shape != (3,):
        raise TypeError("omega must be a 3-vector")

    norm = np.linalg.norm(omega)
    if np.isclose(norm, 0.0):
        return np.eye(3)

    omega_unit = omega / norm
    theta_eff = norm * theta
    omega_hat = skew_3d(omega_unit)
    R = (
        np.eye(3)
        + np.sin(theta_eff) * omega_hat
        + (1.0 - np.cos(theta_eff)) * (omega_hat @ omega_hat)
    )

    return R


def hat_3d(xi):
    """
    Converts a 3D twist to its corresponding 4x4 matrix representation

    Args:
    xi - (6,) ndarray: the 3D twist

    Returns:
    xi_hat - (4,4) ndarray: the corresponding 4x4 matrix
    """
    xi = np.asarray(xi, dtype=np.float64)
    if xi.shape != (6,):
        raise TypeError("xi must be a 6-vector")

    v = xi[0:3]
    omega = xi[3:6]

    xi_hat = np.zeros((4, 4), dtype=np.float64)
    xi_hat[0:3, 0:3] = skew_3d(omega)
    xi_hat[0:3, 3] = v

    return xi_hat


def homog_3d(xi, theta):
    """
    Computes a 4x4 homogeneous transformation matrix given a 3D twist and a
    joint displacement.

    Args:
    xi - (6,) ndarray: the 3D twist
    theta: the joint displacement
    Returns:
    g - (4,4) ndarary: the resulting homogeneous transformation matrix
    """
    xi = np.asarray(xi, dtype=np.float64)
    if xi.shape != (6,):
        raise TypeError("xi must be a 6-vector")

    v = xi[0:3]
    omega = xi[3:6]
    w_norm = np.linalg.norm(omega)

    g = np.eye(4, dtype=np.float64)

    if np.isclose(w_norm, 0.0):
        g[0:3, 3] = v * theta
        return g

    omega_unit = omega / w_norm
    theta_eff = w_norm * theta

    R = rotation_3d(omega_unit, theta_eff)
    omega_hat = skew_3d(omega_unit)

    V = (
        np.eye(3) * theta_eff
        + (1.0 - np.cos(theta_eff)) * omega_hat
        + (theta_eff - np.sin(theta_eff)) * (omega_hat @ omega_hat)
    )
    p = V @ (v / w_norm)

    g[0:3, 0:3] = R
    g[0:3, 3] = p

    return g


def prod_exp(xi, theta):
    """
    Computes the product of exponentials for a kinematic chain, given
    the twists and displacements for each joint.

    Args:
    xi - (6, N) ndarray: the twists for each joint
    theta - (N,) ndarray: the displacement of each joint

    Returns:
    g - (4,4) ndarray: the resulting homogeneous transformation matrix
    """
    xi = np.asarray(xi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    if xi.ndim != 2 or xi.shape[0] != 6:
        raise TypeError("xi must be a (6, N) array")
    if theta.ndim != 1 or theta.shape[0] != xi.shape[1]:
        raise TypeError("theta must be a length-N vector")

    g = np.eye(4, dtype=np.float64)
    for i in range(theta.shape[0]):
        g = g @ homog_3d(xi[:, i], theta[i])

    return g


# ---------------------------------TESTING CODE---------------------------------
# -------------------------DO NOT MODIFY ANYTHING BELOW HERE--------------------


def array_func_test(func_name, args, ret_desired):
    ret_value = func_name(*args)
    if not isinstance(ret_value, np.ndarray):
        print(
            "[FAIL] "
            + func_name.__name__
            + "() returned something other than a NumPy ndarray"
        )
    elif ret_value.shape != ret_desired.shape:
        print(
            "[FAIL] "
            + func_name.__name__
            + "() returned an ndarray with incorrect dimensions"
        )
    elif not np.allclose(ret_value, ret_desired, rtol=1e-3):
        print("[FAIL] " + func_name.__name__ + "() returned an incorrect value")
    else:
        print("[PASS] " + func_name.__name__ + "() returned the correct value!")


if __name__ == "__main__":
    # TEST skew_3d()
    arg1 = np.array([1, 2, 3])
    ret1 = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
    array_func_test(skew_3d, [arg1], ret1)

    # TEST rotation_3d()
    arg1 = np.array([2, 1, 3])
    arg2 = 0.587
    ret1 = np.array(
        [[0.8373, -0.4777, 0.2664], [0.5472, 0.8132, -0.1978], [-0.0083, 0.3318, 0.9433]]
    )
    array_func_test(rotation_3d, [arg1, arg2], ret1)

    # TEST hat_3d()
    arg1 = np.array([1, 2, 3, 5, 6, 7])
    ret1 = np.array([[0, -7, 6, 1], [7, 0, -5, 2], [-6, 5, 0, 3], [0, 0, 0, 0]])
    array_func_test(hat_3d, [arg1], ret1)

    # TEST homog_3d()
    arg1 = np.array([2, 1, 3, 5, 6, 7])
    arg2 = 0.658
    ret1 = np.array(
        [
            [0.1931, -0.3324, 0.9233, 1.6358],
            [0.9783, 0.1014, -0.1807, 0.1978],
            [-0.0749, 0.9377, 0.3392, 2.9609],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    array_func_test(homog_3d, [arg1, arg2], ret1)

    # TEST prod_exp()
    arg1 = np.array(
        [
            [2, 1, 3, 5, 6, 7],
            [1, 2, 4, 5, 7, 9],
            [5, 4, 7, 1, 4, 3],
            [1, 1, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ]
    )
    arg2 = np.array([0.658, 0.234, 1.345, 1.234, 0.122, 0.988])
    ret1 = np.array(
        [
            [0.4396, -0.5632, 0.6997, 1.0928],
            [0.4042, 0.8253, 0.3941, 5.6116],
            [-0.8019, 0.0388, 0.5962, 2.3576],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    array_func_test(prod_exp, [arg1, arg2], ret1)
