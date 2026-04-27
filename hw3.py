import numpy as np
from kin_func_skeleton import prod_exp

#########################################
#               EXAMPLE                 #
#########################################


def scara_fk(theta):
    """
    An example implementation of a forward kinematics map.
    Feel free to use this as a template for your own implementations
    in this file.

    This function implements the forward kinematics map of the
    SCARA manipulator, following Example 3.1 from MLS (page 87).

    We take L0 = L1 = L2 = 1

    Arguments:
        theta: numpy.ndarray of size (4,), the values of the joint angles.
               theta[i] is the value of the ith joint, at which the
               FK map should be computed.
    Returns:
        - g (numpy.ndarray of shape (4,4)): the 4x4 configuration of the
          end effector when the joints have been placed at the angles
          specified in theta.
        - xi_array (numpy.ndarray of shape (6, N)): an array with the twists
          stacked in its columns.
    """

    # Specify all twists.
    xi_1 = [0, 0, 0, 0, 0, 1]
    xi_2 = [1, 0, 0, 0, 0, 1]
    xi_3 = [2, 0, 0, 0, 0, 1]
    xi_4 = [0, 0, 1, 0, 0, 0]

    # Specify end effector configuration at theta = 0.
    gst0 = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 2], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float64
    )

    # Stack twists into an array that prod_exp can accept.
    xi_array = np.array([xi_1, xi_2, xi_3, xi_4], dtype=np.float64).T

    # Use product of exponentials formula to compute forward kinematics.
    g = np.matmul(prod_exp(xi_array, theta), gst0)

    # Return the required quantities.
    return g, xi_array


#########################################
#              HW PROBLEMS              #
#########################################


def fk(theta):
    """
    Stanford 6 自由度机械臂（5R+1P）在基座系下的前向运动学。

    几何与作业约定（与 Figure 1 理想 Stanford 臂一致）：
    - l0 = l1 = 1；基座系 S 原点取在关节 1（竖直转轴）上，z 轴竖直向上。
    - 关节 2 位于 (0, 0, l0)，转轴沿 +x（俯仰）；前臂在 xy 平面内沿 +y 延伸至腕心 qw = (0, l1, l0)。
    - 作业文字「qw 与 q1 相距 l1」若将 q1 理解为肩关节/第二轴处参考点 q_s = (0,0,l0)，
      则 ||qw - q_s|| = l1，与图示一致；关节 1 轴过原点，故 ξ1 取 ω1=z 时 v1=0。
    - 关节 3 为移动关节，初始位姿下滑移方向沿前臂即 +y；关节 4–6 为交于 qw 的球腕（Z–X–Y 轴）。

    书面题 (c) 前向映射：T_ST(θ) = e^{[ξ1]θ1} … e^{[ξ6]θ6} · gst0，其中 gst0 = T_ST(0)。

    Arguments:
        theta: numpy.ndarray of size (6,), the values of the joint angles.
               theta[i] is the value of the ith joint, at which the
               FK map should be computed.
    Returns:
        - g (numpy.ndarray of shape (4,4)): the 4x4 configuration of the
          end effector when the joints have been placed at the angles
          specified in theta.
        - xi_array (numpy.ndarray of shape (6, N)): an array with the twists
          stacked in its columns.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (6,):
        raise TypeError("theta must have shape (6,)")

    l0 = 1.0
    l1 = 1.0

    # Joint axis points in the space frame at the home configuration.
    q_w = np.array([0.0, l1, l0], dtype=np.float64)

    # Revolute twists: xi = [v; w], v = -w x q.
    w1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    q1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    xi_1 = np.hstack((-np.cross(w1, q1), w1))

    w2 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    q2 = np.array([0.0, 0.0, l0], dtype=np.float64)
    xi_2 = np.hstack((-np.cross(w2, q2), w2))

    # Prismatic twist: xi = [u; 0], where u is the sliding direction.
    u3 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    xi_3 = np.hstack((u3, np.zeros(3, dtype=np.float64)))

    w4 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    xi_4 = np.hstack((-np.cross(w4, q_w), w4))

    w5 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    xi_5 = np.hstack((-np.cross(w5, q_w), w5))

    w6 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    xi_6 = np.hstack((-np.cross(w6, q_w), w6))

    xi_array = np.array([xi_1, xi_2, xi_3, xi_4, xi_5, xi_6], dtype=np.float64).T

    # Home pose T_ST(0): tool frame at q_w with identity orientation.
    gst0 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, l1],
            [0.0, 0.0, 1.0, l0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    g = np.matmul(prod_exp(xi_array, theta), gst0)

    # Return the required quantities.
    return g, xi_array
