"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys

sys.path.append("..")
import numpy as np
from scipy.optimize import minimize
from tools.rotations import Euler2Quaternion, Quaternion2Euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta


def compute_model(mav, trim_state, trim_input):
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)

    print("\nEigenvalues of A_lon")
    print(np.linalg.eig(A_lon)[0])

    print("\nEigenvalues of A_lat")
    print(np.linalg.eig(A_lat)[0])

    breakpoint()
    input("Press Enter to continue")
    (
        Va_trim,
        alpha_trim,
        theta_trim,
        a_phi1,
        a_phi2,
        a_theta1,
        a_theta2,
        a_theta3,
        a_V1,
        a_V2,
        a_V3,
    ) = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open("model_coef.py", "w")
    file.write("import numpy as np\n")
    file.write(
        "x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n"
        % (
            trim_state.item(0),
            trim_state.item(1),
            trim_state.item(2),
            trim_state.item(3),
            trim_state.item(4),
            trim_state.item(5),
            trim_state.item(6),
            trim_state.item(7),
            trim_state.item(8),
            trim_state.item(9),
            trim_state.item(10),
            trim_state.item(11),
            trim_state.item(12),
        )
    )
    file.write(
        "u_trim = np.array([[%f, %f, %f, %f]]).T\n"
        % (
            trim_input.elevator,
            trim_input.aileron,
            trim_input.rudder,
            trim_input.throttle,
        )
    )
    file.write("Va_trim = %f\n" % Va_trim)
    file.write("alpha_trim = %f\n" % alpha_trim)
    file.write("theta_trim = %f\n" % theta_trim)
    file.write("a_phi1 = %f\n" % a_phi1)
    file.write("a_phi2 = %f\n" % a_phi2)
    file.write("a_theta1 = %f\n" % a_theta1)
    file.write("a_theta2 = %f\n" % a_theta2)
    file.write("a_theta3 = %f\n" % a_theta3)
    file.write("a_V1 = %f\n" % a_V1)
    file.write("a_V2 = %f\n" % a_V2)
    file.write("a_V3 = %f\n" % a_V3)
    file.write(
        "A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    "
        "[%f, %f, %f, %f, %f],\n    "
        "[%f, %f, %f, %f, %f],\n    "
        "[%f, %f, %f, %f, %f],\n    "
        "[%f, %f, %f, %f, %f]])\n"
        % (
            A_lon[0][0],
            A_lon[0][1],
            A_lon[0][2],
            A_lon[0][3],
            A_lon[0][4],
            A_lon[1][0],
            A_lon[1][1],
            A_lon[1][2],
            A_lon[1][3],
            A_lon[1][4],
            A_lon[2][0],
            A_lon[2][1],
            A_lon[2][2],
            A_lon[2][3],
            A_lon[2][4],
            A_lon[3][0],
            A_lon[3][1],
            A_lon[3][2],
            A_lon[3][3],
            A_lon[3][4],
            A_lon[4][0],
            A_lon[4][1],
            A_lon[4][2],
            A_lon[4][3],
            A_lon[4][4],
        )
    )
    file.write(
        "B_lon = np.array([\n    [%f, %f],\n    "
        "[%f, %f],\n    "
        "[%f, %f],\n    "
        "[%f, %f],\n    "
        "[%f, %f]])\n"
        % (
            B_lon[0][0],
            B_lon[0][1],
            B_lon[1][0],
            B_lon[1][1],
            B_lon[2][0],
            B_lon[2][1],
            B_lon[3][0],
            B_lon[3][1],
            B_lon[4][0],
            B_lon[4][1],
        )
    )
    file.write(
        "A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    "
        "[%f, %f, %f, %f, %f],\n    "
        "[%f, %f, %f, %f, %f],\n    "
        "[%f, %f, %f, %f, %f],\n    "
        "[%f, %f, %f, %f, %f]])\n"
        % (
            A_lat[0][0],
            A_lat[0][1],
            A_lat[0][2],
            A_lat[0][3],
            A_lat[0][4],
            A_lat[1][0],
            A_lat[1][1],
            A_lat[1][2],
            A_lat[1][3],
            A_lat[1][4],
            A_lat[2][0],
            A_lat[2][1],
            A_lat[2][2],
            A_lat[2][3],
            A_lat[2][4],
            A_lat[3][0],
            A_lat[3][1],
            A_lat[3][2],
            A_lat[3][3],
            A_lat[3][4],
            A_lat[4][0],
            A_lat[4][1],
            A_lat[4][2],
            A_lat[4][3],
            A_lat[4][4],
        )
    )
    file.write(
        "B_lat = np.array([\n    [%f, %f],\n    "
        "[%f, %f],\n    "
        "[%f, %f],\n    "
        "[%f, %f],\n    "
        "[%f, %f]])\n"
        % (
            B_lat[0][0],
            B_lat[0][1],
            B_lat[1][0],
            B_lat[1][1],
            B_lat[2][0],
            B_lat[2][1],
            B_lat[3][0],
            B_lat[3][1],
            B_lat[4][0],
            B_lat[4][1],
        )
    )
    file.write("Ts = %f\n" % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = Quaternion2Euler(trim_state[6:10])

    # define transfer function constants (slides 12, 21, 25), not sure about S_wing?, Va_trim?
    a_phi1 = -(
        0.5
        * MAV.rho
        * Va_trim ** 2
        * MAV.S_wing
        * MAV.b
        * MAV.C_p_p
        * MAV.b
        / (2 * Va_trim)
    )
    a_phi2 = 0.5 * MAV.rho * Va_trim ** 2 * MAV.S_wing * MAV.b * MAV.C_p_delta_a
    a_theta1 = (
        -MAV.rho
        * Va_trim ** 2
        * MAV.c
        * MAV.S_wing
        * MAV.C_m_q
        * MAV.c
        / (2 * MAV.Jy * 2 * Va_trim)
    )
    a_theta2 = (
        -MAV.rho * Va_trim ** 2 * MAV.c * MAV.S_wing * MAV.C_m_alpha / (2 * MAV.Jy)
    )
    a_theta3 = (
        MAV.rho * Va_trim ** 2 * MAV.c * MAV.S_wing * MAV.C_m_delta_e / (2 * MAV.Jy)
    )

    # Compute transfer function coefficients using new propulsion model
    delta_t_trim = trim_input.throttle
    delta_e_trim = trim_input.elevator
    a_V1 = ((MAV.rho * Va_trim * MAV.S_prop) / (MAV.mass)) * (
        MAV.C_D_0 + MAV.C_D_alpha * alpha_trim + MAV.C_D_delta_e * delta_e_trim
    ) - (1 / MAV.mass) * dT_dVa(mav, Va_trim, delta_t_trim)
    a_V2 = (1 / MAV.mass) * dT_ddelta_t(mav, Va_trim, delta_t_trim)
    a_V3 = MAV.gravity * np.cos(theta_trim - alpha_trim)

    return (
        Va_trim,
        alpha_trim,
        theta_trim,
        a_phi1,
        a_phi2,
        a_theta1,
        a_theta2,
        a_theta3,
        a_V1,
        a_V2,
        a_V3,
    )


def compute_ss_model(mav, trim_state, trim_input):
    x_euler = euler_state(trim_state)
    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)
    # breakpoint()
    # extract longitudinal states (u, w, q, theta, pd) and change pd to h
    lon_state_mask = np.array([3, 5, 10, 7, 2])
    lon_input_mask = np.array([0, 3])
    A_lon = A[lon_state_mask, :][:, lon_state_mask]
    B_lon = B[lon_state_mask, :][:, lon_input_mask]
    # extract lateral states (v, p, r, phi, psi)
    lat_state_mask = np.array([4, 9, 11, 6, 8])
    lat_input_mask = np.array([1, 2])
    A_lat = A[lat_state_mask, :][:, lat_state_mask]
    B_lat = B[lat_state_mask, :][:, lat_input_mask]
    return A_lon, B_lon, A_lat, B_lat


def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    quat = x_quat[6:10]
    phi, theta, psi = Quaternion2Euler(quat)

    x_euler = np.zeros((12, 1))

    x_euler[:6] = x_quat[:6]
    x_euler[6] = phi
    x_euler[7] = theta
    x_euler[8] = psi
    x_euler[9:] = x_quat[10:]

    return x_euler


def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    phi = x_euler[6, 0]
    theta = x_euler[7, 0]
    psi = x_euler[8, 0]

    quat = Euler2Quaternion(phi, theta, psi)

    x_quat = np.zeros((13, 1))
    x_quat[:6] = x_euler[:6]
    x_quat[6:10] = quat
    x_quat[10:] = x_euler[9:]

    return x_quat


def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state

    # get x_quaternion
    x_quat = quaternion_state(x_euler)
    quat = x_quat[6:10]
    euler = x_euler[6:9]

    # update velocity, states, and forces in mav
    mav.external_set_state(x_quat)
    mav._update_velocity_data()
    forces_moments = mav._forces_moments(delta)

    # get quaternion derivatives (f_quat = d(quaterion)/dt) with mav._dynamics
    f_quat = mav._derivatives(x_quat, forces_moments)
    dquat_dt = f_quat[6:10]
    # We want d(euler angles)/dt = d(euler angles/d(quaternion) * d(quaternion)/dt
    # get d(euler angles/d(quaternion)
    # initialize d(euler angles/d(quaternion) as 3x4 zeros
    deuler_dquat = np.zeros((3, 4))

    # for each column
    for i in range(4):
        # perturb quaterion state by eps
        eps = 0.001
        dquat = np.zeros((4, 1))
        dquat[i] = eps
        quat_perturb = quat + dquat

        # normalize perturbed quaternion
        quat_perturb = quat_perturb / np.linalg.norm(quat_perturb)

        # covnert perturbed quaterion to perturbed euler angles
        phi_pert, theta_pert, psi_pert = Quaternion2Euler(quat_perturb)
        euler_perturb = np.array([[phi_pert], [theta_pert], [psi_pert]])

        # d(euler angles)/d(quaternion) column = (euler angles - perturbed euler angles)/eps
        temp = (euler_perturb - euler) / eps
        deuler_dquat[:, i] = temp.flatten()

    deuler_dt = deuler_dquat @ dquat_dt
    f_euler_ = np.zeros((12, 1))
    f_euler_[:6] = f_quat[:6]
    f_euler_[6:9] = deuler_dt
    f_euler_[9:] = f_quat[10:]
    return f_euler_


def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    # call f_euler for each column of A.
    perturb = 0.001
    A = np.zeros((12, 12))
    f = f_euler(mav, x_euler, delta)
    for i in range(12):
        dx = np.zeros((12, 1))
        dx[i] = perturb
        fdx = f_euler(mav, x_euler + dx, delta)
        A[:, i] = ((fdx - f) / perturb).flatten()

    return A


def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input
    perturb = 0.001
    B = np.zeros((12, 4))
    f = f_euler(mav, x_euler, delta)
    delta_array = delta.to_array()
    perturbed_delta = MsgDelta()

    for i in range(4):
        d_delta = np.zeros((4, 1))
        d_delta[i] = perturb
        perturbed_delta.from_array(delta_array + d_delta)
        fdu = f_euler(mav, x_euler, perturbed_delta)
        B[:, i] = ((fdu - f) / perturb).flatten()

    return B


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.001
    T_eps, Q_eps = mav._motor_thrust_torque(Va + eps, delta_t)
    T, Q = mav._motor_thrust_torque(Va, delta_t)
    return (T_eps - T) / eps


def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.001
    T_eps, Q_eps = mav._motor_thrust_torque(Va, delta_t + eps)
    T, Q = mav._motor_thrust_torque(Va, delta_t)
    return (T_eps - T) / eps
