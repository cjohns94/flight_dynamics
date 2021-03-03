"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import sys

sys.path.append("..")
import numpy as np

# load message types
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
from message_types.msg_delta import MsgDelta

import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Rotation


class MavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array(
            [
                [MAV.north0],  # (0)
                [MAV.east0],  # (1)
                [MAV.down0],  # (2)
                [MAV.u0],  # (3)
                [MAV.v0],  # (4)
                [MAV.w0],  # (5)
                [MAV.e0],  # (6)
                [MAV.e1],  # (7)
                [MAV.e2],  # (8)
                [MAV.e3],  # (9)
                [MAV.p0],  # (10)
                [MAV.q0],  # (11)
                [MAV.r0],
            ]
        )  # (12)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.0], [0.0], [0.0]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.0], [0.0], [0.0]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # initialize true_state message
        self.true_state = MsgState()
        # initialize the sensors message
        self._sensors = MsgSensors()
        # random walk parameters for GPS
        self._gps_nu_n = 0.0
        self._gps_nu_e = 0.0
        self._gps_nu_h = 0.0
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.0  # large value ensures gps updates at initial time.
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())

    ###################################
    # public functions
    def update(self, delta, wind):
        """
        Integrate the differential equations defining dynamics, update sensors
        delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
        wind is the wind vector in inertial coordinates
        Ts is the time step between function calls.
        """
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step / 2.0 * k1, forces_moments)
        k3 = self._derivatives(self._state + time_step / 2.0 * k2, forces_moments)
        k4 = self._derivatives(self._state + time_step * k3, forces_moments)
        self._state += time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0 ** 2 + e1 ** 2 + e2 ** 2 + e3 ** 2)
        self._state[6][0] = self._state.item(6) / normE
        self._state[7][0] = self._state.item(7) / normE
        self._state[8][0] = self._state.item(8) / normE
        self._state[9][0] = self._state.item(9) / normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        # update the message class for the true state
        self._update_true_state()

    def sensors(self):
        "Return value of sensors on MAV: gyros, accels, absolute_pressure, dynamic_pressure, GPS"
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)

        # euler angles
        phi, theta, psi = Quaternion2Euler(np.array([e0, e1, e2, e3]))
        R = Quaternion2Rotation(np.array([e0, e1, e2, e3]))

        # position kinematics using quaternion to avoid singularity
        u = self._state.item(3)
        v = self._state.item(4)
        w = self._state.item(5)
        pdot = R @ np.array([[u], [v], [w]])

        # simulate rate gyros(units are rad / sec)
        self._sensors.gyro_x = self._state[10, 0] + np.random.normal(
            SENSOR.gyro_x_bias, SENSOR.gyro_sigma
        )
        self._sensors.gyro_y = self._state[11, 0] + np.random.normal(
            SENSOR.gyro_y_bias, SENSOR.gyro_sigma
        )
        self._sensors.gyro_z = self._state[12, 0] + np.random.normal(
            SENSOR.gyro_z_bias, SENSOR.gyro_sigma
        )

        # simulate accelerometers(units of g)
        self._sensors.accel_x = (
            self._forces.item(0) / MAV.mass
            + MAV.gravity * np.sin(theta)
            + np.random.normal(0, SENSOR.accel_sigma)
        )
        self._sensors.accel_y = (
            self._forces.item(1) / MAV.mass
            - MAV.gravity * np.cos(theta) * np.sin(phi)
            + np.random.normal(0, SENSOR.accel_sigma)
        )
        self._sensors.accel_z = (
            self._forces.item(2) / MAV.mass
            - MAV.gravity * np.cos(theta) * np.cos(phi)
            + np.random.normal(0, SENSOR.accel_sigma)
        )
        # simulate magnetometers
        # magnetic field in provo has magnetic declination of 12.5 degrees
        # and magnetic inclination of 66 degrees
        iota = np.radians(66)
        delta = np.radians(12.5)
        R_mag = Euler2Rotation(0, -iota, delta).T
        # magnetic field in inertial frame: unit vector
        mag_inertial = R_mag @ np.array([[1, 0, 0]]).T
        R = R.T  # body to inertial
        # magnetic field in body frame: unit vector
        mag_body = R @ mag_inertial
        self._sensors.mag_x = mag_body.item(0) + np.random.normal(
            SENSOR.mag_beta, SENSOR.mag_sigma
        )
        self._sensors.mag_y = mag_body.item(1) + np.random.normal(
            SENSOR.mag_beta, SENSOR.mag_sigma
        )
        self._sensors.mag_z = mag_body.item(2) + np.random.normal(
            SENSOR.mag_beta, SENSOR.mag_sigma
        )
        # simulate pressure sensors
        self._sensors.abs_pressure = MAV.rho * MAV.gravity * -self._state.item(
            2
        ) + np.random.normal(0, SENSOR.abs_pres_sigma)
        self._sensors.diff_pressure = MAV.rho * self._Va / 2 + np.random.normal(
            0, SENSOR.diff_pres_sigma
        )
        # simulate GPS sensor
        if self._t_gps >= SENSOR.ts_gps:
            self._gps_nu_n = np.exp(
                -SENSOR.gps_k * SENSOR.ts_gps
            ) * self._gps_nu_n + np.random.normal(0, SENSOR.gps_n_sigma)
            self._gps_nu_e = np.exp(
                -SENSOR.gps_k * SENSOR.ts_gps
            ) * self._gps_nu_e + np.random.normal(0, SENSOR.gps_e_sigma)
            self._gps_nu_h = np.exp(
                -SENSOR.gps_k * SENSOR.ts_gps
            ) * self._gps_nu_h + np.random.normal(0, SENSOR.gps_h_sigma)
            self._sensors.gps_n = self._state.item(0) + self._gps_nu_n
            self._sensors.gps_e = self._state.item(1) + self._gps_nu_e
            self._sensors.gps_h = self._state.item(2) + self._gps_nu_h
            self._sensors.gps_Vg = np.sqrt(
                (self._Va * np.cos(psi) + self._wind.item(0)) ** 2 + (self._Va * np.sin(psi) + self._wind.item(1)) ** 2
            ) + np.random.normal(0, SENSOR.gps_Vg_sigma)
            self._sensors.gps_course = np.arctan2(
                self._Va * np.sin(psi) + self._wind.item(1), self._Va * np.cos(psi) + self._wind.item(0)
            ) + np.random.normal(0, SENSOR.gps_course_sigma)
            self._t_gps = 0.0
        else:
            self._t_gps += self._ts_simulation
        return self._sensors

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        # north = state.item(0)
        # east = state.item(1)
        # altitude = state.item(2)
        # print(state)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # euler angles
        phi, theta, psi = Quaternion2Euler(np.array([e0, e1, e2, e3]))
        R = Quaternion2Rotation(np.array([e0, e1, e2, e3]))

        # position kinematics using quaternion to avoid singularity
        pdot = R @ np.array([[u], [v], [w]])

        pn_dot = pdot[0, 0]
        pe_dot = pdot[1, 0]
        pd_dot = pdot[2, 0]

        # position dynamics
        u_dot = (r * v - q * w) + fx / MAV.mass
        v_dot = (p * w - r * u) + fy / MAV.mass
        w_dot = (q * u - p * v) + fz / MAV.mass

        # rotational kinematics
        e0_dot = 0.5 * (-p * e1 - q * e2 - r * e3)
        e1_dot = 0.5 * (p * e0 + r * e2 - q * e3)
        e2_dot = 0.5 * (q * e0 - r * e1 + p * e3)
        e3_dot = 0.5 * (r * e0 + q * e1 - p * e2)

        # gammas
        gamma1 = MAV.gamma1
        gamma2 = MAV.gamma2
        gamma3 = MAV.gamma3
        gamma4 = MAV.gamma4
        gamma5 = MAV.gamma5
        gamma6 = MAV.gamma6
        gamma7 = MAV.gamma7
        gamma8 = MAV.gamma8

        # rotatonal dynamics
        p_dot = gamma1 * p * q - gamma2 * q * r + gamma3 * l + gamma4 * n
        q_dot = gamma5 * p * r - gamma6 * (p ** 2 - r ** 2) + m / MAV.Jy
        r_dot = gamma7 * p * q - gamma1 * q * r + gamma4 * l + gamma8 * n

        # collect the derivative of the states
        x_dot = np.array(
            [
                [
                    pn_dot,
                    pe_dot,
                    pd_dot,
                    u_dot,
                    v_dot,
                    w_dot,
                    e0_dot,
                    e1_dot,
                    e2_dot,
                    e3_dot,
                    p_dot,
                    q_dot,
                    r_dot,
                ]
            ]
        ).T
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6, 1))):
        steady_state = wind[0:3]
        gust = wind[3:6]
        # convert wind vector from world to body frame and add gust
        wind_body_frame = Quaternion2Rotation(self._state[6:10])
        self._wind = wind_body_frame @ steady_state + gust  # wind in the world frame
        # velocity vector relative to the airmass
        ur = self._state[3, 0] - self._wind[0, 0]
        vr = self._state[4, 0] - self._wind[1, 0]
        wr = self._state[5, 0] - self._wind[2, 0]
        # compute airspeed
        self._Va = np.sqrt(ur ** 2 + vr ** 2 + wr ** 2)
        # compute angle of attack
        if ur == 0:
            print("ur is zero!!")
            self._alpha = 0
        else:
            self._alpha = np.arctan2(wr, ur)
        # compute sideslip angle
        if self._Va == 0:
            print("Va is zero!!")
            self._beta = 0
        else:
            self._beta = np.arcsin(vr / self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        delta_a = delta.aileron
        delta_e = delta.elevator
        delta_r = delta.rudder
        delta_t = delta.throttle

        phi, theta, psi = Quaternion2Euler(self._state[6:10, 0])
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        # compute gravitational forces, using quaternion
        ex = self._state.item(7)
        ey = self._state.item(8)
        ez = self._state.item(9)
        e0 = self._state.item(6)

        # gravity force expressed in body frame, try quaternion2rotation
        f_g = (
            MAV.mass
            * MAV.gravity
            * np.array(
                [
                    [2 * (ex * ez - ey * e0)],
                    [2 * (ey * ez + ex * e0)],
                    [ez ** 2 + e0 ** 2 - ex ** 2 - ey ** 2],
                ]
            )
        )

        # compute Lift (eq. 4.9) and Drag (eq 4.11) coefficients, sigma (eq 4.10)
        # using nonlinear blended model for CL(alpha) and CD(alpha)
        sigma = (
            1
            + np.exp(-MAV.M * (self._alpha - MAV.alpha0))
            + np.exp(MAV.M * (self._alpha + MAV.alpha0))
        ) / (
            (1 + np.exp(-MAV.M * (self._alpha - MAV.alpha0)))
            * (1 + np.exp(MAV.M * (self._alpha + MAV.alpha0)))
        )
        CL = (1 - sigma) * (MAV.C_L_0 + MAV.C_L_alpha * self._alpha) + sigma * (
            2 * np.sign(self._alpha) * np.sin(self._alpha) ** 2 * np.cos(self._alpha)
        )
        CD = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha * self._alpha) ** 2 / (
            np.pi * MAV.e * MAV.AR
        )
        # compute Lift(eq 4.6) and Drag (eq 4.7) Forces
        temp = 0.5 * MAV.rho * self._Va ** 2 * MAV.S_wing
        F_lift = temp * (
            CL + MAV.C_L_q * MAV.c * q / (2 * self._Va) + MAV.C_L_delta_e * delta_e
        )
        F_drag = temp * (
            CD + MAV.C_D_q * MAV.c * q / (2 * self._Va) + MAV.C_D_delta_e * delta_e
        )

        # compute propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta_t)
        # print("Thrust: ", thrust_prop)

        # compute longitudinal forces in body frame (4.24)
        sa = np.sin(self._alpha)
        ca = np.cos(self._alpha)

        # #LONG WAY
        # C_X = -CD * ca + CL * sa
        # C_X_q = -MAV.C_D_q * ca + MAV.C_L_q * sa
        # C_X_delta_e = -MAV.C_D_delta_e * ca + MAV.C_L_delta_e * sa
        # C_Z = -CD * sa - CL * ca
        # C_Z_q = -MAV.C_D_q * sa - MAV.C_L_q * ca
        # C_Z_delta_e = -MAV.C_D_delta_e * sa - MAV.C_L_delta_e * ca

        # fx = (
        #     f_g[0, 0]
        #     + thrust_prop
        #     + temp * (C_X + C_X_q * MAV.c * q / (2 * self._Va))
        #     + temp * (C_X_delta_e * delta_e)
        # )
        # fz = (
        #     f_g[2, 0]
        #     + temp * (C_Z + C_Z_q * MAV.c * q / (2 * self._Va))
        #     + temp * (C_Z_delta_e * delta_e)
        # )

        # SHORT WAY
        fx = f_g[0, 0] - F_drag * ca + F_lift * sa + thrust_prop
        fz = f_g[2, 0] - F_drag * sa - F_lift * ca

        # compute lateral forces in body frame (4.24)
        fy = (
            f_g[1, 0]
            + temp
            * (
                MAV.C_Y_0
                + MAV.C_Y_beta * self._beta
                + MAV.C_Y_p * MAV.b * p / (2 * self._Va)
                + MAV.C_Y_r * MAV.b * r / (2 * self._Va)
            )
            + temp * (MAV.C_Y_delta_a * delta_a + MAV.C_Y_delta_r * delta_r)
        )

        # compute logitudinal torque in body frame, My = m (4.26)
        My = temp * (
            MAV.c
            * (
                MAV.C_m_0
                + MAV.C_m_alpha * self._alpha
                + MAV.C_m_q * MAV.c * q / (2 * self._Va)
            )
        ) + temp * (MAV.c * (MAV.C_m_delta_e * delta_e))
        # compute lateral torques in body frame Mx = l, Mz = n
        Mx = (
            temp
            * (
                MAV.b
                * (
                    MAV.C_ell_0
                    + MAV.C_ell_beta * self._beta
                    + MAV.C_ell_p * MAV.b * p / (2 * self._Va)
                    + MAV.C_ell_r * MAV.b * r / (2 * self._Va)
                )
            )
            + temp
            * (MAV.b * (MAV.C_ell_delta_a * delta_a + MAV.C_ell_delta_r * delta_r))
            + torque_prop
        )
        Mz = temp * (
            MAV.b
            * (
                MAV.C_n_0
                + MAV.C_n_beta * self._beta
                + MAV.C_n_p * MAV.b * p / (2 * self._Va)
                + MAV.C_n_r * MAV.b * r / (2 * self._Va)
            )
        ) + temp * (MAV.b * (MAV.C_n_delta_a * delta_a + MAV.C_n_delta_r * delta_r))

        self._forces[0] = fx
        self._forces[1] = fy
        self._forces[2] = fz
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        # print("forces\n", forces_moments)
        # return np.array([[0, 0, 0, 0.0, 0.0, 0.0]]).T
        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller  (See addendum by McLain)
        # map delta_t throttle command(0 to 1) into motor input voltage (eq 4.22)
        V_in = MAV.V_max * delta_t

        # Angular speed of propeller (eq 4.21)
        a = MAV.rho * MAV.D_prop ** 5 * MAV.C_Q0 / (2 * np.pi) ** 2
        b = (
            MAV.rho * MAV.D_prop ** 4 * MAV.C_Q1 * Va / (2 * np.pi)
            + MAV.KQ ** 2 / MAV.R_motor
        )
        c = (
            MAV.rho * MAV.D_prop ** 3 * MAV.C_Q2 * Va ** 2
            - MAV.KQ * V_in / MAV.R_motor
            + MAV.KQ * MAV.i0
        )
        Omega_p = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        # calculate advance ratio J
        J = (2 * np.pi * Va) / (Omega_p * MAV.D_prop)

        # calculate nondimensional coefficients
        C_T = MAV.C_T2 * J ** 2 + MAV.C_T1 * J + MAV.C_T0
        C_Q = MAV.C_Q2 * J ** 2 + MAV.C_Q1 * J + MAV.C_Q0

        # thrust (eq 4.17) and torque (eq 4.18) due to propeller
        thrust_prop = C_T * MAV.rho * MAV.D_prop ** 4 * Omega_p ** 2 / (4 * np.pi ** 2)
        torque_prop = C_Q * MAV.rho * MAV.D_prop ** 5 * Omega_p ** 2 / (4 * np.pi ** 2)
        return thrust_prop, torque_prop

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        pdot = Quaternion2Rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
