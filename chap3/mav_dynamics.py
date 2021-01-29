"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
part of mavsimPy
    - Beard & McLain, PUP, 2012
    - Update history:  
        12/17/2018 - RWB
        1/14/2019 - RWB
        2/24/2020 - RWB
"""
import sys

sys.path.append("..")
import numpy as np

# load message types
from message_types.msg_state import MsgState

import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Euler, Quaternion2Rotation


class mavDynamics:
    def __init__(self, Ts):
        self.ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        self._state = np.array(
            [
                [MAV.pn0],  # (0)
                [MAV.pe0],  # (1)
                [MAV.pd0],  # (2)
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
        self.true_state = MsgState()

    ###################################
    # public functions
    def update(self, forces_moments):
        """
        Integrate the differential equations defining dynamics.
        Inputs are the forces and moments on the aircraft.
        Ts is the time step between function calls.
        """

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self.ts_simulation
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

        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        # pn = state.item(0)
        # pe = state.item(1)
        # pd = state.item(2)
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

        pn_dot = pdot[0][0]
        pe_dot = pdot[1][0]
        pd_dot = pdot[2][0]

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

    def _update_true_state(self):
        # update the true state message:
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.true_state.pn = self._state.item(0)
        self.true_state.pe = self._state.item(1)
        self.true_state.h = -self._state.item(2)
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)

        # print()
