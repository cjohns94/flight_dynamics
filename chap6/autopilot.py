"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import numpy as np

sys.path.append("..")
import parameters.control_parameters as AP
from tools.transfer_function import transferFunction
from tools.wrap import wrap
from chap6.pi_control import PIControl
from chap6.pd_control_with_rate import PDControlWithRate
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from chap5.model_coef import u_trim, Va_trim


class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = PDControlWithRate(
            kp=AP.roll_kp, kd=AP.roll_kd, limit=np.radians(45)
        )
        self.course_from_roll = PIControl(
            kp=AP.course_kp, ki=AP.course_ki, Ts=ts_control, limit=np.radians(30)
        )
        self.yaw_damper = transferFunction(
            num=np.array([[AP.yaw_damper_kr, 0]]),
            den=np.array([[1, AP.yaw_damper_p_wo]]),
            Ts=ts_control,
        )

        # instantiate longitudinal controllers
        self.pitch_from_elevator = PDControlWithRate(
            kp=AP.pitch_kp, kd=AP.pitch_kd, limit=np.radians(45)
        )
        self.altitude_from_pitch = PIControl(
            kp=AP.altitude_kp, ki=AP.altitude_ki, Ts=ts_control, limit=np.radians(30)
        )
        self.airspeed_from_throttle = PIControl(
            kp=AP.airspeed_throttle_kp,
            ki=AP.airspeed_throttle_ki,
            Ts=ts_control,
            limit=1.0,
        )
        self.commanded_state = MsgState()

        self.trim_inputs = MsgDelta(
            elevator=u_trim[0, 0],
            aileron=u_trim[1, 0],
            rudder=u_trim[2, 0],
            throttle=u_trim[3, 0],
        )

        self.Va_trim = Va_trim

    def update(self, cmd, state):
        """
        cmd: MsgAutopilot(
        """

        # extract commands
        Va_cmd = cmd.airspeed_command
        Chi_cmd = cmd.course_command

        # lateral autopilot
        chi_c = 0
        phi_c = 0
        delta_a = 0
        delta_r = 0

        # longitudinal autopilot
        # saturate the altitude command
        h_cmd = cmd.altitude_command
        h_cmd = self.saturate(h_cmd, h_cmd - AP.altitude_zone, h_cmd + AP.altitude_zone)
        theta_c = 0
        delta_e = 0

        # airspeed using throttle loop
        Va_cmd_bar = Va_cmd - self.Va_trim
        Va_bar = state.Va - self.Va_trim
        delta_t_bar = self.airspeed_from_throttle.update(Va_cmd_bar, Va_bar)
        delta_t = delta_t_bar + self.trim_inputs.throttle

        # altitude hold loop
        theta_cmd = self.altitude_from_pitch.update(h_cmd, state.altitude)

        # pitch attitude hold loop
        # theta_dot calculated directly from dynamics
        theta_dot = np.cos(state.phi) * state.q - np.sin(state.phi) * state.r
        delta_e = self.pitch_from_elevator.update(theta_cmd, state.theta, theta_dot)

        # construct output and commanded states
        # delta = MsgDelta(
        #     elevator=delta_e, aileron=delta_a, rudder=delta_r, throttle=delta_t
        # )

        delta = MsgDelta(
            elevator=delta_e,
            aileron=self.trim_inputs.aileron,
            rudder=self.trim_inputs.rudder,
            throttle=delta_t,
        )

        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
