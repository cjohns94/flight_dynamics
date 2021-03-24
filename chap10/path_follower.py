import numpy as np
from math import sin, cos
import sys

sys.path.append("..")
from message_types.msg_autopilot import MsgAutopilot
from tools.wrap import wrap


class PathFollower:
    def __init__(self):
        self.chi_inf = np.radians(
            60
        )  # approach angle for large distance from straight-line path
        self.k_path = 0.05  # proportional gain for straight-line path following
        self.k_orbit = 10.0  # proportional gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = MsgAutopilot()  # message sent to autopilot

    def update(self, path, state):
        if path.type == "line":
            self._follow_straight_line(path, state)
        elif path.type == "orbit":
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):
        self.autopilot_commands.airspeed_command = path.airspeed

        chi_q = np.arccos(path.line_direction.item(0))

        # smallest turn logic
        while chi_q - state.chi < -np.pi:
            chi_q += 2 * np.pi
        while chi_q - state.chi > np.pi:
            chi_q += -2 * np.pi

        R = np.array(
            [
                [np.cos(chi_q), np.sin(chi_q), 0],
                [-np.sin(chi_q), np.cos(chi_q), 0],
                [0, 0, 1],
            ]
        )

        p = np.array([[state.north, state.east, -state.altitude]]).T
        r = path.line_origin

        error_p = R @ (p - r)

        # course command
        self.autopilot_commands.course_command = (
            chi_q - self.chi_inf * 2 * np.arctan(self.k_path * error_p.item(1)) / np.pi
        )

        ki = np.array([[0, 0, 1]]).T
        n = np.cross(ki.flatten(), path.line_direction.flatten()) / np.linalg.norm(
            np.cross(ki.flatten(), path.line_direction.flatten())
        ).reshape(-1, 1)

        s = error_p - np.dot(error_p, n) * n

        # altitude command
        self.autopilot_commands.altitude_command = -r.item(2) - np.sqrt(
            s.item(0) ** 2 + s.item(1) ** 2
        ) * (
            path.line_direction.item(2)
            / np.sqrt(
                path.line_direction.item(0) ** 2 + path.line_direction.item(1) ** 2
            )
        )
        # feedforward roll angle for straight line is zero
        self.autopilot_commands.phi_feedforward = 0

    def _follow_orbit(self, path, state):
        if path.orbit_direction == "CW":
            direction = 1.0
        else:
            direction = -1.0
        # airspeed command
        self.autopilot_commands.airspeed_command = path.airspeed
        # distance from orbit center
        c = path.orbit_center
        p = np.array([[state.north, state.east, -state.altitude]]).T

        d = np.sqrt((p.item(0) - c.item(0)) ** 2 + (p.item(1) - c.item(1)) ** 2)
        # compute wrapped version of angular position on orbit
        varphi = np.arctan2(p.item(1) - c.item(1), p.item(0) - c.item(0))

        # smallest turn logic
        while varphi - state.chi < -np.pi:
            varphi += 2 * np.pi

        while varphi - state.chi > np.pi:
            varphi += -2 * np.pi

        # compute normalized orbit error
        rho = path.orbit_radius
        orbit_error = d - rho
        # course command
        chi_0 = varphi + direction * np.pi / 2

        self.autopilot_commands.course_command = chi_0 + direction * np.arctan(
            self.k_orbit * (orbit_error / rho)
        )

        # altitude command
        self.autopilot_commands.altitude_command = -c.item(2)
        # roll feedforward command
        if orbit_error < 10:
            self.autopilot_commands.phi_feedforward = direction * np.arctan2(
                state.Va ** 2, self.gravity * rho
            )
        else:
            self.autopilot_commands.phi_feedforward = 0
