import numpy as np
import sys
sys.path.append('..')
# from chap11.dubins_parameters import DubinsParameters
from message_types.msg_path import MsgPath


class PathManager:
    def __init__(self):
        # message sent to path follower
        self.path = MsgPath()
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        self.num_waypoints = 0
        self.halfspace_n = np.inf * np.ones((3,1))
        self.halfspace_r = np.inf * np.ones((3,1))
        # state of the manager state machine
        self.manager_state = 1
        self.manager_requests_waypoints = True
        # self.dubins_path = DubinsParameters()

    def update(self, waypoints, radius, state):
        if waypoints.num_waypoints == 0:
            self.manager_requests_waypoints = True
        if self.manager_requests_waypoints is True \
                and waypoints.flag_waypoints_changed is True:
            self.manager_requests_waypoints = False
        if waypoints.type == 'straight_line':
            self.line_manager(waypoints, state)
        elif waypoints.type == 'fillet':
            self.fillet_manager(waypoints, radius, state)
        elif waypoints.type == 'dubins':
            pass
            # self.dubins_manager(waypoints, radius, state)
        else:
            print('Error in Path Manager: Undefined waypoint type.')
        return self.path

    def line_manager(self, waypoints, state):
        # update self.path with the appropriate things
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer
        if waypoints.flag_waypoints_changed is True: #49:03
            # reset request flag and changed flag
            waypoints.flag_manager_requests_waypoints = False
            waypoints.flag_waypoints_changed = False
            self.num_waypoints = waypoints.num_waypoints
            # initialize pointers 
            self.initialize_pointers()
            # form a line
            self.construct_line(waypoints)

    def fillet_manager(self, waypoints, radius, state):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T

        if waypoints.flag_waypoints_changed is True:
            waypoints.flag_waypoints_changed = False
            self.num_waypoints = waypoints.num_waypoints
            self.initialize_pointers()
            self.construct_fillet_line(waypoints, radius)
            self.manager_state = 1

        # state machine for line path

        if self.manager_state == 1:
            # straight line path from previous to current
            if self.inHalfSpace(mav_pos):
                self.construct_fillet_circle(waypoints, radius)
                self.manager_state = 3
        elif self.manager_state == 2:
            # follow start orbit until out of half plane
            if not self.inHalfSpace(mav_pos):
                self.manager_state = 3
        elif self.manager_state == 3:
            # follow orbit from previous-> current to current->next
            if self.inHalfSpace(mav_pos):
                self.increment_pointers()
                self.construct_fillet_line(waypoints, radius)
                self.manager_state = 1

                if self.ptr_current == 0:
                    self.manager_requests_waypoints=True

    #def dubins_manager(self, waypoints, radius, state):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer

        # state machine for dubins path

    def initialize_pointers(self):
        if self.num_waypoints >= 3:
            self.ptr_previous = 0
            self.ptr_current = 1
            self.ptr_next = 2
        else:
            print('Error Path Manager: need at least three waypoints')

    def increment_pointers(self):
        self.ptr_previous = self.ptr_current
        self.ptr_current = self.ptr_next
        self.ptr_next = self.ptr_next + 1

        if self.ptr_next > self.num_waypoints -1:
            self.ptr_next = 9999
        if self.ptr_current > self.num_waypoints - 1:
            self.ptr_current = 9999

    def inHalfSpace(self, pos):
        if (pos-self.halfspace_r).T @ self.halfspace_n >=0: #implement code here
            return True
        else:
            return False

    def construct_line(self, waypoints):
        previous = waypoints.ned[:, self.ptr_previous:self.ptr_previous+1]
        if self.ptr_current == 9999:
            # fly straight forever 50:58
            current = previous + 100*self.path.line_direction
        else:
            # get current waypoint out of waypoints.ned
            current = waypoints.ned[:, self.ptr_current:self.ptr_current+1]
        if self.ptr_next == 9999:
            # fly straight for even longer 
            next = previous + 200*self.path.line_direction
        else:
            # get next waypoint out of waypoints.ned
            next = waypoints.ned[:, self.ptr_next:self.ptr_next+1]

        #update path variables
        #set path type
        self.path.type = 'line'
        #set path airspeed, origin, and line_direction
        self.path.airspeed = waypoints.airspeed.item(self.ptr_current)
        self.path.line_origin = previous
        q_previous = (current-previous) / np.linalg.norm(current - previous)
        self.path.line_direction = q_previous

        #slide 6
        q_next = (next-current)/ np.linalg.norm(next - current)
        self.halfspace_n = (q_previous + q_next)/2
        self.halfspace_n = self.halfspace_n/ np.linalg.norm(self.halfspace_n)
        self.halfspace_r = current
        self.path.plot_updated = False

    def construct_fillet_line(self, waypoints, radius):
        previous = waypoints.ned[:, self.ptr_previous:self.ptr_previous+1]
        if self.ptr_current == 9999:
            current = previous + 100*self.path.line_direction
        else:
            current = waypoints.ned[:, self.ptr_current:self.ptr_current+1]
        if self.ptr_next == 9999:
            next = previous + 200*self.path.line_direction
        else:
            next = waypoints.ned[:, self.ptr_next:self.ptr_next+1]

        #update path variables
        self.path.type = 'line'
        self.path.airspeed = waypoints.airspeed.item(self.ptr_current)
        self.path.line_origin = previous
        q_previous = (current - previous)/np.linalg.norm(current - previous)
        self.path.line_direction = q_previous
        q_next = (next - current)/np.linalg.norm(next-current)
        beta = np.arccos(-q_previous.T @ q_next)
        self.halfspace_n = q_previous
        self.halfspace_r = current - radius / np.tan(beta/2) * q_previous
        self.path.plot_updated = False

    def construct_fillet_circle(self, waypoints, radius):
        previous = waypoints.ned[:, self.ptr_previous:self.ptr_previous+1]
        if self.ptr_current == 9999:
            current = previous + 100*self.path.line_direction
        else:
            current = waypoints.ned[:, self.ptr_current:self.ptr_current+1]
        if self.ptr_next == 9999:
            next = previous + 200*self.path.line_direction
        else:
            next = waypoints.ned[:,self.ptr_next:self.ptr_next+1]
        #update path variables
        self.path.type = 'orbit'
        self.path.airspeed = waypoints.airspeed.item(self.ptr_current)
        q_previous = (current - previous)/np.linalg.norm(current-previous)
        q_next = (next-current) / np.linalg.norm(next-current)
        varrho = np.arccos(-q_previous.T@q_next)
        q_tmp = (q_previous - q_next)/np.linalg.norm(q_previous - q_next)
        self.path.orbit_center = current - radius/(np.sin(varrho/2)+.001) * q_tmp #add small number to keep from dividing by 0
        self.path.orbit_radius = radius

        if np.sign(q_previous.item(0) * q_next.item(1) - q_previous.item(1) * q_next.item(0)) > 0:
             self.path.orbit_direction = 'CW'
        else:
            self.path.orbit_direction = 'CCW'

        self.halfspace_n = q_next
        self.halfspace_r = current + radius/(np.tan(varrho/2)+.001) * q_next #add small number to keep from division by zero
        self.path.plot_updated = False

    #def construct_dubins_circle_start(self, waypoints, dubins_path):
        #update path variables

    #def construct_dubins_line(self, waypoints, dubins_path):
        #update path variables

    #def construct_dubins_circle_end(self, waypoints, dubins_path):
        #update path variables

