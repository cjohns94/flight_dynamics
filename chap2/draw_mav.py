"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        1/13/2021 - TWM
"""
import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import Euler2Rotation


class DrawMav:
    def __init__(self, state, window):
        """
        Draw the MAV.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.pn  # north position
            state.pe  # east position
            state.h   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        # get points that define the non-rotated, non-translated spacecraft and the mesh colors
        self.sc_points, self.sc_meshColors = self.get_points()

        sc_position = np.array([[state.pn], [state.pe], [-state.h]])  # NED coordinates
        # attitude of spacecraft as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining spacecraft
        rotated_points = self.rotate_points(self.sc_points, R)
        translated_points = self.translate_points(rotated_points, sc_position)

        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        self.sc_body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.sc_meshColors,  # defines mesh colors (Nx1)
                                      drawEdges=True,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
        window.addItem(self.sc_body)  # add body to plot

    def update(self, state):
        sc_position = np.array([[state.pn], [state.pe], [-state.h]])  # NED coordinates
        # attitude of spacecraft as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining spacecraft
        rotated_points = self.rotate_points(self.sc_points, R)
        translated_points = self.translate_points(rotated_points, sc_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        # draw spacecraft by resetting mesh using rotated and translated points
        self.sc_body.setMeshData(vertexes=mesh, vertexColors=self.sc_meshColors)

    def rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points

    def translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1, points.shape[1]]))
        return translated_points

    def get_points(self):
        """"
            Points that define the MAV, and the colors of the triangular mesh
            Define the points on the MAV following information in Appendix C.3
        """
       
        # points are in XYZ coordinates
        #   define the points on the MAV according to Fig 2.14
        points = np.array([[2.1, 0, 0],  # point 1 [0]
                           [1.1, 0.6, -0.6],  # point 2 [1]
                           [1.1, -0.6, -0.6],  # point 3 [2]
                           [1.1, -0.6, 0.6],  # point 4 [3]
                           [1.1, 0.6, 0.6],  # point 5 [4]
                           [-4.3, 0, 0],  # point 6 [5]
                           [0, 3, 0],  # point 7 [6]
                           [-1.2, 3, 0],  # point 8 [7]
                           [-1.2, -3, 0],  # point 9 [8]
                           [0, -3, 0],  # point 10 [9]
                           [-3.2, 1.5, 0],  # point 11 [10]
                           [-4.3, 1.5, 0],  # point 12 [10]
                           [-4.3, -1.5, 0],  # point 13 [10]
                           [-3.2, -1.5, 0],  # point 14 [10]
                           [-3.2, 0, 0],  # point 15 [10]
                           [-4.3, 0, -1.2]  # point 16 [10]                           
                           ]).T

        # scale points for better rendering
        scale = 10
        points = scale * points

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)
        meshColors[0] = yellow  # nose 1
        meshColors[1] = yellow  # nose 2
        meshColors[2] = yellow  # nose 3
        meshColors[3] = yellow  # nose 4
        meshColors[4] = blue  # fuselage 1
        meshColors[5] = blue  # fueslage 2
        meshColors[6] = blue  # fuselage 3
        meshColors[7] = blue  # fuselage 4
        meshColors[8] = green  # wing 1
        meshColors[9] = green  # wing 2
        meshColors[10] = red  # tailwing 1
        meshColors[11] = red  # tailwing 2
        meshColors[12] = red #tail
        
        return points, meshColors
    def points_to_mesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)
        """
        points = points.T
        mesh = np.array([[points[0], points[1], points[2]],  # nose 1
                         [points[0], points[2], points[3]],  # nose 2
                         [points[0], points[4], points[3]],  # nose 3
                         [points[0], points[1], points[4]],  # nose 4
                         [points[1], points[2], points[5]],  # fuselage 1
                         [points[2], points[3], points[5]],  # fuselage 2
                         [points[4], points[3], points[5]],  # fuselage 3
                         [points[1], points[4], points[5]],   # fuselage 4
                         [points[6], points[7], points[9]],   # wing 1
                         [points[8], points[7], points[9]],   # wing 2
                         [points[10], points[11], points[13]],   # tailwing 1
                         [points[11], points[13], points[12]],   # tailwing 2
                         [points[14], points[15], points[5]]   # tail
        ])

        
        return mesh
