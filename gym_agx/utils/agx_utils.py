import agx
import agxCollide
import agxSDK
import agxOSG
import agxRender

import numpy as np

import logging

logger = logging.getLogger(__name__)

# TODO: Is it really good practice to have classes being defined inside methods (instead of just instantiated)?


def create_help_text(sim, app, text_table=None):
    """Write help text indicating how to start simulation textTable is a table with strings that will be drawn above
     the default text.
    :param sim: AGX Simulation object
    :param app: OSG Example Application object
    :param text_table: table with text to be printed on screen
    """
    class HelpListener(agxSDK.StepEventListener):
        def __init__(self, app, text_table):
            super().__init__(agxSDK.StepEventListener.PRE_STEP)
            self.text_table = text_table
            self.app = app
            self.row = 31

        def pre(self, t):
            if t > 3.0:

                self.app.getSceneDecorator().setText(self.row, "", agx.Vec4f(1, 1, 1, 1))

                if self.text_table:
                    start_row = self.row - len(self.text_table)
                    for i, v in enumerate(self.text_table):
                        self.app.getSceneDecorator().setText(start_row + i - 1, "", agx.Vec4f(0.3, 0.6, 0.7, 1))

                self.getSimulation().remove(self)

        def addNotification(self):
            if self.text_table:
                start_row = self.row - len(self.text_table)
                for i, v in enumerate(self.text_table):
                    self.app.getSceneDecorator().setText(start_row + i - 1, v, agx.Vec4f(0.3, 0.6, 0.7, 1))

            self.app.getSceneDecorator().setText(self.row, "Press e to start simulation", agx.Vec4f(0.3, 0.6, 0.7, 1))

    sim.add(HelpListener(sim, app, text_table))


def create_body(sim, shape, **args):
    """Helper function that creates a RigidBody according to the given definition.
    Returns the body itself, it's geometry and the OSG node that was created for it.
    :param sim: AGX Simulation object
    :param shape: shape of object - agxCollide.Shape.
    :param args: The definition contains the following parts:
    name - string. Optional. Defaults to "". The name of the new body.
    geometryTransform - agx.AffineMatrix4x4. Optional. Defaults to identity transformation. The local transformation of
    the shape relative to the body.
    motionControl - agx.RigidBody.MotionControl. Optional. Defaults to DYNAMICS.
    material - agx.Material. Optional. Ignored if not given. Material assigned to the geometry created for the body.
    :return: body, geometry
    """
    geometry = agxCollide.Geometry(shape)

    if "geometryTransform" not in args.keys():
        geometry_transform = agx.AffineMatrix4x4()
    else:
        geometry_transform = args["geometryTransform"]

    if "name" in args.keys():
        body = agx.RigidBody(args["name"])
    else:
        body = agx.RigidBody("")

    body.add(geometry, geometry_transform)

    if "position" in args.keys():
        body.setPosition(args["position"])

    if "motionControl" in args.keys():
        body.setMotionControl(args["motionControl"])

    if "material" in args.keys():
        geometry.setMaterial(args["material"])

    sim.add(body)

    return body, geometry


def create_info_printer(sim, app, text_table=None, text_color=None):
    """Write information to screen from lambda functions during the simulation.
    :param sim: AGX Simulation object
    :param app: OSG Example Application object
    :param text_table: table with text to be printed on screen
    :param text_color: Color of text
    """
    class InfoPrinter(agxSDK.StepEventListener):
        def __init__(self, app, text_table, text_color):
            super().__init__(agxSDK.StepEventListener.POST_STEP)
            self.text_table = text_table
            self.text_color = text_color
            self.app = app
            self.row = 31

        def post(self, t):
            if self.textTable:
                color = agx.Vec4(0.3, 0.6, 0.7, 1)
                if self.text_color:
                    color = self.text_color
                for i, v in enumerate(self.text_table):
                    self.app.getSceneDecorator().setText(i, str(v[0]) + " " + v[1](), color)

    sim.add(InfoPrinter(sim, app, text_table, text_color))


def add_wire_rendering(sim, length):
    """Create ExampleApplication instance and add rendering information.
    :param sim: AGX Simulation object
    :param length: Length of the DLO
    :return: app
    """
    camera_distance = 0.5
    light_pos = agx.Vec4(length / 2, - camera_distance, camera_distance, 1.)
    light_dir = agx.Vec3(0., 0., -1.)

    app = agxOSG.ExampleApplication(sim)
    app.setAutoStepping(False)
    app.setEnableDebugRenderer(False)
    app.setEnableOSGRenderer(True)

    root = app.getRoot()
    rbs = sim.getRigidBodies()
    for rb in rbs:
        if rb.getName() == "ground":
            ground_node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(ground_node, agxRender.Color(1.0, 1.0, 1.0, 1.0))
        elif rb.getName() == "gripper_left":
            gripper_left_node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(gripper_left_node, agxRender.Color(1.0, 0.0, 0.0, 1.0))
        elif rb.getName() == "gripper_right":
            gripper_right_node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(gripper_right_node, agxRender.Color(0.0, 0.0, 1.0, 1.0))
        else:  # Cable segments
            cable_node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(cable_node, agxRender.Color(0.0, 1.0, 0.0, 1.0))

    scene_decorator = app.getSceneDecorator()
    light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
    light_source_0.setPosition(light_pos)
    light_source_0.setDirection(light_dir)

    return app, root


def get_state(sim):
    """Return dictionary with object list.
    :param sim: AGX simulation object
    :return: dictionary with rigid objects
    """
    rbs = sim.getRigidBodies()
    state = dict(list(enumerate(rbs)))
    return state


def to_numpy_array(agx_list):
    """Convert from AGX data structure to NumPy array.
    :param agx_list: AGX data structure
    :return: NumPy array
    """
    agx_type = type(agx_list)
    if agx_type == agx.Vec3:
        np_array = np.zeros(shape=(3,), dtype=float)
        for i in range(3):
            np_array[i] = agx_list[i]
    elif agx_type == agx.Quat:
        np_array = np.zeros(shape=(4,), dtype=float)
        for i in range(4):
            np_array[i] = agx_list[i]
    else:
        logger.warning('Conversion for type {} type is not supported.'.format(agx_type))

    return np_array


def to_agx_list(np_array, agx_type):
    """Convert from Numpy array to AGX data structure.
    :param np_array:  NumPy array
    :param agx_type: Target AGX data structure
    :return: AGX data object
    """
    agx_list = None
    if agx_type == agx.Vec3:
        agx_list = agx.Vec3(np_array[0].item(), np_array[1].item(), np_array[2].item())
    elif agx_type == agx.Quat:
        agx_list = agx.Quat(np_array[0].item(), np_array[1].item(), np_array[2].item(), np_array[3].item())
    else:
        logger.warning('Conversion for type {} type is not supported.'.format(agx_type))

    return agx_list


def get_cable_state(cable):
    """Get AGX Cable segments' positions and rotations.
    :param cable: AGX Cable object
    :return: NumPy array with segments' position and rotations
    """
    num_segments = cable.getNumSegments()
    cable_state = np.zeros(shape=(7, num_segments))
    segment_iterator = cable.begin()
    for i in range(num_segments):
        if not segment_iterator.isEnd():
            position = segment_iterator.getGeometry().getPosition()
            cable_state[:3, i] = to_numpy_array(position)

            rotation = segment_iterator.getGeometry().getRotation()
            cable_state[3:, i] = to_numpy_array(rotation)

            segment_iterator.inc()
        else:
            logger.error('AGX segment iteration finished early. Number or cable segments may be wrong.')

    return cable_state


def ctrl_set_action(sim, obj_name, pos_ctrl, rot_ctrl, grip_ctrl=None):
    """Apply action to simulation.
    :param sim: AGX simulation object
    :param obj_name: Kinematic Rigid Object (gripper)
    :param pos_ctrl: Displacement of object in x,y,z coordinates
    :param rot_ctrl: Rotation of object around x,y,z axes
    :param grip_ctrl: (optional) Displacement of object relative to deformable object
    """
    obj = sim.getObject(obj_name)
    transform = agx.AffineMatrix4x4()
    transform.setTranslate(pos_ctrl)
    transform.setRotate(rot_ctrl)
    obj.moveTo(transform)

    # TODO: Implement a change in relative position of gripper on deformable object
