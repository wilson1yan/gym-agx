"""Simulation of rope pushing

This module creates the simulation files which will be used in PushRope environments.
TODO: Instead of setting all parameters in this file, there should be a parameter file (e.g. YAML or XML).
"""
# AGX Dynamics imports
import agx
import agxPython
import agxCollide
import agxSDK
import agxCable
import agxIO
import agxOSG
import agxRender

# Python modules
import sys
import logging

# Local modules
from gym_agx.utils.agx_utils import create_body, create_locked_prismatic_base, save_simulation, save_goal_simulation
from gym_agx.utils.agx_classes import KeyboardMotorHandler

logger = logging.getLogger('gym_agx.sims')

FILE_NAME = "rope_obstacle3"
# Simulation parameters
TIMESTEP = 1 / 1000
N_SUBSTEPS = 20
GRAVITY = True
# Rope parameters
RADIUS = 0.0015  # meters
LENGTH = 0.075  # meters
RESOLUTION = 500  # segments per meter
ROPE_POISSON_RATIO = 0.01  # no unit
YOUNG_MODULUS_BEND = 1e3  # 1e5
YOUNG_MODULUS_TWIST = 1e3  # 1e10
YOUNG_MODULUS_STRETCH = 8e9  # Pascals
ROPE_ROUGHNESS = 10
ROPE_ADHESION = 0.001
ROPE_DENSITY = 3.5  # kg/m3
# Aluminum Parameters
ALUMINUM_POISSON_RATIO = 0.35  # no unit
ALUMINUM_YOUNG_MODULUS = 69e9  # Pascals
ALUMINUM_YIELD_POINT = 5e7  # Pascals
# Pusher Parameters
PUSHER_RADIUS = 0.002
PUSHER_HEIGHT = 0.01
PUSHER_ROUGHNESS = 1
PUSHER_ADHESION = 0
PUSHER_ADHESION_OVERLAP = 0
# Ground Parameters
GROUND_ROUGHNESS = 0.1
GROUND_ADHESION = 1e2
GROUND_ADHESION_OVERLAP = 0
# NOTE: At this overlap, no force is applied. At lower overlap, the adhesion force will work, at higher overlap, the
# (usual) contact forces will be applied
# Rendering Parameters
GROUND_LENGTH_X = 0.05
GROUND_LENGTH_Y = 0.05
GROUND_WIDTH = 0.001  # meters
CABLE_GRIPPER_RATIO = 2
SIZE_GRIPPER = CABLE_GRIPPER_RATIO * RADIUS
EYE = agx.Vec3(0, 0, 0.5)
CENTER = agx.Vec3(0, 0, 0)
UP = agx.Vec3(0., 0., 0.)
# Control parameters
MAX_MOTOR_FORCE = 1  # Newtons


def add_rendering(sim):
    camera_distance = 0.5
    light_pos = agx.Vec4(LENGTH / 2, - camera_distance, camera_distance, 1.)
    light_dir = agx.Vec3(0., 0., -1.)

    app = agxOSG.ExampleApplication(sim)

    app.setAutoStepping(True)
    app.setEnableDebugRenderer(False)
    app.setEnableOSGRenderer(True)

    scene_decorator = app.getSceneDecorator()
    light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
    light_source_0.setPosition(light_pos)
    light_source_0.setDirection(light_dir)

    root = app.getRoot()
    rbs = sim.getRigidBodies()
    for rb in rbs:
        name = rb.getName()
        node = agxOSG.createVisual(rb, root)
        if name == "ground":
            agxOSG.setDiffuseColor(node, agxRender.Color.Gray())
        elif name == "pusher":
            agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 0.0, 1.0, 1.0))
        elif "obstacle" in name:
            agxOSG.setDiffuseColor(node, agxRender.Color(1.0, 0.0, 0.0, 1.0))
        elif "dlo" in name:
            agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 1.0, 0.0, 1.0))
        else:  # Base segments
            agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
            agxOSG.setAlpha(node, 0.2)

    scene_decorator = app.getSceneDecorator()
    light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
    light_source_0.setPosition(light_pos)
    light_source_0.setDirection(light_dir)
    scene_decorator.setEnableLogo(False)

    return app


def build_simulation(sim, pusher_init_pos):
    # By default the gravity vector is 0,0,-9.81 with a uniform gravity field. (we CAN change that
    # too by creating an agx.PointGravityField for example).
    # AGX uses a right-hand coordinate system (That is Z defines UP. X is right, and Y is into the screen)
    if not GRAVITY:
        logger.info("Gravity off.")
        g = agx.Vec3(0, 0, 0)  # remove gravity
        sim.setUniformGravity(g)

    # Get current delta-t (timestep) that is used in the simulation?
    dt = sim.getTimeStep()
    logger.debug("default dt = {}".format(dt))

    # Change the timestep
    sim.setTimeStep(TIMESTEP)

    # Confirm timestep changed
    dt = sim.getTimeStep()
    logger.debug("new dt = {}".format(dt))

    # Define materials
    material_ground = agx.Material("Aluminum")
    bulk_material = material_ground.getBulkMaterial()
    bulk_material.setPoissonsRatio(ALUMINUM_POISSON_RATIO)
    bulk_material.setYoungsModulus(ALUMINUM_YOUNG_MODULUS)
    surface_material = material_ground.getSurfaceMaterial()
    surface_material.setRoughness(GROUND_ROUGHNESS)
    surface_material.setAdhesion(GROUND_ADHESION, GROUND_ADHESION_OVERLAP)

    material_pusher = agx.Material("Aluminum")
    bulk_material = material_pusher.getBulkMaterial()
    bulk_material.setPoissonsRatio(ALUMINUM_POISSON_RATIO)
    bulk_material.setYoungsModulus(ALUMINUM_YOUNG_MODULUS)
    surface_material = material_pusher.getSurfaceMaterial()
    surface_material.setRoughness(PUSHER_ROUGHNESS)
    surface_material.setAdhesion(PUSHER_ADHESION, PUSHER_ADHESION_OVERLAP)

    # Create a ground plane
    ground = create_body(name="ground", shape=agxCollide.Box(GROUND_LENGTH_X, GROUND_LENGTH_Y, GROUND_WIDTH),
                         position=agx.Vec3(0, 0, -GROUND_WIDTH / 2),
                         motion_control=agx.RigidBody.STATIC,
                         material=material_ground)
    sim.add(ground)

    bounding_box = create_body(name="bounding_box_1", shape=agxCollide.Box(GROUND_LENGTH_X, GROUND_WIDTH, RADIUS * 4),
                               position=agx.Vec3(0, GROUND_LENGTH_Y, RADIUS * 4),
                               motion_control=agx.RigidBody.STATIC,
                               material=material_ground)
    sim.add(bounding_box)
    bounding_box = create_body(name="bounding_box_2", shape=agxCollide.Box(GROUND_LENGTH_X, GROUND_WIDTH, RADIUS * 4),
                               position=agx.Vec3(0, - GROUND_LENGTH_Y, RADIUS * 4),
                               motion_control=agx.RigidBody.STATIC,
                               material=material_ground)
    sim.add(bounding_box)
    bounding_box = create_body(name="bounding_box_3", shape=agxCollide.Box(GROUND_WIDTH, GROUND_LENGTH_Y, RADIUS * 4),
                               position=agx.Vec3(GROUND_LENGTH_X, 0, RADIUS * 4),
                               motion_control=agx.RigidBody.STATIC,
                               material=material_ground)
    sim.add(bounding_box)
    bounding_box = create_body(name="bounding_box_4", shape=agxCollide.Box(GROUND_WIDTH, GROUND_LENGTH_Y, RADIUS * 4),
                               position=agx.Vec3(- GROUND_LENGTH_X, 0, RADIUS * 4),
                               motion_control=agx.RigidBody.STATIC,
                               material=material_ground)
    sim.add(bounding_box)

    rotation_cylinder = agx.OrthoMatrix3x3()
    rotation_cylinder.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.Z_AXIS())

    pusher = create_body(name="pusher",
                         shape=agxCollide.Cylinder(PUSHER_RADIUS, PUSHER_HEIGHT),
                         position=agx.Vec3(*pusher_init_pos, PUSHER_HEIGHT / 2),
                         rotation=rotation_cylinder,
                         motion_control=agx.RigidBody.DYNAMICS,
                         material=material_pusher)
    sim.add(pusher)

    # Create base for pusher motors
    pusher_pos_x, pusher_pos_y = pusher_init_pos
    prismatic_base = create_locked_prismatic_base("pusher", pusher.getRigidBody("pusher"),
                                                  position_ranges=[(-GROUND_LENGTH_X - pusher_pos_x, GROUND_LENGTH_X - pusher_pos_x),
                                                            (-GROUND_LENGTH_Y - pusher_pos_y, GROUND_LENGTH_Y - pusher_pos_y), (0., 3*RADIUS)],
                                                  motor_ranges=[(-MAX_MOTOR_FORCE, MAX_MOTOR_FORCE),
                                                         (-MAX_MOTOR_FORCE, MAX_MOTOR_FORCE),
                                                         (-MAX_MOTOR_FORCE, MAX_MOTOR_FORCE)])

    sim.add(prismatic_base)

    return pusher


def main(args):
    # Instantiate a simulation
    sim = agxSDK.Simulation()
    # Build simulation object
    build_simulation(sim)

    # Save simulation to file
    save_simulation(sim, "test_sim.agx")

    # # Save goal simulation to file (but first make grippers static, remove clutter and rename)
    # cable = agxCable.Cable.find(sim, "DLO")
    # cable.setName("DLO_goal")
    # success = save_goal_simulation(sim, FILE_NAME, ['obstacle', 'ground', "pusher_prismatic_base", "bounding_box_1",
    #                                                 "bounding_box_2", "bounding_box_3", "bounding_box_4"])
    # if success:
    #     logger.debug("Goal simulation saved!")
    # else:
    #     logger.debug("Goal simulation not saved!")

    # Render simulation
    app = add_rendering(sim)
    app.init(agxIO.ArgumentParser([sys.executable] + args))
    app.setCameraHome(EYE, CENTER, UP)  # should only be added after app.init
    app.initSimulation(sim, True)  # This changes timestep and Gravity!
    sim.setTimeStep(TIMESTEP)
    if not GRAVITY:
        logger.info("Gravity off.")
        g = agx.Vec3(0, 0, 0)  # remove gravity
        sim.setUniformGravity(g)

    n_seconds = 60
    n_steps = int(n_seconds / (TIMESTEP * N_SUBSTEPS))
    for k in range(n_steps):
        app.executeOneStepWithGraphics()

        t = sim.getTimeStamp()
        t_0 = t
        while t < t_0 + TIMESTEP * N_SUBSTEPS:
            sim.stepForward()
            t = sim.getTimeStamp()

    # Save simulation to file
    success = save_simulation(sim, FILE_NAME)
    if success:
        logger.debug("Simulation saved!")
    else:
        logger.debug("Simulation not saved!")


if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)
