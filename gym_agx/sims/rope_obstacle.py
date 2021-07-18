"""Simulation of cable pushing

This module creates the simulation files which will be used in cable_pushing environments.
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
import numpy as np

# Local modules
from gym_agx.utils.agx_utils import create_body, create_locked_prismatic_base, save_simulation, to_numpy_array
from gym_agx.utils.utils import all_points_below_z, point_inside_polygon
from gym_agx.utils.agx_classes import KeyboardMotorHandler

logger = logging.getLogger('gym_agx.sims')

FILE_NAME = "cable_closing"
# Simulation parameters
TIMESTEP = 1 / 100
N_SUBSTEPS = 20
GRAVITY = True
# Rubber band parameters
LENGTH = 0.15
DLO_CIRCLE_STEPS = 20
RADIUS = 0.004  # meters
RESOLUTION = 100  # segments per meter
PEG_POISSON_RATIO = 0.1  # no unit
YOUNG_MODULUS_BEND = 2.5e5
YOUNG_MODULUS_TWIST = 1e6
YOUNG_MODULUS_STRETCH = 5e6

# Aluminum Parameters
ALUMINUM_POISSON_RATIO = 0.35  # no unit
ALUMINUM_YOUNG_MODULUS = 69e9  # Pascals
ALUMINUM_YIELD_POINT = 5e7  # Pascals

# Ground Parameters
EYE = agx.Vec3(0, 0, 0.65)
CENTER = agx.Vec3(0, 0, 0)
UP = agx.Vec3(0., 1., 0.0)


# Control parameters
MAX_MOTOR_FORCE = 2
OBSTACLE_POSITIONS = [[0.0,0.05], [-0.075,0.05], [0.075, 0.05]]
R_OBSTACLE = 0.005
OFFSET_Y = 0.1


def add_rendering(sim):
    app = agxOSG.ExampleApplication(sim)

    # Set renderer
    app.setAutoStepping(True)
    app.setEnableDebugRenderer(False)
    app.setEnableOSGRenderer(True)

    # Create scene graph for rendering
    root = app.getSceneRoot()
    rbs = sim.getRigidBodies()
    for rb in rbs:
        node = agxOSG.createVisual(rb, root)
        if rb.getName() == "ground":
            agxOSG.setDiffuseColor(node, agxRender.Color(0.8, 0.8, 0.8, 1.0))
        elif rb.getName() == "walls":
            agxOSG.setDiffuseColor(node, agxRender.Color.Burlywood())
        elif rb.getName() == "cylinder":
            agxOSG.setDiffuseColor(node, agxRender.Color.DarkGray())
        elif rb.getName() == "cylinder_inner":
            agxOSG.setDiffuseColor(node, agxRender.Color.LightSteelBlue())
        elif 'gripper' in rb.getName():
            agxOSG.setDiffuseColor(node, agxRender.Color(0., 0., 0., 1.0))
        elif rb.getName() == "gripper_0":
            agxOSG.setDiffuseColor(node, agxRender.Color(1.0, 0., 0., 1.0))
        elif rb.getName() == "gripper_1":
            agxOSG.setDiffuseColor(node, agxRender.Color(0., 0., 1.0, 1.0))
        elif "dlo" in  rb.getName():  # Cable segments
            agxOSG.setDiffuseColor(node, agxRender.Color(0.1, 0.5, 0.0, 1.0))
            agxOSG.setAmbientColor(node, agxRender.Color(0.2, 0.5, 0.0, 1.0))
        elif rb.getName() == "obstacle":
            agxOSG.setDiffuseColor(node, agxRender.Color(0.5, 0.5, 0.5, 1.0))
        else:
            agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
            agxOSG.setAlpha(node, 0.0)

    # Set rendering options
    scene_decorator = app.getSceneDecorator()
    scene_decorator.setEnableLogo(False)
    scene_decorator.setBackgroundColor(agxRender.Color(1.0, 1.0,1.0, 1.0))

    return app


def build_simulation(sim, n_grippers):
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
    material_hard = agx.Material("Aluminum")
    material_hard_bulk = material_hard.getBulkMaterial()
    material_hard_bulk.setPoissonsRatio(ALUMINUM_POISSON_RATIO)
    material_hard_bulk.setYoungsModulus(ALUMINUM_YOUNG_MODULUS)

    # Create box with pocket
    side_length = 0.15
    thickness_outer_wall = 0.01
    body = create_body(name="ground", shape=agxCollide.Box(side_length, side_length, 0.01),
                       position=agx.Vec3(0, 0, -0.005),
                       motion_control=agx.RigidBody.STATIC,
                       material=material_hard)

    sim.add(body)

    body = create_body(name="walls", shape=agxCollide.Box(thickness_outer_wall, side_length, 0.04),
                       position=agx.Vec3(side_length + thickness_outer_wall, 0, 0.0),
                       motion_control=agx.RigidBody.STATIC,
                       material=material_hard)
    sim.add(body)

    body = create_body(name="walls", shape=agxCollide.Box(thickness_outer_wall, side_length, 0.04),
                       position=agx.Vec3(-(side_length + thickness_outer_wall), 0, 0.0),
                       motion_control=agx.RigidBody.STATIC,
                       material=material_hard)
    sim.add(body)

    body = create_body(name="walls",
                       shape=agxCollide.Box(side_length + 2 * thickness_outer_wall, thickness_outer_wall, 0.04),
                       position=agx.Vec3(0, -(side_length + thickness_outer_wall), 0.0),
                       motion_control=agx.RigidBody.STATIC,
                       material=material_hard)
    sim.add(body)

    body = create_body(name="walls",
                       shape=agxCollide.Box(side_length + 2 * thickness_outer_wall, thickness_outer_wall, 0.04),
                       position=agx.Vec3(0, side_length + thickness_outer_wall, 0.0),
                       motion_control=agx.RigidBody.STATIC,
                       material=material_hard)
    sim.add(body)

    # Create grippers
    grippers, prismatic_bases = [], []
    for i in range(n_grippers):
        name = f"gripper_{i}"
        offset_x = -(LENGTH/2) + i * LENGTH / (n_grippers - 1)
        gripper = create_body(name=name,
                              shape=agxCollide.Sphere(0.005),
                              position=agx.Vec3(offset_x, OFFSET_Y, 0.0025),
                              motion_control=agx.RigidBody.DYNAMICS,
                              material=material_hard)
        gripper.getRigidBody(name).getGeometry(name).setEnableCollisions(False)
        sim.add(gripper)

        prismatic_base = create_locked_prismatic_base(name, gripper.getRigidBody(name),
                                                      position_ranges=[(-side_length*2, side_length*2),
                                                                       (-side_length*2, side_length*2),
                                                                       (-0.1, 0.01)],
                                                      motor_ranges=[(-MAX_MOTOR_FORCE, MAX_MOTOR_FORCE),
                                                                    (-MAX_MOTOR_FORCE, MAX_MOTOR_FORCE),
                                                                    (-MAX_MOTOR_FORCE, MAX_MOTOR_FORCE)],
                                                      lock_status=[False, False, False])
        sim.add(prismatic_base)

        grippers.append(gripper)
        prismatic_bases.append(prismatic_base)
 
    # Create obstacles
    rotation_cylinder = agx.OrthoMatrix3x3()
    rotation_cylinder.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.Z_AXIS())
    obs_pos = OBSTACLE_POSITIONS
    for i in range(0, len(obs_pos)):
        obstacle = create_body(name="obstacle",
                               shape=agxCollide.Cylinder(2 * R_OBSTACLE, 0.1),
                               position=agx.Vec3(obs_pos[i][0], obs_pos[i][1], 0.005),
                               motion_control=agx.RigidBody.STATIC,
                               material=material_hard)
        sim.add(obstacle)
        obstacle.setRotation(rotation_cylinder)

    # Create rope and set name + properties
    dlo = agxCable.Cable(RADIUS, RESOLUTION)
    dlo.setName("DLO")
    material_rubber_band= dlo.getMaterial()
    rubber_band_material = material_rubber_band.getBulkMaterial()
    rubber_band_material.setPoissonsRatio(PEG_POISSON_RATIO)
    properties = dlo.getCableProperties()
    properties.setYoungsModulus(YOUNG_MODULUS_BEND, agxCable.BEND)
    properties.setYoungsModulus(YOUNG_MODULUS_TWIST, agxCable.TWIST)
    properties.setYoungsModulus(YOUNG_MODULUS_STRETCH, agxCable.STRETCH)

    for i, gripper in enumerate(grippers):
        dlo.add(agxCable.FreeNode(gripper.getRigidBody(f"gripper_{i}").getPosition()))

    # Set angular damping for segments
    sim.add(dlo)
    segment_iterator = dlo.begin()
    n_segments = dlo.getNumSegments()
    segments = []
    for i in range(n_segments):
        if not segment_iterator.isEnd():
            seg = segment_iterator.getRigidBody()
            segments.append(seg)
            seg.setAngularVelocityDamping(2e4)
            segment_iterator.inc()
            mass_props = seg.getMassProperties()
            mass_props.setMass(1.25*mass_props.getMass())

    n_segments = len(segments)
    for i, gripper in enumerate(grippers):
        name = f"gripper_{i}"
        seg_idx = round(i / (n_grippers - 1) * (n_segments - 1))
        
        h = agx.HingeFrame()
        h.setCenter(gripper.getRigidBody(name).getPosition())
        h.setAxis(agx.Vec3(0, 0, 1))
        l = agx.Hinge(h, segments[seg_idx], gripper.getRigidBody(name))
        sim.add(l)

    # Try to initialize dlo
    report = dlo.tryInitialize()
    # if report.successful():
    #     print("Successful dlo initialization.")
    # else:
    #     print(report.getActualError())

    # Add rope to simulation
    sim.add(dlo)

    # Set rope material
    material_rubber_band = dlo.getMaterial()
    material_rubber_band.setName("rope_material")

    contactMaterial = sim.getMaterialManager().getOrCreateContactMaterial(material_hard, material_rubber_band)
    contactMaterial.setYoungsModulus(1e12)
    fm = agx.IterativeProjectedConeFriction()
    fm.setSolveType(agx.FrictionModel.DIRECT)
    contactMaterial.setFrictionModel(fm)

    # Add keyboard listener
    # motor_x_0 = sim.getConstraint1DOF("gripper_0_joint_base_x").getMotor1D()
    # motor_y_0 = sim.getConstraint1DOF("gripper_0_joint_base_y").getMotor1D()
    # motor_x_1 = sim.getConstraint1DOF("gripper_1_joint_base_x").getMotor1D()
    # motor_y_1 = sim.getConstraint1DOF("gripper_1_joint_base_y").getMotor1D()
    # key_motor_map = {agxSDK.GuiEventListener.KEY_Up: (motor_y_0, 0.5),
    #                  agxSDK.GuiEventListener.KEY_Down: (motor_y_0, -0.5),
    #                  agxSDK.GuiEventListener.KEY_Right: (motor_x_0, 0.5),
    #                  agxSDK.GuiEventListener.KEY_Left: (motor_x_0, -0.5),
    #                  120: (motor_x_1, 0.5),
    #                  60: (motor_x_1, -0.5),
    #                  97: (motor_y_1, 0.5),
    #                  121: (motor_y_1, -0.5)}
    # sim.add(KeyboardMotorHandler(key_motor_map))

    rbs = dlo.getRigidBodies()
    for i in range(len(rbs)):
        rbs[i].setName('dlo_' + str(i+1))

    return sim


def compute_segments_pos(sim):
    segments_pos = []
    dlo = agxCable.Cable.find(sim, "DLO")
    segment_iterator = dlo.begin()
    n_segments = dlo.getNumSegments()
    for i in range(n_segments):
        if not segment_iterator.isEnd():
            pos = segment_iterator.getGeometry().getPosition()
            segments_pos.append(to_numpy_array(pos))
            segment_iterator.inc()

    return segments_pos


def main(args):
    # Build simulation object
    sim = build_simulation()

    # Save simulation to file
    success = save_simulation(sim, FILE_NAME)
    if success:
        logger.debug("Simulation saved!")
    else:
        logger.debug("Simulation not saved!")

    # Add app
    app = add_rendering(sim)
    app.init(agxIO.ArgumentParser([sys.executable, '--window', '600', '600'] + args))
    app.setTimeStep(TIMESTEP)
    app.setCameraHome(EYE, CENTER, UP)
    app.initSimulation(sim, True)

    reward_type = "sparse"
    segment_pos_old = compute_segments_pos(sim)
    for _ in range(10000):
        sim.stepForward()
        app.executeOneStepWithGraphics()

        # Compute reward
        segment_pos = compute_segments_pos(sim)

        # Compute reward
        reward = 0.

        segment_pos_old = segment_pos

        if reward !=0:
            print("reward: ", reward)


if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)
