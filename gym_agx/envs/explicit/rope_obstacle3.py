import os
import sys
import logging
import numpy as np

import agx
import agxSDK
import agxOSG
import agxRender
from agxPythonModules.utils.numpy_utils import create_numpy_array

from gym_agx.sims.rope_obstacle3 import build_simulation, LENGTH
from gym_agx.envs import agx_env
from gym_agx.utils.agx_utils import to_numpy_array
from gym_agx.utils.agx_classes import CameraConfig
from gym_agx.rl.end_effector import EndEffector, EndEffectorConstraint

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'IRRELEVANT')

logger = logging.getLogger('gym_agx.envs')


class RopeObstacle3Env(agx_env.AgxEnv):
    """Subclass which inherits from DLO environment."""

    def __init__(self, n_substeps=1, reward_type="dense", observation_type="gt", headless=False, 
                 pushers=None, max_episode_length=40, **kwargs):
        self.headless = headless
        self.max_episode_length = max_episode_length
        self.timestep = 0
        
        camera_distance = 0.19  # meters
        camera_config = CameraConfig(
            eye=agx.Vec3(0, 0, camera_distance),
            center=agx.Vec3(0, 0, 0),
            up=agx.Vec3(0., 0., 0.),
            light_position=agx.Vec4(0.1, - camera_distance, camera_distance, 1.),
            light_direction=agx.Vec3(0., 0., -1.)
        )

        if not pushers:
            pusher = EndEffector(
                name='pusher',
                controllable=True,
                observable=True,
                max_velocity=5 / 100,  # m/s
                max_acceleration=10 / 100,  # m/s^2
            )
            pusher.add_constraint(name='pusher_joint_base_x',
                                  end_effector_dof=EndEffectorConstraint.Dof.X_TRANSLATION,
                                  compute_forces_enabled=False,
                                  velocity_control=True,
                                  compliance_control=False)
            pusher.add_constraint(name='pusher_joint_base_y',
                                  end_effector_dof=EndEffectorConstraint.Dof.Y_TRANSLATION,
                                  compute_forces_enabled=False,
                                  velocity_control=True,
                                  compliance_control=False)
            pusher.add_constraint(name='pusher_joint_base_z',
                                  end_effector_dof=EndEffectorConstraint.Dof.Z_TRANSLATION,
                                  compute_forces_enabled=False,
                                  velocity_control=False,
                                  compliance_control=False)
            pushers = [pusher]
        self.end_effectors = pushers


        if 'agxViewer' in kwargs:
            args = sys.argv + kwargs['agxViewer']
        else:
            args = sys.argv
        
        # Change window size
        args.extend(["--window", "600", "600"])

        no_graphics = headless and observation_type not in ("rgb", "depth", "rgb_and_depth")

        # Disable rendering in headless mode
        if headless:
            args.extend(["--osgWindow", "0"])

        if headless and observation_type == "gt":
            # args.extend(["--osgWindow", "0"])
            args.extend(["--agxOnly", "1", "--osgWindow", "0"])
        

        super(RopeObstacle3Env, self).__init__(scene_path=SCENE_PATH,
                                              n_substeps=n_substeps,
                                              observation_type=observation_type,
                                              n_actions=2,
                                              camera_pose=camera_config.camera_pose,
                                              image_size=(64, 64),
                                              no_graphics=no_graphics,
                                              args=args)

    def _build_simulation(self):
        pusher_init_pos = (np.random.uniform(-0.5, 0.5, size=2) * 0.05).tolist()
        self.pusher, self.rope = build_simulation(self.sim, pusher_init_pos)
        
        self.gravity = self.sim.getUniformGravity()
        self.time_step = self.sim.getTimeStep()
        logger.debug("Timestep after readFile is: {}".format(self.time_step))
        logger.debug("Gravity after readFile is: {}".format(self.gravity))

    def step(self, action):
        logger.info("step")

        rbs = [seg.getRigidBody() for seg in self.rope.segments]

        info = self._set_action(action)
        self._step_callback()

        if not self.headless or self.observation_type in ("rgb", "depth", "rgb_and_depth"):
            self._render_callback()

        # Compute rewards
        reward = 0.
        self.timestep += 1

        info['is_success'] = False
        done = info['is_success'] or self.timestep > self.max_episode_length

        obs = self._get_observation()

        return obs, reward, done, info
    
    def _set_action(self, action):
        info = dict()
        for end_effector in self.end_effectors:
            if end_effector.controllable:
                logger.debug("action: {}".format(action))
                info[end_effector.name] = end_effector.apply_control(self.sim, action, self.dt)

        return info

    def reset(self):
        logger.info("reset")
        self.timestep = 0
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
 
        force0 = np.random.uniform(-0.025, 0.025, size=2).tolist()
        force1 = np.random.uniform(-0.025, 0.025, size=2).tolist()
        self.rope.segments[0].getRigidBody().setForce(*force0, 0.)
        self.rope.segments[-1].getRigidBody().setForce(*force1, 0.)
        for _ in range(20):
            self._step_callback()

        for _ in range(20):
            for seg in self.rope.segments:
                seg.getRigidBody().setForce(0., 0., 0.)
                seg.getRigidBody().setVelocity(0., 0., 0.)
                seg.getRigidBody().setAngularVelocity(0., 0., 0.)
            self._step_callback()
        self._render_callback()

        obs = self._get_observation()
        return obs

    def _reset_sim(self):
        self.sim.cleanup(agxSDK.Simulation.CLEANUP_ALL, True)
        self._build_simulation()
        self._add_rendering(mode='osg')
        return True

    def _add_rendering(self, mode='osg'): 
        camera_distance = 0.5
        light_pos = agx.Vec4(0.05 / 2, - camera_distance, camera_distance, 1.)
        light_dir = agx.Vec3(0., 0., -1.)

        self.app.setAutoStepping(False)
        self.app.setEnableDebugRenderer(False)
        self.app.setEnableOSGRenderer(True)

        scene_decorator = self.app.getSceneDecorator()
        light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
        light_source_0.setPosition(light_pos)
        light_source_0.setDirection(light_dir)

        root = self.app.getRoot()
        rbs = self.sim.getRigidBodies()
        for rb in rbs:
            name = rb.getName()
            node = agxOSG.createVisual(rb, root)
            if name == "ground":
                agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
            elif name == "pusher":
                agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 0.0, 1.0, 1.0))
            elif "obstacle" in name:
                agxOSG.setDiffuseColor(node, agxRender.Color(1.0, 0.0, 0.0, 1.0))
            elif "dlo" in name:
                agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 1.0, 0.0, 1.0))
            else:  # Base segments
                agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
                agxOSG.setAlpha(node, 0.)

        scene_decorator = self.app.getSceneDecorator()
        light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
        light_source_0.setPosition(light_pos)
        light_source_0.setDirection(light_dir)
        scene_decorator.setEnableLogo(False)

    def _get_observation(self):
        rgb_buffer = None
        depth_buffer = None
        for buffer in self.render_to_image:
            name = buffer.getName()
            if name == 'rgb_buffer':
                rgb_buffer = buffer
            elif name == 'depth_buffer':
                depth_buffer = buffer

        assert self.observation_type in ("rgb", "depth", "rgb_and_depth", "pos", "pos_and_vel")

        if self.observation_type == "rgb":
            image_ptr = rgb_buffer.getImageData()
            image_data = create_numpy_array(image_ptr, (self.image_size[0], self.image_size[1], 3), np.uint8)
            obs = np.flipud(image_data)
        elif self.observation_type == "depth":
            image_ptr = depth_buffer.getImageData()
            image_data = create_numpy_array(image_ptr, (self.image_size[0], self.image_size[1]), np.float32)
            obs = np.flipud(image_data)
        elif self.observation_type == "rgb_and_depth":

            obs = np.zeros((self.image_size[0], self.image_size[1], 4), dtype=np.float32)

            image_ptr = rgb_buffer.getImageData()
            image_data = create_numpy_array(image_ptr, (self.image_size[0], self.image_size[1], 3), np.uint8)
            obs[:, :, 0:3] = np.flipud(image_data.astype(np.float32)) / 255

            image_ptr = depth_buffer.getImageData()
            image_data = create_numpy_array(image_ptr, (self.image_size[0], self.image_size[1]), np.float32)
            obs[:, :, 3] = np.flipud(image_data)
        elif self.observation_type == "pos":
            seg_pos = get_cable_segment_positions(cable=agxCable.Cable.find(self.sim, "DLO")).flatten()
            gripper = self.sim.getRigidBody("gripper_body")
            gripper_pos = to_numpy_array(gripper.getPosition())[0:3]
            cylinder = self.sim.getRigidBody("hollow_cylinder")
            cylinder_pos = cylinder.getPosition()
            ea = agx.EulerAngles().set(gripper.getRotation())
            gripper_rot = ea.y()

            obs = np.concatenate([gripper_pos, [gripper_rot], seg_pos, [cylinder_pos[0], cylinder_pos[1]]])

        elif  self.observation_type == "pos_and_vel":
            seg_pos, seg_vel = get_cable_segment_positions_and_velocities(cable=agxCable.Cable.find(self.sim, "DLO"))
            seg_pos = seg_pos.flatten()
            seg_vel = seg_vel.flatten()
            gripper = self.sim.getRigidBody("gripper_body")
            gripper_pos = to_numpy_array(gripper.getPosition())[0:3]
            gripper_vel = to_numpy_array(gripper.getVelocity())[0:3]
            gripper_vel_rot = to_numpy_array(gripper.getAngularVelocity())[2]
            cylinder = self.sim.getRigidBody("hollow_cylinder")
            cylinder_pos = cylinder.getPosition()
            ea = agx.EulerAngles().set(gripper.getRotation())
            gripper_rot = ea.y()

            obs = np.concatenate([gripper_pos, [gripper_rot], gripper_vel, [gripper_vel_rot], seg_pos, seg_vel, [cylinder_pos[0], cylinder_pos[1]]])

        return obs