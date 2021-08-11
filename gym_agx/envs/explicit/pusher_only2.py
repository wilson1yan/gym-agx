import os
import sys
import logging
import numpy as np

import agx
import agxSDK
import agxOSG
import agxRender
from agxPythonModules.utils.numpy_utils import create_numpy_array

from gym_agx.sims.pusher_only import build_simulation
from gym_agx.envs import agx_env
from gym_agx.utils.agx_utils import to_numpy_array
from gym_agx.utils.agx_classes import CameraConfig
from gym_agx.rl.end_effector import EndEffector, EndEffectorConstraint

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'IRRELEVANT')

logger = logging.getLogger('gym_agx.envs')


class PusherOnly2Env(agx_env.AgxEnv):
    """Subclass which inherits from DLO environment."""

    def __init__(self, n_substeps=1, reward_type="dense", observation_type="gt", headless=False, 
                 pushers=None, max_episode_length=80, **kwargs):
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
                max_velocity=3200 / 100,  # m/s
                max_acceleration=6400 / 100,  # m/s^2
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
        

        super(PusherOnly2Env, self).__init__(scene_path=SCENE_PATH,
                                              n_substeps=n_substeps,
                                              observation_type=observation_type,
                                              n_actions=2,
                                              camera_pose=camera_config.camera_pose,
                                              image_size=(64, 64),
                                              no_graphics=no_graphics,
                                              args=args)

    def construct_policy(self):
        goal_pos = np.random.uniform(-1, 1, size=2) * 0.05

        def policy():
            nonlocal goal_pos
            if self.timestep % 5 == 0:
                goal_pos = np.random.uniform(-1, 1, size=2) * 0.05 
            pusher_pos = to_numpy_array(self.pusher.getRigidBody('pusher').getPosition())[:2]
            direction = goal_pos - pusher_pos
            direction /= np.linalg.norm(direction)
            return direction
        return policy


    def _build_simulation(self):
        pusher_init_pos = (np.random.uniform(-0.5, 0.5, size=2) * 0.05).tolist()
        self.pusher = build_simulation(self.sim, pusher_init_pos)
        
        self.gravity = self.sim.getUniformGravity()
        self.time_step = self.sim.getTimeStep()
        logger.debug("Timestep after readFile is: {}".format(self.time_step))
        logger.debug("Gravity after readFile is: {}".format(self.gravity))

    def step(self, action):
        logger.info("step")

        pusher_pos = to_numpy_array(self.pusher.getRigidBody('pusher').getPosition())[:2]
        goal_pos = pusher_pos + action * 0.0025

        for i in range(10):
            pusher_pos = to_numpy_array(self.pusher.getRigidBody('pusher').getPosition())[:2]
            direction = goal_pos - pusher_pos
            direction = direction / np.linalg.norm(direction)
            self._set_action(direction)
            self._step_callback()

            pusher_pos = to_numpy_array(self.pusher.getRigidBody('pusher').getPosition())[:2]
            diff = np.linalg.norm(pusher_pos - goal_pos)
            if diff < 0.0001:
                break
        self._set_action([0., 0.], rescale=False)
        self._step_callback()

        if not self.headless or self.observation_type in ("rgb", "depth", "rgb_and_depth"):
            self._render_callback()

        # Compute rewards
        reward = 0.
        self.timestep += 1

        info = dict()
        info['is_success'] = False
        done = info['is_success'] or self.timestep > self.max_episode_length

        obs = self._get_observation()

        return obs, reward, done, info
    
    def _set_action(self, action, rescale=True):
        info = dict()
        for end_effector in self.end_effectors:
            if end_effector.controllable:
                logger.debug("action: {}".format(action))
                info[end_effector.name] = end_effector.apply_control(self.sim, action, self.dt, rescale=rescale)

        return info

    def reset(self):
        logger.info("reset")
        self.timestep = 0
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim() 
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
        for buffer in self.render_to_image:
            name = buffer.getName()
            if name == 'rgb_buffer':
                rgb_buffer = buffer

        assert self.observation_type == 'rgb'

        image_ptr = rgb_buffer.getImageData()
        image_data = create_numpy_array(image_ptr, (self.image_size[0], self.image_size[1], 3), np.uint8)
        obs = np.flipud(image_data)

        return obs