import os
import sys
import logging
import numpy as np
from gym import spaces

import agx
import agxSDK
import agxOSG
import agxRender

from gym_agx.sims.rope_obstacle2 import build_simulation
from gym_agx.envs import agx_env
from gym_agx.utils.agx_classes import CameraConfig
from gym_agx.utils.agx_utils import to_numpy_array

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'push_rope.agx')
GOAL_SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'IRRELEVANT')

logger = logging.getLogger('gym_agx.envs')


class RopeObstacle2Env(agx_env.AgxEnv):
    """Subclass which inherits from DLO environment."""

    def __init__(self, n_substeps=1, reward_type="dense", observation_type="gt", headless=False, **kwargs):
        self.headless = headless
        
        camera_distance = 0.21  # meters
        camera_config = CameraConfig(
            eye=agx.Vec3(0, 0, camera_distance),
            center=agx.Vec3(0, 0, 0),
            up=agx.Vec3(0., 0., 0.),
            light_position=agx.Vec4(0.1, - camera_distance, camera_distance, 1.),
            light_direction=agx.Vec3(0., 0., -1.)
        )

        # pusher = EndEffector(
        #     name='pusher',
        #     controllable=True,
        #     observable=True,
        #     max_velocity=5 / 100,  # m/s
        #     max_acceleration=10 / 100,  # m/s^2
        # )
        # pusher.add_constraint(name='pusher_joint_base_x',
        #                         end_effector_dof=EndEffectorConstraint.Dof.X_TRANSLATION,
        #                         compute_forces_enabled=False,
        #                         velocity_control=True,
        #                         compliance_control=False)
        # pusher.add_constraint(name='pusher_joint_base_y',
        #                         end_effector_dof=EndEffectorConstraint.Dof.Y_TRANSLATION,
        #                         compute_forces_enabled=False,
        #                         velocity_control=True,
        #                         compliance_control=False)
        # pusher.add_constraint(name='pusher_joint_base_z',
        #                         end_effector_dof=EndEffectorConstraint.Dof.Z_TRANSLATION,
        #                         compute_forces_enabled=False,
        #                         velocity_control=False,
        #                         compliance_control=False)
        # pushers = [pusher]

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
        

        super(RopeObstacle2Env, self).__init__(scene_path=SCENE_PATH,
                                              n_substeps=n_substeps,
                                              observation_type=observation_type,
                                              n_actions=4,
                                              camera_pose=camera_config.camera_pose,
                                              image_size=(64, 64),
                                              no_graphics=no_graphics,
                                              args=args)
        self.action_space = spaces.Dict({
            'idx': spaces.Discrete(len(self.rope.segments)),
            'delta': spaces.Box(-1., 1., shape=(2,), dtype='float32')
        })
    
    def _build_simulation(self):
        self.rope = build_simulation(self.sim)
        self.gravity = self.sim.getUniformGravity()
        self.time_step = self.sim.getTimeStep()
        logger.debug("Timestep after readFile is: {}".format(self.time_step))
        logger.debug("Gravity after readFile is: {}".format(self.gravity))

    def render(self, mode="human"):
        return super(RopeObstacle2Env, self).render(mode)

    def step(self, action):
        logger.info("step")

        info = self._apply_action(action)
        # info = self._set_action(action)
        # self._step_callback()

        # if not self.headless or self.observation_type in ("rgb", "depth", "rgb_and_depth"):
        #     self._render_callback()

        # Compute rewards
        reward = 0.

        info['is_success'] = False
        done = info['is_success']

        obs = self._get_observation()

        return obs, reward, done, info

    def _apply_action(self, action):
        info = dict()

        idx = action['idx']
        delta = action['delta'] * 0.01
        delta = np.append(delta, 0.)

        segment = self.rope.segments[idx]
        mass_props = segment.getRigidBody().getMassProperties()
        old_mass = mass_props.getMass()
        mass_props.setMass(old_mass * 100)
        
        end_pos = to_numpy_array(segment.getRigidBody().getPosition()) + delta
        
        force_magnitude = 0.05
        i = 0
        for _ in range(40):
            current_pos = to_numpy_array(segment.getRigidBody().getPosition())
            force = end_pos - current_pos
            force = force / np.linalg.norm(force) * force_magnitude

            segment.getRigidBody().addForce(*force)
            self._step_callback()

            diff = np.linalg.norm(to_numpy_array(segment.getRigidBody().getPosition()) - current_pos)
            if diff < 0.001:
                force_magnitude *= 1.1
            else:
                force_magnitude *= 0.95
            force_magnitude = min(force_magnitude, 0.5)
            
            if np.linalg.norm(to_numpy_array(segment.getRigidBody().getPosition()) - end_pos) < 0.001:
                break
            i += 1
        mass_props.setMass(old_mass)
        if not self.headless or self.observation_type in ("rgb", "depth", "rgb_and_depth"):
                self._render_callback()
                
        return info
        

    def reset(self):
        logger.info("reset")
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

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

        self.app.setAutoStepping(True)
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

        scene_decorator = self.app.getSceneDecorator()
        light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
        light_source_0.setPosition(light_pos)
        light_source_0.setDirection(light_dir)
        scene_decorator.setEnableLogo(False)

    def _get_observation(self):
        return np.zeros(2, dtype='float32')
    
    def _set_action(self, action):
        self.rope.segments[9].getRigidBody().addForce(-0.015, -0.0, 0.0)
        return dict()