import sys
import logging
import os
import numpy as np
from gym import spaces

import agx
import agxSDK
import agxCable
import agxOSG
import agxRender
from gym_agx.utils.utils import point_inside_polygon, all_points_below_z

from gym_agx.envs import agx_env
from gym_agx.rl.observation import get_cable_segment_positions, get_cable_segment_positions_and_velocities
from gym_agx.rl.end_effector import EndEffector, EndEffectorConstraint
from gym_agx.utils.agx_classes import CameraConfig
from gym_agx.utils.agx_utils import to_numpy_array
from gym_agx.sims.rope_obstacle import build_simulation
from agxPythonModules.utils.numpy_utils import create_numpy_array

logger = logging.getLogger('gym_agx.envs')

# Set paths
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.split(FILE_DIR)[0]
SCENE_PATH = os.path.join(PACKAGE_DIR, "assets/IRRELEVANT")

OBSTACLE_POSITIONS = [[0.0, 0.0], [0.075, 0.075], [-0.075, 0.075], [0.075, -0.075], [-0.075, -0.075]]

N_GRIPPERS = 2


class RopeObstacleEnv(agx_env.AgxEnv):
    """Cable closing environment."""

    def __init__(self, n_substeps=1, reward_type="dense", observation_type="gt", headless=False, **kwargs):
        """Initializes a CableClosingEnv object
        :param args: arguments for agxViewer.
        :param scene_path: path to binary file in assets/ folder containing serialized simulation defined in sim/ folder
        :param n_substeps: number os simulation steps per call to step().
        :param end_effectors: list of EndEffector objects, defining controllable constraints.
        :param observation_config: ObservationConfig object, defining the types of observations.
        :param camera_config: dictionary containing EYE, CENTER, UP information for rendering, with lighting info.
        :param reward_config: reward configuration object, defines success condition and reward function.
        """

        self.reward_type = reward_type
        self.segment_pos_old = None
        self.headless = headless

        camera_distance = 0.1  # meters
        camera_config = CameraConfig(
            eye=agx.Vec3(0, 0.0, 0.65),
            center=agx.Vec3(0, 0, 0.0),
            up=agx.Vec3(0., 0., 0.0),
            light_position=agx.Vec4(0, - camera_distance, camera_distance, 1.),
            light_direction=agx.Vec3(0., 0., -1.))

        self.end_effectors = []
        for i in range(N_GRIPPERS):
            name = f"gripper_{i}"
            gripper = EndEffector(
                name=name,
                controllable=True,
                observable=True,
                max_velocity=3,
                max_acceleration=3
            )
            gripper.add_constraint(name=f"{name}_joint_base_x",
                                   end_effector_dof=EndEffectorConstraint.Dof.X_TRANSLATION,
                                   compute_forces_enabled=False,
                                   velocity_control=True,
                                   compliance_control=False)
            gripper.add_constraint(name=f"{name}_joint_base_y",
                                   end_effector_dof=EndEffectorConstraint.Dof.Y_TRANSLATION,
                                   compute_forces_enabled=False,
                                   velocity_control=True,
                                   compliance_control=False)
            self.end_effectors.append(gripper)

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

        super(RopeObstacleEnv, self).__init__(scene_path=SCENE_PATH,
                                              n_substeps=n_substeps,
                                              observation_type=observation_type,
                                              n_actions=4,
                                              camera_pose=camera_config.camera_pose,
                                              image_size=(64, 64),
                                              no_graphics=no_graphics,
                                              args=args)
        self.action_space = spaces.Dict({
            'idx': spaces.Discrete(N_GRIPPERS),
            'velocity': spaces.Box(-1., 1., shape=(2,), dtype='float32')
        })
    
    def _build_simulation(self):
        build_simulation(self.sim, N_GRIPPERS)
        self.gravity = self.sim.getUniformGravity()
        self.time_step = self.sim.getTimeStep()
        logger.debug("Timestep after readFile is: {}".format(self.time_step))
        logger.debug("Gravity after readFile is: {}".format(self.gravity))

    def render(self, mode="human"):
        return super(RopeObstacleEnv, self).render(mode)

    def step(self, action):
        logger.info("step")

        action['velocity'] = np.clip(action['velocity'],
                                     self.action_space['velocity'].low,
                                     self.action_space['velocity'].high)
        action['velocity'] *= 0.5
        info = self._set_action(action)
        self._step_callback()

        if not self.headless or self.observation_type in ("rgb", "depth", "rgb_and_depth"):
            self._render_callback()

        # Get segments positions
        segment_pos = self._compute_segments_pos()

        # Compute rewards
        reward = 0.

        # Set old segment pos for next time step
        self.segment_pos_old = segment_pos

        info['is_success'] = False
        done = info['is_success']

        obs = self._get_observation()

        return obs, reward, done, info

    def reset(self):
        logger.info("reset")
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        n_inital_random = 20
        for k in range(n_inital_random):
            if k == 0 or not k % 5:
                action = self.action_space.sample()
            self._set_action(action)
            self.sim.stepForward()

        # Wait several steps after initalization
        n_inital_wait = 10
        for k in range(n_inital_wait):
            self.sim.stepForward()

        self.segment_pos_old = self._compute_segments_pos()

        obs = self._get_observation()
        return obs

    def _reset_sim(self):
        self.sim.cleanup(agxSDK.Simulation.CLEANUP_ALL, True)
        self._build_simulation()
        self._add_rendering(mode='osg')
        return True

    def _compute_segments_pos(self):
        segments_pos = []
        dlo = agxCable.Cable.find(self.sim, "DLO")
        segment_iterator = dlo.begin()
        n_segments = dlo.getNumSegments()
        for i in range(n_segments):
            if not segment_iterator.isEnd():
                pos = segment_iterator.getGeometry().getPosition()
                segments_pos.append(to_numpy_array(pos))
                segment_iterator.inc()

        return segments_pos

    def _add_rendering(self, mode='osg'):
        # Set renderer
        self.app.setAutoStepping(True)
        self.app.setEnableDebugRenderer(False)
        self.app.setEnableOSGRenderer(True)

        # Create scene graph for rendering
        root = self.app.getSceneRoot()
        rbs = self.sim.getRigidBodies()
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
                i = int(rb.getName().split('_')[1])
                if rb.getName() != f"gripper_{i}":
                    # cases such as "gripper_0_base_x"
                    agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
                    agxOSG.setAlpha(node, 0.0)
                else:
                    val = (i + 1) / N_GRIPPERS
                    agxOSG.setDiffuseColor(node, agxRender.Color(val, val, val, 1.0))
            elif "dlo" in rb.getName():  # Cable segments
                agxOSG.setDiffuseColor(node, agxRender.Color(0.1, 0.5, 0.0, 1.0))
                agxOSG.setAmbientColor(node, agxRender.Color(0.2, 0.5, 0.0, 1.0))
            elif rb.getName() == "obstacle":
                agxOSG.setDiffuseColor(node, agxRender.Color(0.5, 0.5, 0.5, 1.0))
            else:
                agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
                agxOSG.setAlpha(node, 0.0)

        # Set rendering options
        scene_decorator = self.app.getSceneDecorator()
        scene_decorator.setEnableLogo(False)
        scene_decorator.setBackgroundColor(agxRender.Color(1.0, 1.0, 1.0, 1.0))

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
            gripper_pos = [to_numpy_array(self.sim.getRigidBody(f"gripper_{i}").getPosition())[0:2]
                           for i in range(N_GRIPPERS)]
            obs = np.concatenate([*gripper_pos, seg_pos])
        elif self.observation_type == "pos_and_vel":
            seg_pos,seg_vel = get_cable_segment_positions_and_velocities(cable=agxCable.Cable.find(self.sim, "DLO"))
            seg_pos = seg_pos.flatten()
            seg_vel = seg_vel.flatten()

            gripper_obs = []
            for i in range(N_GRIPPERS):
                gripper = self.sim.getRigidBody(f"gripper_{i}")
                gripper_pos = to_numpy_array(gripper.getPosition())[0:2]
                gripper_vel = to_numpy_array(gripper.getVelocity())[0:2]
                gripper_obs.extend([gripper_pos, gripper_vel])
            obs = np.concatenate([*gripper_obs, seg_pos, seg_vel])
        return obs

    def _set_action(self, action):
        idx = action['idx']
        action = action['velocity']
        full_action = np.zeros(2 * N_GRIPPERS, dtype='float32')
        full_action[2*idx:2*(idx+1)] = action
 
        info = dict()
        logger.debug("action: {} {}".format(idx, action))
        info[self.end_effectors[idx].name] = self.end_effectors[idx].apply_control(self.sim, full_action, self.dt)
        return info
