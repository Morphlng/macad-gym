"""
multi_env.py: Multi-actor environment interface for CARLA-Gym
Should support two modes of operation. See CARLA-Gym developer guide for
more information
__author__: @Praveen-Palanisamy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from datetime import datetime
import logging
import json
import os
import random
import sys
import time
import traceback
import math
import numpy as np  # linalg.norm is used

from gym.spaces import Box, Discrete, Tuple, Dict
from macad_gym.multi_actor_env import MultiActorEnv
from macad_gym.core.sensors.utils import preprocess_image
from macad_gym.core.maps.nodeid_coord_map import MAP_TO_COORDS_MAPPING

from macad_gym.core.data.simulator import Simulator, CARLA_OUT_PATH
from macad_gym.core.agents import AgentWrapper, HumanAgent, RLAgent

from macad_gym.carla.reward import Reward
from macad_gym.carla.scenarios import Scenarios
from macad_gym.viz.render import Render

# The following imports depend on these paths being in sys path
sys.path.append(os.path.join(os.environ.get("CARLA_ROOT", "~/software/Carla0.9.13"), "PythonAPI/carla"))
from macad_gym.core.maps.nav_utils import PathTracker  # noqa: E402
from agents.navigation.local_planner import RoadOption  # noqa: E402

logger = logging.getLogger(__name__)

# TODO: Clean env & actor configs to have appropriate keys based on the nature
# of env
DEFAULT_MULTIENV_CONFIG = {
    "scenarios": "DEFAULT_SCENARIO_TOWN1",
    "env": {
        # Since Carla 0.9.6, you have to use `client.load_world(server_map)`
        # instead of passing the map name as an argument
        "server_map": "/Game/Carla/Maps/Town01",
        "render": True,
        "render_x_res": 800,
        "render_y_res": 600,
        "x_res": 84,
        "y_res": 84,
        "framestack": 1,
        "discrete_actions": True,
        "squash_action_logits": False,
        "verbose": False,
        "use_depth_camera": False,
        "send_measurements": False,
        "enable_planner": True,
        "sync_server": True,
        "fixed_delta_seconds": 0.05,
    },
    "actors": {
        "vehicle1": {
            "enable_planner": True,
            "render": True,  # Whether to render to screen or send to VFB
            "framestack": 1,  # note: only [1, 2] currently supported
            "convert_images_to_video": False,
            "early_terminate_on_collision": True,
            "verbose": False,
            "reward_function": "corl2017",
            "x_res": 84,
            "y_res": 84,
            "use_depth_camera": False,
            "squash_action_logits": False,
            "manual_control": False,
            "auto_control": False,
            "camera_type": "rgb",
            "camera_position": 0,
            "collision_sensor": "on",  # off
            "lane_sensor": "on",  # off
            "server_process": False,
            "send_measurements": False,
            "log_images": False,
            "log_measurements": False,
        }
    },
}

# Carla planner commands
COMMANDS_ENUM = {
    0.0: "REACH_GOAL",
    5.0: "GO_STRAIGHT",
    4.0: "TURN_RIGHT",
    3.0: "TURN_LEFT",
    2.0: "LANE_FOLLOW",
}

# Mapping from string repr to one-hot encoding index to feed to the model
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
    "LANE_FOLLOW": 4,
}

ROAD_OPTION_TO_COMMANDS_MAPPING = {
    RoadOption.VOID: "REACH_GOAL",
    RoadOption.STRAIGHT: "GO_STRAIGHT",
    RoadOption.RIGHT: "TURN_RIGHT",
    RoadOption.LEFT: "TURN_LEFT",
    RoadOption.LANEFOLLOW: "LANE_FOLLOW",
}

# Threshold to determine that the goal has been reached based on distance
DISTANCE_TO_GOAL_THRESHOLD = 0.5

# Threshold to determine that the goal has been reached based on orientation
ORIENTATION_TO_GOAL_THRESHOLD = math.pi / 4.0

# Number of retries if the server doesn't respond
RETRIES_ON_ERROR = 2

# Dummy Z coordinate to use when we only care about (x, y)
GROUND_Z = 22

DISCRETE_ACTIONS = {
    # coast
    0: [0.0, 0.0],
    # turn left
    1: [0.0, -0.5],
    # turn right
    2: [0.0, 0.5],
    # forward
    3: [1.0, 0.0],
    # brake
    4: [-0.5, 0.0],
    # forward left
    5: [0.5, -0.05],
    # forward right
    6: [0.5, 0.05],
    # brake left
    7: [-0.5, -0.5],
    # brake right
    8: [-0.5, 0.5],
}

MultiAgentEnvBases = [MultiActorEnv]
try:
    from ray.rllib.env import MultiAgentEnv

    MultiAgentEnvBases.append(MultiAgentEnv)
except ImportError:
    logger.warning("\n Disabling RLlib support.", exc_info=True)


class MultiCarlaEnv(*MultiAgentEnvBases):
    def __init__(self, configs=None):
        """MACAD-Gym environment implementation.

        Provides a generic MACAD-Gym environment implementation that can be
        customized further to create new or variations of existing
        multi-agent learning environments. The environment settings, scenarios
        and the actors in the environment can all be configured using
        the `configs` dict.

        Args:
            configs (dict): Configuration for environment specified under the
                `env` key and configurations for each actor specified as dict
                under `actor`.
                Example:
                    >>> configs = {"env":{
                    "server_map":"/Game/Carla/Maps/Town05",
                    "discrete_actions":True,...},
                    "actor":{
                    "actor_id1":{"enable_planner":True,...},
                    "actor_id2":{"enable_planner":False,...}
                    }}
        """

        if configs is None:
            configs = DEFAULT_MULTIENV_CONFIG
        # Functionalities classes
        self._reward_policy = Reward()
        configs["scenarios"] = Scenarios.resolve_scenarios_parameter(
            configs["scenarios"]
        )

        self._scenario_config = configs["scenarios"]
        self._env_config = configs["env"]
        self._actor_configs = configs["actors"]

        # At most one actor can be manual controlled
        manual_control_count = 0
        for _, actor_config in self._actor_configs.items():
            if actor_config["manual_control"]:
                if "vehicle" not in actor_config["type"]:
                    raise ValueError("Only vehicles can be manual controlled.")

                manual_control_count += 1

        assert manual_control_count <= 1, (
            "At most one actor can be manually controlled. "
            f"Found {manual_control_count} actors with manual_control=True"
        )

        # Camera position is problematic for certain vehicles and even in
        # autopilot they are prone to error
        self.exclude_hard_vehicles = True
        # list of str: Supported values for `type` filed in `actor_configs`
        # for actors than can be actively controlled
        self._supported_active_actor_types = [
            "vehicle_4W",
            "vehicle_2W",
            "pedestrian",
            "traffic_light",
        ]
        # list of str: Supported values for `type` field in `actor_configs`
        # for actors that are passive. Example: A camera mounted on a pole
        self._supported_passive_actor_types = ["camera"]

        # Set attributes as in gym's specs
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {"render.modes": "human"}

        # Belongs to env_config. Optional parameters are retrieved with .get()
        self._server_port = self._env_config.get("server_port", None)
        self._render = self._env_config.get("render", False)
        self._framestack = self._env_config.get("framestack", 1)
        self._squash_action_logits = self._env_config.get("squash_action_logits", False)
        self._verbose = self._env_config.get("verbose", False)
        self._render_x_res = self._env_config.get("render_x_res", 800)
        self._render_y_res = self._env_config.get("render_y_res", 600)
        self._use_depth_camera = self._env_config.get("use_depth_camera", False)
        self._sync_server = self._env_config.get("sync_server", True)
        self._fixed_delta_seconds = self._env_config.get("fixed_delta_seconds", 0.05)

        # Belongs to env_config. Required parameters are retrieved directly (Exception if not found)
        self._server_map = self._env_config["server_map"]
        self._map = self._server_map.split("/")[-1]
        self._discrete_actions = self._env_config["discrete_actions"]
        self._x_res = self._env_config["x_res"]
        self._y_res = self._env_config["y_res"]

        # For manual_control
        self.human_agent = None

        # Render related
        Render.resize_screen(self._render_x_res, self._render_y_res)

        self._camera_poses, window_dim = Render.get_surface_poses(
            [self._x_res, self._y_res], self._actor_configs
        )

        if manual_control_count == 0:
            Render.resize_screen(window_dim[0], window_dim[1])
        else:
            self._manual_control_render_pose = (0, window_dim[1])
            Render.resize_screen(
                max(self._render_x_res, window_dim[0]),
                self._render_y_res + window_dim[1],
            )

        # Actions space
        if self._discrete_actions:
            self.action_space = Dict(
                {
                    actor_id: Discrete(len(DISCRETE_ACTIONS))
                    for actor_id in self._actor_configs.keys()
                }
            )
        else:
            self.action_space = Dict(
                {
                    actor_id: Box(-1.0, 1.0, shape=(2,))
                    for actor_id in self._actor_configs.keys()
                }
            )

        # Output space of images after preprocessing
        if self._use_depth_camera:
            self._image_space = Box(
                0.0, 255.0, shape=(self._y_res, self._x_res, 1 * self._framestack)
            )
        else:
            self._image_space = Box(
                -1.0, 1.0, shape=(self._y_res, self._x_res, 3 * self._framestack)
            )

        # TODO: The observation space should be actor specific
        # Observation space in output
        if self._env_config["send_measurements"]:
            self.observation_space = Dict(
                {
                    actor_id: Tuple(
                        [
                            self._image_space,  # image
                            Discrete(len(COMMANDS_ENUM)),  # next_command
                            Box(
                                -128.0, 128.0, shape=(2,)
                            ),  # forward_speed, dist to goal
                        ]
                    )
                    for actor_id in self._actor_configs.keys()
                }
            )
        else:
            self.observation_space = Dict(
                {actor_id: self._image_space for actor_id in self._actor_configs.keys()}
            )

        # Set appropriate node-id to coordinate mappings for Town01 or Town02.
        self.pos_coor_map = MAP_TO_COORDS_MAPPING[self._map]

        self._spec = lambda: None
        self._spec.id = "Carla-v0"
        self._simulator = None
        self._num_steps = {}
        self._total_reward = {}
        self._prev_measurement = {}
        self._prev_image = None
        self._episode_id_dict = {}
        self._measurements_file_dict = {}
        self._weather_spec = None
        self._start_pos = {}  # Start pose for each actor
        self._end_pos = {}  # End pose for each actor
        self._start_coord = {}  # Start coordinate for each actor
        self._end_coord = {}    # End coordinate for each actor
        self._last_obs = None
        self._obs_dict = {}
        self._done_dict = {}
        self._previous_actions = {}
        self._previous_rewards = {}
        self._last_reward = {}
        self._agents = {}  # Dictionary of macad_agents with actor_id as key
        self._actors = {}  # Dictionary of actor.id with actor_id as key
        # TODO: move PathTracker into MacadAgent
        self._path_trackers = {}  # Dictionary of sensors with actor_id as key
        self._scenario_map = {}  # Dictionary with current scenario map config

    def _clean_world(self):
        """Destroy all actors cleanly before exiting

        Returns:
            N/A
        """
        for actor_id, agent in self._agents.items():
            agent.cleanup()

        self._simulator.cleanup()

        self._actors = {}
        self._agents = {}
        self._path_trackers = {}
        self.human_agent = None

        print("Cleaned-up the world...")

    def reset(self):
        """Reset the carla world, call _init_server()

        Returns:
            N/A
        """
        # World reset and new scenario selection if multiple are available
        self._load_scenario(self._scenario_config)

        for retry in range(RETRIES_ON_ERROR):
            try:
                if not self._simulator:
                    self._simulator = Simulator(self)
                    self._reset(clean_world=False)
                else:
                    self._reset()
                break
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                print("reset(): Retry #: {}/{}".format(retry + 1, RETRIES_ON_ERROR))
                self._simulator = None
                raise e

        # Set appropriate initial values for all actors
        for actor_id, actor_config in self._actor_configs.items():
            if self._done_dict.get(actor_id, True):
                self._last_reward[actor_id] = None
                self._total_reward[actor_id] = None
                self._num_steps[actor_id] = 0

                py_measurement = self._read_observation(actor_id)
                self._prev_measurement[actor_id] = py_measurement

                # Get initial observation
                image = None
                while image is None:
                    try:
                        cameras_data = self._simulator.get_actor_camera_data(actor_id)
                        image = cameras_data[actor_config["camera_type"]][0]
                    except KeyError as e:
                        self._simulator.tick()

                obs = self._encode_obs(actor_id, image, py_measurement)
                self._obs_dict[actor_id] = obs
                # Actor correctly reset
                self._done_dict[actor_id] = False

        return self._obs_dict

    # ! Deprecated method
    def _on_render(self):
        """Render the pygame window.

        Args:

        Returns:
            N/A
        """
        pass

    def _spawn_new_actor(self, actor_id):
        """Spawn an agent as per the blueprint at the given pose

        Args:
            blueprint: Blueprint of the actor. Can be a Vehicle or Pedestrian
            pose: carla.Transform object with location and rotation

        Returns:
            An instance of a subclass of carla.Actor. carla.Vehicle in the case
            of a Vehicle agent.

        """
        actor_type = self._actor_configs[actor_id].get("type", "vehicle_4W")
        if actor_type not in self._supported_active_actor_types:
            print("Unsupported actor type:{}. Using vehicle_4W as the type")
            actor_type = "vehicle_4W"

        if actor_type == "traffic_light":
            # Traffic lights already exist in the world & can't be spawned.
            # Find closest traffic light actor in world.actor_list and return
            from macad_gym.core.controllers import traffic_lights

            transform = self._simulator.generate_spawn_point(self._start_pos[actor_id])
            self._actor_configs[actor_id]["start_transform"] = transform
            tls = traffic_lights.get_tls(self._simulator.get_world(), transform, sort=True)
            #: Return the key (carla.TrafficLight object) of closest match
            return tls[0][0]

        if actor_type == "pedestrian":
            model = "walker.pedestrian.*"
        elif actor_type == "vehicle_4W" or actor_type == "vehicle_2W":
            # TODO: We lost the ability to spawn 2W vehicles. Need to fix this
            model = "vehicle"

        transform = self._simulator.generate_spawn_point(self._start_pos[actor_id])
        self._actor_configs[actor_id]["start_transform"] = transform
        vehicle = None
        for retry in range(RETRIES_ON_ERROR):
            vehicle = self._simulator.request_new_actor(
                model, transform, rolename=actor_id,
                autopilot=self._actor_configs[actor_id].get("auto_control", False),
                safe_blueprint=self.exclude_hard_vehicles
            )

            if vehicle is not None and vehicle.get_location().z > 0.0:
                break

            print("spawn_actor: Retry#:{}/{}".format(retry + 1, RETRIES_ON_ERROR))

        if vehicle is None:
            raise RuntimeError("Failed to spawn actor: {}".format(actor_id))

        return vehicle

    def _reset(self, clean_world=True):
        """Reset the state of the actors.
        A "soft" reset is performed in which the existing actors are destroyed
        and the necessary actors are spawned into the environment without
        affecting other aspects of the environment.
        If the "soft" reset fails, a "hard" reset is performed in which
        the environment's entire state is destroyed and a fresh instance of
        the server is created from scratch. Note that the "hard" reset is
        expected to take more time. In both of the reset modes ,the state/
        pose and configuration of all actors (including the sensor actors) are
        (re)initialized as per the actor configuration.

        Returns:
            dict: Dictionaries of observations for actors.

        Raises:
            RuntimeError: If spawning an actor at its initial state as per its'
            configuration fails (eg.: Due to collision with an existing object
            on that spot). This Error will be handled by the caller
            `self.reset()` which will perform a "hard" reset by creating
            a new server instance
        """

        self._done_dict["__all__"] = False
        if clean_world:
            self._clean_world()

        self._weather_spec = self._simulator.set_weather(self._scenario_map.get("weather_distribution", 0))

        for actor_id, actor_config in self._actor_configs.items():
            if self._done_dict.get(actor_id, True):
                self._measurements_file_dict[actor_id] = None
                self._episode_id_dict[actor_id] = datetime.today().strftime(
                    "%Y-%m-%d_%H-%M-%S_%f"
                )

                # Try to spawn actor (soft reset) or fail and reinitialize the server before get back here
                try:
                    actor_spawned = self._spawn_new_actor(actor_id)
                    self._actors[actor_id] = actor_spawned.id
                except RuntimeError as spawn_err:
                    del self._done_dict[actor_id]
                    # Chain the exception & re-raise to be handled by the caller `self.reset()`
                    raise spawn_err from RuntimeError(
                        "Unable to spawn actor:{}".format(actor_id)
                    )

                # We'll use this in SensorDataProvider
                # The "actor_id" is user defined and is used to identify the actor
                # The "id" is the carla actor id and is used to identify the actor in the carla world
                # The "simulator" is used to register callback
                actor_config.update({"actor_id": actor_id, "id": self._actors[actor_id], "simulator": self._simulator})

                if self._env_config["enable_planner"]:
                    self._path_trackers[actor_id] = PathTracker(
                        (
                            self._start_pos[actor_id][0],
                            self._start_pos[actor_id][1],
                            self._start_pos[actor_id][2],
                        ),
                        (
                            self._end_pos[actor_id][0],
                            self._end_pos[actor_id][1],
                            self._end_pos[actor_id][2],
                        ),
                        actor_spawned,
                    )

                # Spawn collision and lane sensors if necessary
                if actor_config["collision_sensor"] == "on":
                    self._simulator.register_collision_sensor(actor_id, actor_spawned)
                if actor_config["lane_sensor"] == "on":
                    self._simulator.register_lane_invasion_sensor(actor_id, actor_spawned)

                if not actor_config["manual_control"]:
                    agent = AgentWrapper(RLAgent(actor_config))
                    # Spawn cameras
                    agent.setup_sensors(actor_spawned)
                else:
                    actor_config.update({
                        "render_config": {
                            "width": self._env_config["render_x_res"],
                            "height": self._env_config["render_y_res"],
                            "render_pos": self._manual_control_render_pose
                        }
                    })
                    agent = AgentWrapper(HumanAgent(actor_config))
                    agent.setup_sensors(actor_spawned)
                    self.human_agent = agent    # quick access to human agent

                self._agents[actor_id] = agent

                self._start_coord.update(
                    {
                        actor_id: [
                            self._start_pos[actor_id][0] // 100,
                            self._start_pos[actor_id][1] // 100,
                        ]
                    }
                )
                self._end_coord.update(
                    {
                        actor_id: [
                            self._end_pos[actor_id][0] // 100,
                            self._end_pos[actor_id][1] // 100,
                        ]
                    }
                )

                print(
                    "Actor: {} start_pos_xyz(coordID): {} ({}), "
                    "end_pos_xyz(coordID) {} ({})".format(
                        actor_id,
                        self._start_pos[actor_id],
                        self._start_coord[actor_id],
                        self._end_pos[actor_id],
                        self._end_coord[actor_id],
                    )
                )

        print("New episode initialized with actors:{}".format(self._actors.keys()))

        self._simulator.apply_traffic(
            self._scenario_map.get("num_vehicles", 0),
            self._scenario_map.get("num_pedestrians", 0),
            safe = self.exclude_hard_vehicles
        )


    def _load_scenario(self, scenario_parameter):
        self._scenario_map = {}
        # If config contains a single scenario, then use it,
        # if it's an array of scenarios,randomly choose one and init
        if isinstance(scenario_parameter, dict):
            scenario = scenario_parameter
        else:  # instance array of dict
            scenario = random.choice(scenario_parameter)

        self._scenario_map = scenario
        for actor_id, actor in scenario["actors"].items():
            if isinstance(actor["start"], int):
                self._start_pos[actor_id] = self.pos_coor_map[str(
                    actor["start"])]
            else:
                self._start_pos[actor_id] = actor["start"]

            if isinstance(actor["end"], int):
                self._end_pos[actor_id] = self.pos_coor_map[str(actor["end"])]
            else:
                self._end_pos[actor_id] = actor["end"]

    def _decode_obs(self, actor_id, obs):
        """Decode actor observation into original image reversing the pre_process() operation.
        Args:
            actor_id (str): Actor identifier
            obs (dict): Properly encoded observation data of an actor

        Returns:
            image (array): Original actor camera view
        """
        if self._actor_configs[actor_id]["send_measurements"]:
            obs = obs[0]
        # Reverse the processing operation
        if self._actor_configs[actor_id]["use_depth_camera"]:
            img = np.tile(obs.swapaxes(0, 1), 3)
        else:
            img = obs.swapaxes(0, 1) * 128 + 128
        return img

    def _encode_obs(self, actor_id, image, py_measurements):
        """Encode sensor and measurements into obs based on state-space config.

        Args:
            actor_id (str): Actor identifier
            image (array): original unprocessed image
            py_measurements (dict): measurement file

        Returns:
            obs (dict): properly encoded observation data for each actor
        """
        assert self._framestack in [1, 2]
        # Apply preprocessing
        config = self._actor_configs[actor_id]
        image = preprocess_image(image, config)
        # Stack frames
        prev_image = self._prev_image
        self._prev_image = image
        if prev_image is None:
            prev_image = image
        if self._framestack == 2:
            image = np.concatenate([prev_image, image])
        # Structure the observation
        if not self._actor_configs[actor_id]["send_measurements"]:
            return image
        obs = (
            image,
            COMMAND_ORDINAL[py_measurements["next_command"]],
            [py_measurements["forward_speed"], py_measurements["distance_to_goal"]],
        )

        self._last_obs = obs
        return obs

    def step(self, action_dict):
        """Execute one environment step for the specified actors.

        Executes the provided action for the corresponding actors in the
        environment and returns the resulting environment observation, reward,
        done and info (measurements) for each of the actors. The step is
        performed asynchronously i.e. only for the specified actors and not
        necessarily for all actors in the environment.

        Args:
            action_dict (dict): Actions to be executed for each actor. Keys are
                agent_id strings, values are corresponding actions.

        Returns
            obs (dict): Observations for each actor.
            rewards (dict): Reward values for each actor. None for first step
            dones (dict): Done values for each actor. Special key "__all__" is
            set when all actors are done and the env terminates
            info (dict): Info for each actor.

        Raises
            RuntimeError: If `step(...)` is called before calling `reset()`
            ValueError: If `action_dict` is not a dictionary of actions
            ValueError: If `action_dict` contains actions for nonexistent actor
        """

        if self._simulator is None:
            raise RuntimeError("Cannot call step(...) before calling reset()")

        assert len(self._actors), (
            "No actors exist in the environment. Either"
            " the environment was not properly "
            "initialized using`reset()` or all the "
            "actors have exited. Cannot execute `step()`"
        )

        if not isinstance(action_dict, dict):
            raise ValueError(
                "`step(action_dict)` expected dict of actions. "
                "Got {}".format(type(action_dict))
            )
        # Make sure the action_dict contains actions only for actors that
        # exist in the environment
        if not set(action_dict).issubset(set(self._actors)):
            raise ValueError(
                "Cannot execute actions for non-existent actors."
                " Received unexpected actor ids:{}".format(
                    set(action_dict).difference(set(self._actors))
                )
            )

        try:
            obs_dict = {}
            reward_dict = {}
            info_dict = {}

            for actor_id, action in action_dict.items():
                obs, reward, done, info = self._step(actor_id, action)
                obs_dict[actor_id] = obs
                reward_dict[actor_id] = reward
                if not self._done_dict.get(actor_id, False):
                    self._done_dict[actor_id] = done
                info_dict[actor_id] = info
            self._done_dict["__all__"] = sum(self._done_dict.values()) >= len(
                self._actors
            )
            # Find if any actor's config has render=True & render only for
            # that actor. NOTE: with async server stepping, enabling rendering
            # affects the step time & therefore MAX_STEPS needs adjustments
            render_required = [
                k for k, v in self._actor_configs.items() if v.get("render", False)
            ]
            if render_required:
                images = {
                    k: self._decode_obs(k, v)
                    for k, v in obs_dict.items()
                    if self._actor_configs[k]["render"]
                }

                Render.multi_view_render(images, self._camera_poses)
                if self.human_agent is None:
                    Render.dummy_event_handler()

            return obs_dict, reward_dict, self._done_dict, info_dict
        except Exception:
            print(
                "Error during step, terminating episode early.", traceback.format_exc()
            )
            self._simulator.clear_server_state()

    def _step(self, actor_id, action):
        """Perform the actual step in the CARLA environment

        Applies control to `actor_id` based on `action`, process measurements,
        compute the rewards and terminal state info (dones).

        Args:
            actor_id(str): Actor identifier
            action: Actions to be executed for the actor.

        Returns
            obs (obs_space): Observation for the actor whose id is actor_id.
            reward (float): Reward for actor. None for first step
            done (bool): Done value for actor.
            info (dict): Info for actor.
        """

        if self._discrete_actions:
            action = DISCRETE_ACTIONS[int(action)]
        assert len(action) == 2, "Invalid action {}".format(action)
        if self._squash_action_logits:
            forward = 2 * float(sigmoid(action[0]) - 0.5)
            throttle = float(np.clip(forward, 0, 1))
            brake = float(np.abs(np.clip(forward, -1, 0)))
            steer = 2 * float(sigmoid(action[1]) - 0.5)
        else:
            throttle = float(np.clip(action[0], 0, 0.6))
            brake = float(np.abs(np.clip(action[0], -1, 0)))
            steer = float(np.clip(action[1], -1, 1))
        reverse = False
        hand_brake = False

        action_dict = {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "reverse": reverse,
            "hand_brake": hand_brake,
        }

        if self._verbose:
            print(action_dict)

        cur_id = self._actors[actor_id]
        if self.human_agent is not None:
            human_action = self.human_agent(action_dict)
            if self.human_agent._agent.use_autopilot:
                self._simulator.toggle_actor_autopilot(cur_id, True)
            else:
                self._simulator.toggle_actor_autopilot(cur_id, False)
                self._simulator.apply_actor_control(cur_id, human_action)

        config = self._actor_configs[actor_id]
        if config["enable_planner"]:
            # update planned route, this will affect _read_observation()
            path_tracker = self._path_trackers[actor_id]
            planned_action = path_tracker.run_step()

        if config["manual_control"]:
            self.human_agent._agent._hic._hud.tick(
                self._simulator.get_world(),
                self._simulator.get_actor_by_id(cur_id),
                self._simulator.get_actor_collision_sensor(actor_id),
                self.human_agent._agent._hic._clock,
            )
        elif config["auto_control"]:
            if config["enable_planner"]:
                if path_tracker.agent.done() or self._done_dict[actor_id]:
                    self._simulator.toggle_actor_autopilot(cur_id, True)
                else:
                    # Apply BasicAgent's action, this will navigate the actor to defined destination.
                    # However, the BasicAgent doesn't take consideration of some rules, such as stop sign.
                    # Thus it's not guaranteed to drive safely.
                    self._simulator.apply_actor_control(cur_id, planned_action)
                    # TODO: For debugging, remove drawing when PathTracker is complete
                    path_tracker.draw()
            else:
                self._simulator.toggle_actor_autopilot(cur_id, True)
        else:
            # Apply RL agent action
            self._simulator.apply_actor_control(cur_id, self._agents[actor_id](action_dict))

        self._simulator.tick()

        # Process observations
        py_measurements = self._read_observation(actor_id)
        if self._verbose:
            print("Next command", py_measurements["next_command"])
        # Store previous action
        self._previous_actions[actor_id] = action
        if type(action) is np.ndarray:
            py_measurements["action"] = [float(a) for a in action]
        else:
            py_measurements["action"] = action
        py_measurements["control"] = action_dict

        # Compute done
        done = (
            self._num_steps[actor_id] > self._scenario_map["max_steps"]
            or py_measurements["next_command"] == "REACH_GOAL"
            or (
                config["early_terminate_on_collision"]
                and collided_done(py_measurements)
            )
        )
        py_measurements["done"] = done

        # Compute reward
        config = self._actor_configs[actor_id]
        flag = config["reward_function"]
        reward = self._reward_policy.compute_reward(
            self._prev_measurement[actor_id], py_measurements, flag
        )

        self._previous_rewards[actor_id] = reward
        if self._total_reward[actor_id] is None:
            self._total_reward[actor_id] = reward
        else:
            self._total_reward[actor_id] += reward

        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self._total_reward[actor_id]

        # End iteration updating parameters and logging
        self._prev_measurement[actor_id] = py_measurements
        self._num_steps[actor_id] += 1

        if config["log_measurements"] and CARLA_OUT_PATH:
            # Write out measurements to file
            if not self._measurements_file_dict[actor_id]:
                self._measurements_file_dict[actor_id] = open(
                    os.path.join(
                        CARLA_OUT_PATH,
                        "measurements_{}.json".format(
                            self._episode_id_dict[actor_id]),
                    ),
                    "w",
                )
            self._measurements_file_dict[actor_id].write(
                json.dumps(py_measurements))
            self._measurements_file_dict[actor_id].write("\n")
            if done:
                self._measurements_file_dict[actor_id].close()
                self._measurements_file_dict[actor_id] = None

        # TODO: Only support one camera for each actor now
        for sensor_id, data in self._simulator.get_actor_camera_data(actor_id).items():
            original_image = data[0]
            return (
                self._encode_obs(actor_id, original_image, py_measurements),
                reward,
                done,
                py_measurements,
            )

    def _read_observation(self, actor_id):
        """Read observation and return measurement.

        Args:
            actor_id (str): Actor identifier

        Returns:
            dict: measurement data.

        """
        cur_id = self._actors[actor_id]
        cur_config = self._actor_configs[actor_id]
        planner_enabled = cur_config["enable_planner"]
        if planner_enabled:
            dist = self._path_trackers[actor_id].get_distance_to_end()
            orientation_diff = self._path_trackers[
                actor_id
            ].get_orientation_difference_to_end_in_radians()
            commands = self._path_trackers[actor_id].get_path()
            if len(commands) > 0:
                next_command = ROAD_OPTION_TO_COMMANDS_MAPPING.get(
                    commands[0], "LANE_FOLLOW"
                )
            elif (
                dist <= DISTANCE_TO_GOAL_THRESHOLD
                and orientation_diff <= ORIENTATION_TO_GOAL_THRESHOLD
            ):
                next_command = "REACH_GOAL"
            else:
                next_command = "LANE_FOLLOW"

            # DEBUG
            # self.path_trackers[actor_id].draw()
        else:
            next_command = "LANE_FOLLOW"

        # Sensor Data
        collision_vehicles = None
        collision_other = None
        collision_pedestrians = None
        if cur_config.get("collision_sensor", "off") == "on":
            collision_sensor = self._simulator.get_actor_collision_sensor(actor_id)
            collision_vehicles = collision_sensor.collision_vehicles
            collision_pedestrians = collision_sensor.collision_pedestrians
            collision_other = collision_sensor.collision_other

        intersection_otherlane = None
        intersection_offroad = None
        if cur_config.get("lane_sensor", "off") == "on":
            lane_sensor = self._simulator.get_actor_lane_invasion_sensor(actor_id)
            intersection_otherlane = lane_sensor.offlane
            intersection_offroad = lane_sensor.offroad

        (cur_x, cur_y, cur_z), (cur_pitch, cur_yaw, cur_roll) = self._simulator.get_actor_transform(cur_id, decompose=True)
        cur_velocity = self._simulator.get_actor_velocity(cur_id)

        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0
        elif planner_enabled:
            distance_to_goal = self._path_trackers[actor_id].get_distance_to_end()
        else:
            distance_to_goal = -1

        distance_to_goal_euclidean = float(
            np.linalg.norm(
                [
                    cur_x - self._end_pos[actor_id][0],
                    cur_y - self._end_pos[actor_id][1],
                ]
            )
        )

        py_measurements = {
            "episode_id": self._episode_id_dict[actor_id],
            "step": self._num_steps[actor_id],
            "x": cur_x,
            "y": cur_y,
            "pitch": cur_pitch,
            "yaw": cur_yaw,
            "roll": cur_roll,
            "forward_speed": cur_velocity,  # Used to be vx
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": collision_vehicles,
            "collision_pedestrians": collision_pedestrians,
            "collision_other": collision_other,
            "intersection_offroad": intersection_offroad,
            "intersection_otherlane": intersection_otherlane,
            "weather": self._weather_spec,
            "map": self._server_map,
            "start_coord": self._start_coord[actor_id],
            "end_coord": self._end_coord[actor_id],
            "current_scenario": self._scenario_map,
            "x_res": self._x_res,
            "y_res": self._y_res,
            "max_steps": self._scenario_map["max_steps"],
            "next_command": next_command,
            "previous_action": self._previous_actions.get(actor_id, None),
            "previous_reward": self._previous_rewards.get(actor_id, None),
        }

        return py_measurements

    def close(self):
        """Clean-up the world, clear server state & close the Env"""
        self._clean_world()
        self._simulator.clear_server_state()


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = "Vehicle at ({pos_x:.1f}, {pos_y:.1f}), "
    message += "{speed:.2f} km/h, "
    message += "Collision: {{vehicles={col_cars:.0f}, "
    message += "pedestrians={col_ped:.0f}, other={col_other:.0f}}}, "
    message += "{other_lane:.0f}% other lane, {offroad:.0f}% off-road, "
    message += "({agents_num:d} non-player macad_agents in the scene)"
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents,
    )
    print(message)


def sigmoid(x):
    x = float(x)
    return np.exp(x) / (1 + np.exp(x))


def collided_done(py_measurements):
    """Define the main episode termination criteria"""
    m = py_measurements
    collided = (
        m["collision_vehicles"] > 0
        or m["collision_pedestrians"] > 0
        or m["collision_other"] > 0
    )
    return bool(collided)  # or m["total_reward"] < -100)


def get_next_actions(measurements, is_discrete_actions):
    """Get/Update next action, work with way_point based planner.

    Args:
        measurements (dict): measurement data.
        is_discrete_actions (bool): whether use discrete actions

    Returns:
        dict: action_dict, dict of len-two integer lists.
    """
    action_dict = {}
    for actor_id, meas in measurements.items():
        m = meas
        command = m["next_command"]
        if command == "REACH_GOAL":
            action_dict[actor_id] = 0
        elif command == "GO_STRAIGHT":
            action_dict[actor_id] = 3
        elif command == "TURN_RIGHT":
            action_dict[actor_id] = 6
        elif command == "TURN_LEFT":
            action_dict[actor_id] = 5
        elif command == "LANE_FOLLOW":
            action_dict[actor_id] = 3
        # Test for discrete actions:
        if not is_discrete_actions:
            action_dict[actor_id] = [1, 0]
    return action_dict


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="CARLA Manual Control Client")
    argparser.add_argument("--scenario", default="3",
                           help="print debug information")
    # TODO: Fix the default path to the config.json;Should work after packaging
    argparser.add_argument(
        "--config",
        default="src/macad_gym/carla/config.json",
        help="print debug information",
    )

    argparser.add_argument("--map", default="Town01",
                           help="print debug information")

    args = argparser.parse_args()

    multi_env_config = json.load(open(args.config))
    env = MultiCarlaEnv(multi_env_config)

    for _ in range(2):
        obs = env.reset()

        total_reward_dict = {}
        action_dict = {}

        env_config = multi_env_config["env"]
        actor_configs = multi_env_config["actors"]
        for actor_id in actor_configs.keys():
            total_reward_dict[actor_id] = 0
            if env._discrete_actions:
                action_dict[actor_id] = 3  # Forward
            else:
                action_dict[actor_id] = [1, 0]  # test values

        start = time.time()
        i = 0
        done = {"__all__": False}
        while not done["__all__"]:
            # while i < 20:  # TEST
            i += 1
            obs, reward, done, info = env.step(action_dict)
            action_dict = get_next_actions(info, env._discrete_actions)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(
                ":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(
                    i, reward, total_reward_dict, done
                )
            )

        print("{} fps".format(i / (time.time() - start)))
