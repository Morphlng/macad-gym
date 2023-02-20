"""
This file containts the Simulator class, which is used to directly interact with Simulator
__author__: Morphlng
"""
import os
import sys
import socket
import signal
import subprocess
import atexit
import GPUtil
import shutil
import logging
import random
import time
import math
import carla

from macad_gym.core.data.carla_data_provider import CarlaDataProvider
from macad_gym.core.data.sensor_interface import SensorDataProvider
from macad_gym.core.sensors.derived_sensors import LaneInvasionSensor
from macad_gym.core.sensors.derived_sensors import CollisionSensor
from macad_gym.core.data.timer import GameTime
from macad_gym import LOG_DIR


logger = logging.getLogger(__name__)

# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/software/CARLA_0.9.13/CarlaUE4.sh")
)

# Check if is using on Windows
IS_WINDOWS_PLATFORM = "win" in sys.platform

assert os.path.exists(SERVER_BINARY), (
    "Make sure CARLA_SERVER environment"
    " variable is set & is pointing to the"
    " CARLA server startup script (Carla"
    "UE4.sh). Refer to the README file/docs."
)


def get_tcp_port(port=0):
    """
    Get a free tcp port number
    :param port: (default 0) port number. When `0` it will be assigned a free port dynamically
    :return: a port number requested if free otherwise an unhandled exception would be thrown
    """
    s = socket.socket()
    s.bind(("", port))
    server_port = s.getsockname()[1]
    s.close()
    return server_port


class Weather:
    """Weather presets for Simulator"""

    CARLA_PRESETS = {
        0: carla.WeatherParameters.ClearNoon,
        1: carla.WeatherParameters.CloudyNoon,
        2: carla.WeatherParameters.WetNoon,
        3: carla.WeatherParameters.WetCloudyNoon,
        4: carla.WeatherParameters.MidRainyNoon,
        5: carla.WeatherParameters.HardRainNoon,
        6: carla.WeatherParameters.SoftRainNoon,
        7: carla.WeatherParameters.ClearSunset,
        8: carla.WeatherParameters.CloudySunset,
        9: carla.WeatherParameters.WetSunset,
        10: carla.WeatherParameters.WetCloudySunset,
        11: carla.WeatherParameters.MidRainSunset,
        12: carla.WeatherParameters.HardRainSunset,
        13: carla.WeatherParameters.SoftRainSunset,
    }


class Simulator:
    """Simulator class for interacting with CARLA simulator

    This class will establish a connection with CARLA simulator and provide a set of APIs for
    interacting with the simulator. It also provides a set of APIs for interacting with the
    sensors attached to the ego vehicle.

    The connection could either via carla.Client or a BridgeServer. The former is used for
    connecting to a simulator running on the same machine. The latter is used for connecting
    to a simulator running on a remote machine.

    Note:
        There are two kinds of id used in this class:
        1. actor_id: the id which is speicified by user in the config file
        2. id: the id which is assigned by CARLA simulator
        You should judge by the name and the argument type to determine which id is used.
    """

    def __init__(self, env):
        self._client = None
        self._process = None
        self._data_provider = CarlaDataProvider()
        self._sensor_provider = SensorDataProvider()
        self._game_time = GameTime()

        self.init_server(env)

        # handle termination
        def termination_cleanup(*_):
            self.clear_server_state()
            sys.exit(0)

        signal.signal(signal.SIGTERM, termination_cleanup)
        signal.signal(signal.SIGINT, termination_cleanup)
        atexit.register(self.clear_server_state)

    def init_server(self, env):
        """Create the server based on MultiCarlaEnv's config

        Args:
            env (MultiCarlaEnv): MultiCarlaEnv instance
        """

        # Create server if not already specified
        if env._server_port is None:
            env._server_port = get_tcp_port()

            multigpu_success = False
            gpus = GPUtil.getGPUs()
            log_file = os.path.join(LOG_DIR, "server_" +
                                    str(env._server_port) + ".log")
            logger.info(
                f"1. Port: {env._server_port}\n"
                f"2. Map: {env._server_map}\n"
                f"3. Binary: {SERVER_BINARY}"
            )

            if not env._render and (gpus is not None and len(gpus)) > 0:
                try:
                    min_index = random.randint(0, len(gpus) - 1)
                    for i, gpu in enumerate(gpus):
                        if gpu.load < gpus[min_index].load:
                            min_index = i
                    # Check if vglrun is setup to launch sim on multipl GPUs
                    if shutil.which("vglrun") is not None:
                        self._process = subprocess.Popen(
                            (
                                "DISPLAY=:8 vglrun -d :7.{} {} -benchmark -fps=20"
                                " -carla-server -world-port={}"
                                " -carla-streaming-port=0".format(
                                    min_index,
                                    SERVER_BINARY,
                                    env._server_port,
                                )
                            ),
                            shell=True,
                            # for Linux
                            preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                            # for Windows (not necessary)
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                            if IS_WINDOWS_PLATFORM
                            else 0,
                            stdout=open(log_file, "w"),
                        )

                    # Else, run in headless mode
                    else:
                        # Since carla 0.9.12+ use -RenderOffScreen to start headlessly
                        # https://carla.readthedocs.io/en/latest/adv_rendering_options/
                        self._process = subprocess.Popen(
                            (  # 'SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE={} DISPLAY='
                                '"{}" -RenderOffScreen -benchmark -fps=20 -carla-server'
                                " -world-port={} -carla-streaming-port=0".format(
                                    SERVER_BINARY,
                                    env._server_port,
                                )
                            ),
                            shell=True,
                            # for Linux
                            preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                            # for Windows (not necessary)
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                            if IS_WINDOWS_PLATFORM
                            else 0,
                            stdout=open(log_file, "w"),
                        )
                # TODO: Make the try-except style handling work with Popen
                # exceptions after launching the server procs are not caught
                except Exception as e:
                    print(e)
                # Temporary soln to check if CARLA server proc started and wrote
                # something to stdout which is the usual case during startup
                if os.path.isfile(log_file):
                    multigpu_success = True
                else:
                    multigpu_success = False

                if multigpu_success:
                    print("Running sim servers in headless/multi-GPU mode")

            # Rendering mode and also a fallback if headless/multi-GPU doesn't work
            if multigpu_success is False:
                try:
                    print("Using single gpu to initialize carla server")

                    self._process = subprocess.Popen(
                        [
                            SERVER_BINARY,
                            "-windowed",
                            "-ResX=",
                            str(env._env_config["render_x_res"]),
                            "-ResY=",
                            str(env._env_config["render_y_res"]),
                            "-benchmark",
                            "-fps=20",
                            "-carla-server",
                            "-carla-rpc-port={}".format(env._server_port),
                            "-carla-streaming-port=0",
                        ],
                        # for Linux
                        preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                        # for Windows (not necessary)
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                        if IS_WINDOWS_PLATFORM
                        else 0,
                        stdout=open(log_file, "w"),
                    )
                    print("Running simulation in single-GPU mode")
                except Exception as e:
                    logger.debug(e)
                    print("FATAL ERROR while launching server:",
                          sys.exc_info()[0])

        # Start client
        while self._client is None:
            try:
                self._client = carla.Client("localhost", env._server_port)
                # The socket establishment could takes some time
                time.sleep(1)
                self._client.set_timeout(2.0)
                print(
                    "Client successfully connected to server, Carla-Server version: ",
                    self._client.get_server_version(),
                )
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    print("Could not connect to Carla server because:", re)
                self._client = None

        self._client.set_timeout(60.0)
        self._client.load_world(env._server_map)
        world = self._client.get_world()
        world_settings = world.get_settings()
        world_settings.synchronous_mode = env._sync_server
        if env._sync_server:
            # Synchronous mode
            # Available with CARLA version>=0.9.6
            # Set fixed_delta_seconds to have reliable physics between sim steps
            world_settings.fixed_delta_seconds = env._fixed_delta_seconds
        world.apply_settings(world_settings)

        # Set up traffic manager
        traffic_manager = self._client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_respawn_dormant_vehicles(True)
        traffic_manager.set_synchronous_mode(env._sync_server)

        # Prepare data provider
        self._data_provider.set_client(self._client)
        self._data_provider.set_world(world)
        self._data_provider.set_traffic_manager_port(
            traffic_manager.get_port()
        )

        # Set GameTime callback (Should be manually removed)
        self.add_callback(lambda snapshot: self._game_time.on_carla_tick(snapshot.timestamp))

        # Set the spectator/server view if rendering is enabled
        if env._render and env._env_config.get("spectator_loc"):
            spectator = world.get_spectator()
            spectator_loc = carla.Location(*env._env_config["spectator_loc"])
            d = 6.4
            angle = 160  # degrees
            a = math.radians(angle)
            location = (
                carla.Location(d * math.cos(a), d *
                               math.sin(a), 2.0) + spectator_loc
            )
            spectator.set_transform(
                carla.Transform(location, carla.Rotation(
                    yaw=180 + angle, pitch=-15))
            )

    def get_world(self):
        """Get the world.

        Returns:
            carla.World: The world.
        """
        return self._data_provider.get_world()

    def get_traffic_manager(self):
        """Get the traffic manager.

        Returns:
            carla.TrafficManager: The traffic manager.
        """
        return self._client.get_trafficmanager()

    def get_traffic_manager_port(self):
        """Get the traffic manager port.

        Returns:
            int: The traffic manager port.
        """
        return self._data_provider.get_traffic_manager_port()

    def get_actor_by_id(self, id):
        """Get an actor by id.

        Args:
            id (int): Actor id.

        Returns:
            carla.Actor: The actor.
        """
        return self._data_provider.get_actor_by_id(id)

    def get_actor_location(self, id, decompose=False):
        """Get an actor's location.

        Args:
            id (int): Actor id.
            decompose (bool): If True, return a list of x, y, z coordinates.

        Returns:
            carla.Location: The actor's location.
            or list: The actor's location decomposed into x, y, z coordinates.
        """
        actor = self.get_actor_by_id(id)
        carla_loc = self._data_provider.get_location(actor)
        if decompose:
            return [carla_loc.x, carla_loc.y, carla_loc.z]

        return carla_loc

    def get_actor_velocity(self, id):
        """Get an actor's velocity.

        Args:
            id (int): Actor id.

        Returns:
            carla.Location: The actor's velocity.
        """
        actor = self.get_actor_by_id(id)
        return self._data_provider.get_velocity(actor)

    def get_actor_transform(self, id, decompose=False):
        """Get an actor's transform.

        Args:
            id (int): Actor id.
            decompose (bool): If True, return a list of x, y, z coordinates and pitch, yaw, roll.

        Returns:
            carla.Transform: The actor's transform.
            or Tuple[list, list]: The actor's transform decomposed into x, y, z coordinates and pitch, yaw, roll.
        """
        actor = self.get_actor_by_id(id)
        carla_transform = self._data_provider.get_transform(actor)
        if decompose:
            return (
                [carla_transform.location.x,
                 carla_transform.location.y,
                 carla_transform.location.z],
                [carla_transform.rotation.pitch,
                 carla_transform.rotation.yaw,
                 carla_transform.rotation.roll],
            )

        return carla_transform

    def get_actor_camera_data(self, actor_id):
        """Get an actor's camera data.

        Args:
            actor_id (str): Actor id.

        Returns:
            Dict: image data from sensor_interface.get_data(). E.g.

            data = {
                "sensor_id": (raw_data : carla.Image, processed_data : ndarray),
                ...
            }
        """
        return self._sensor_provider.get_camera_data(actor_id)

    def get_actor_collision_sensor(self, actor_id):
        """Get an actor's collision sensor.

        Args:
            actor_id (str): Actor id.

        Returns:
            CollisionSensor: The collision sensor.
        """
        coll_sensor = self._sensor_provider.get_collision_sensor(actor_id)
        return coll_sensor

    def get_actor_lane_invasion_sensor(self, actor_id):
        """Get an actor's lane invasion sensor.

        Args:
            actor_id (str): Actor id.

        Returns:
            LaneInvasionSensor: The lane invasion sensor.
        """
        lane_sensor = self._sensor_provider.get_lane_invasion_sensor(actor_id)
        return lane_sensor

    def set_weather(self, index):
        """Set the weather.

        Args:
            index (int | list): The index of the weather.

        Returns:
            weather specs (list): The weather specs (cloudiness, precipitation, precipitation_deposits, wind_intensity)
        """
        if isinstance(index, list):
            index = random.choice(index)

        try:
            weather = Weather.CARLA_PRESETS[index]
        except KeyError as e:
            print("Weather preset {} not found, using default 0".format(e))
            weather = Weather.CARLA_PRESETS[0]

        self.get_world().set_weather(weather)

        return [
            weather.cloudiness,
            weather.precipitation,
            weather.precipitation_deposits,
            weather.wind_intensity,
        ]

    def request_new_actor(self, *args, **kwargs):
        """Request a new actor.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            carla.Actor: The actor.
        """
        actor = self._data_provider.request_new_actor(*args, **kwargs)
        return actor

    def register_collision_sensor(self, actor_id, actor):
        """Register a collision sensor.

        Args:
            actor_id (str): The actor id.
            actor (carla.Actor): The actor which the sensor is attached to.
        """
        self._sensor_provider.update_collision_sensor(
            actor_id, CollisionSensor(actor))

    def register_lane_invasion_sensor(self, actor_id, actor):
        """Register a lane invasion sensor.

        Args:
            actor_id (str): The actor id.
            actor (carla.Actor): The actor which the sensor is attached to.
        """
        self._sensor_provider.update_lane_invasion_sensor(
            actor_id, LaneInvasionSensor(actor))

    def apply_actor_control(self, id, control):
        """Apply control to an actor.

        Args:
            id (int): Actor id.
            control (carla.VehicleControl): The control to apply.
        """
        actor = self.get_actor_by_id(id)
        actor.apply_control(control)

    def toggle_actor_autopilot(self, id, autopilot):
        """Toggle an actor's autopilot.

        Args:
            id (int): Actor id.
            autopilot (bool): Whether to enable autopilot.
        """
        actor = self.get_actor_by_id(id)
        if hasattr(actor, "set_autopilot"):
            actor.set_autopilot(autopilot, self.get_traffic_manager_port())
        else:
            logger.warning("Trying to toggle autopilot on a non-vehicle actor")

    def apply_traffic(self, num_vehicles, num_pedestrians, percentagePedestriansRunning=0.0, percentagePedestriansCrossing=0.0, safe=False):
        """Generate traffic.

        Args:
            num_vehicles (int): Number of vehicles to spawn.
            num_pedestrians (int): Number of pedestrians to spawn.
            percentagePedestriansRunning (float): Percentage of pedestrians running.
            percentagePedestriansCrossing (float): Percentage of pedestrians crossing.
            safe (bool): Whether to spawn vehicles in safe mode.

        Returns:
            list: List of spawned vehicles.
            tuple: Tuple of spawned pedestrians and their controllers.
        """
        # --------------
        # Spawn vehicles
        # --------------
        world = self.get_world()
        traffic_manager = self.get_traffic_manager()

        spawn_points = self._data_provider._spawn_points
        number_of_spawn_points = len(spawn_points)

        random.shuffle(spawn_points)
        if num_vehicles <= number_of_spawn_points:
            spawn_points = random.sample(spawn_points, num_vehicles)
        else:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logger.warning(msg, num_vehicles, number_of_spawn_points)
            num_vehicles = number_of_spawn_points

        vehicles_list = []
        failed_v = 0
        for n, transform in enumerate(spawn_points):
            # spawn the cars and set their autopilot and light state all together
            vehicle = self._data_provider.request_new_actor(
                "vehicle", transform, rolename="autopilot", autopilot=True, safe_blueprint=safe)
            if vehicle is not None:
                vehicles_list.append(vehicle)
            else:
                failed_v += 1

        logger.info(
            "{}/{} vehicles correctly spawned.".format(num_vehicles-failed_v, num_vehicles))

        # -------------
        # Spawn Walkers
        # -------------
        blueprints = self._data_provider._blueprint_library.filter(
            "walker.pedestrian.*")
        pedestrian_controller_bp = self._data_provider._blueprint_library.find(
            "controller.ai.walker")

        # Take all the random locations to spawn
        spawn_points = []
        for i in range(num_pedestrians):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # Spawn the walker object
        pedestrians_list = []
        controllers_list = []
        pedestrians_speed = []
        failed_p = 0
        for spawn_point in spawn_points:
            pedestrian_bp = random.choice(blueprints)
            # set as not invincible
            if pedestrian_bp.has_attribute('is_invincible'):
                pedestrian_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if pedestrian_bp.has_attribute('speed'):
                if random.random() > percentagePedestriansRunning:
                    speed = pedestrian_bp.get_attribute(
                        'speed').recommended_values[1]  # walking
                else:
                    speed = pedestrian_bp.get_attribute(
                        'speed').recommended_values[2]  # running
            else:
                speed = 0.0
            pedestrian = self._data_provider.request_new_actor(
                "walker.pedestrian", spawn_point, actor_category="pedestrian", blueprint=pedestrian_bp)
            if pedestrian is not None:
                controller = self._data_provider.request_new_actor(
                    "controller.ai.walker", carla.Transform(), attach_to=pedestrian, blueprint=pedestrian_controller_bp)
                if controller is not None:
                    pedestrians_list.append(pedestrian)
                    controllers_list.append(controller)
                    pedestrians_speed.append(speed)
                else:
                    self._data_provider.remove_actor_by_id(pedestrian.id)
                    failed_p += 1
            else:
                failed_p += 1

        logger.info(
            "{}/{} pedestrians correctly spawned.".format(num_pedestrians-failed_p, num_pedestrians))
        self.tick()

        # Initialize each controller and set target to walk
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i, controller in enumerate(controllers_list):
            controller.start()  # start walker
            # set walk to random point
            controller.go_to_location(
                world.get_random_location_from_navigation())
            controller.set_max_speed(
                float(pedestrians_speed[int(i / 2)]))  # max speed

        traffic_manager.global_percentage_speed_difference(30.0)

        return vehicles_list, (pedestrians_list, controllers_list)

    def tick(self):
        """Tick the simulator."""
        if self._data_provider.is_sync_mode():
            self.get_world().tick()
        else:
            self.get_world().wait_for_tick()

        self._data_provider.on_carla_tick()

    def add_callback(self, func):
        """Add a callback to the simulator.

        Args:
            func (callable): A function to be called on every tick. E.g. 

            def func(timestamp):
                print(timestamp)

        Returns:
            id (int) : The id of the callback.
        """
        return self.get_world().on_tick(func)

    def remove_callback(self, id):
        """Remove a callback from the simulator.

        Args:
            id (int): The id of the callback.
        """
        self.get_world().remove_on_tick(id)

    def cleanup(self):
        self._sensor_provider.cleanup()
        self._data_provider.cleanup()
        self._game_time.restart()

    def clear_server_state(self):
        """Clear server process"""
        print("Clearing Carla server state")
        self._sensor_provider.cleanup()
        self._data_provider.cleanup(completely=True)
        self._game_time.restart()

        try:
            if self._client:
                self._client = None
        except Exception as e:
            print("Error disconnecting client: {}".format(e))

        if self._process:
            print("Killing live carla process:", self._process)
            if IS_WINDOWS_PLATFORM:
                # for Windows
                subprocess.call(
                    ["taskkill", "/F", "/T", "/PID", str(self._process.pid)])
            else:
                # for Linux
                pgid = os.getpgid(self._process.pid)
                os.killpg(pgid, signal.SIGKILL)

            self._process = None

    def generate_spawn_point(self, pos, rot=None):
        """Generate a spawn point.

        Args:
            pos (list|tuple): The position of the spawn point in (x, y, z, yaw=0)
            rot (list|tuple): The rotation of the spawn point in (pitch, yaw, roll)

        Returns:
            spawn_point (carla.Transform): The spawn point.
        """
        loc = carla.Location(
            pos[0], pos[1], pos[2]
        )

        if rot is None:
            rot = (
                self._data_provider.get_map()
                .get_waypoint(loc, project_to_road=True)
                .transform.rotation
            )
        else:
            rot = carla.Rotation(*rot)

        if len(pos) > 3:
            rot.yaw = pos[3]

        return carla.Transform(loc, rot)
