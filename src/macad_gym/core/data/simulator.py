"""
This file containts the Simulator class, which is used to directly interact with Simulator
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

from macad_gym.core.data.carla_data_provider import CarlaDataProvider
from macad_gym.core.data.sensor_interface import SensorDataProvider
from macad_gym.core.data.timer import GameTime
from macad_gym import LOG_DIR

import carla

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

    _client = None
    _process = None

    @staticmethod
    def init_server(env):
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
                        Simulator._process = subprocess.Popen(
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
                        Simulator._process = subprocess.Popen(
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

                    Simulator._process = subprocess.Popen(
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
        while Simulator._client is None:
            try:
                Simulator._client = carla.Client("localhost", env._server_port)
                # The socket establishment could takes some time
                time.sleep(1)
                Simulator._client.set_timeout(2.0)
                print(
                    "Client successfully connected to server, Carla-Server version: ",
                    Simulator._client.get_server_version(),
                )
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    print("Could not connect to Carla server because:", re)
                Simulator._client = None

        Simulator._client.set_timeout(60.0)
        Simulator._client.load_world(env._server_map)
        world = Simulator._client.get_world()
        world_settings = world.get_settings()
        world_settings.synchronous_mode = env._sync_server
        if env._sync_server:
            # Synchronous mode
            # Available with CARLA version>=0.9.6
            # Set fixed_delta_seconds to have reliable physics between sim steps
            world_settings.fixed_delta_seconds = env._fixed_delta_seconds
        world.apply_settings(world_settings)

        # Set up traffic manager
        traffic_manager = Simulator._client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_respawn_dormant_vehicles(True)
        traffic_manager.set_synchronous_mode(env._sync_server)

        # Prepare data provider
        CarlaDataProvider.set_client(Simulator._client)
        CarlaDataProvider.set_world(world)
        CarlaDataProvider.set_traffic_manager_port(
            traffic_manager.get_port()
        )

        # Set GameTime
        Simulator.add_callback(lambda snapshot: GameTime.on_carla_tick(snapshot.timestamp))

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

    @staticmethod
    def get_world():
        """Get the world.

        Returns:
            carla.World: The world.
        """
        return CarlaDataProvider.get_world()

    @staticmethod
    def get_traffic_manager():
        """Get the traffic manager.

        Returns:
            carla.TrafficManager: The traffic manager.
        """
        return Simulator._client.get_trafficmanager()

    @staticmethod
    def tick():
        """Tick the simulator."""
        if CarlaDataProvider.is_sync_mode():
            CarlaDataProvider.get_world().tick()
        else:
            CarlaDataProvider.get_world().wait_for_tick()
        
        CarlaDataProvider.on_carla_tick()

    @staticmethod
    def add_callback(func):
        """Add a callback to the simulator.

        Args:
            func (callable): A function to be called on every tick. E.g. 
            
            def func(timestamp):
                print(timestamp)

        Returns:
            id (int) : The id of the callback.
        """
        return CarlaDataProvider.get_world().on_tick(func)

    @staticmethod
    def remove_callback(id):
        """Remove a callback from the simulator.

        Args:
            id (int): The id of the callback.
        """
        CarlaDataProvider.get_world().remove_on_tick(id)

    @staticmethod
    def generate_spawn_point(pos, rot=None):
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
                CarlaDataProvider.get_map()
                .get_waypoint(loc, project_to_road=True)
                .transform.rotation
            )
        else:
            rot = carla.Rotation(*rot)
        
        if len(pos) > 3:
            rot.yaw = pos[3]

        return carla.Transform(loc, rot)

    @staticmethod
    def set_weather(index):
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

        CarlaDataProvider.get_world().set_weather(weather)

        return [
            weather.cloudiness,
            weather.precipitation,
            weather.precipitation_deposits,
            weather.wind_intensity,
        ]

    @staticmethod
    def cleanup():
        if Simulator._process:
            print("Killing live carla process:", Simulator._process)
            if IS_WINDOWS_PLATFORM:
                # for Windows
                subprocess.call(["taskkill", "/F", "/T", "/PID",
                                str(Simulator._process.pid)])
            else:
                # for Linux
                pgid = os.getpgid(Simulator._process.pid)
                os.killpg(pgid, signal.SIGKILL)

            Simulator._process = None

    @staticmethod
    def clear_server_state():
        """Clear server process"""
        print("Clearing Carla server state")
        try:
            if Simulator._client:
                Simulator._client = None
        except Exception as e:
            print("Error disconnecting client: {}".format(e))

        SensorDataProvider.cleanup()
        CarlaDataProvider.reset(completely=True)
        GameTime.restart()
        Simulator.cleanup()


def termination_cleanup(*_):
    Simulator.cleanup()
    sys.exit(0)


signal.signal(signal.SIGTERM, termination_cleanup)
signal.signal(signal.SIGINT, termination_cleanup)
atexit.register(Simulator.cleanup)
