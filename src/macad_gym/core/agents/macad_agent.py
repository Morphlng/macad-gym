'''
macad_agent.py: MacadAgent inherit from AutonomousAgent.py. The purpose is to make macad-gym current config works.
__author__: Morphlng
'''

import carla
from macad_gym.core.agents.autonomous_agent import AutonomousAgent
from macad_gym.core.data.timer import GameTime
from macad_gym.core.data.sensor_interface import SensorDataProvider


def sensor_name_to_bp(camera_type):
    """Convert sensor name to blueprint
    """
    if camera_type == "rgb":
        return "sensor.camera.rgb"
    elif "depth" in camera_type:
        return "sensor.camera.depth"
    elif "semseg" in camera_type:
        return "sensor.camera.semantic_segmentation"
    elif camera_type == "lidar":
        return "sensor.lidar.ray_cast"
    else:
        raise ValueError("Unknown sensor name: {}".format(camera_type))


class MacadAgent(AutonomousAgent):
    """
        All agent used in macad-gym should inherit from this class.

        You can override setup(), run_step() and destroy() methods

        Note:
            call super().setup(actor_config) in your setup() method to parse sensors from actor_config
    """

    _camera_transforms = [
        {"x": 1.8, "y": 0.0, "z": 1.7, "pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        {"x": -5.5, "y": 0.0, "z": 2.8, "pitch": -15.0, "yaw": 0.0, "roll": 0.0},
    ]

    def setup(self, config):
        self.obs = None
        self.actor_config = config
        self.sensor_list = []
        self.parse_sensors()

    def sensors(self):
        return self.sensor_list

    def parse_sensors(self):
        """Parse sensors from actor config to make it compatible for setup_sensors in AgentWrapper
        """
        camera_type = self.actor_config["camera_type"]
        camera_pos = self.actor_config.get("camera_position", 0)

        sensor_spec = {
            'id': camera_type,
            'type': sensor_name_to_bp(camera_type),
            'width': self.actor_config["x_res"],
            'height': self.actor_config["y_res"],
            'attachment_type': carla.AttachmentType.Rigid,
        }
        sensor_spec.update(self._camera_transforms[camera_pos])

        # Use default values to meet AgentWrapper's requirement
        if camera_type == 'rgb':
            sensor_spec.update({'fov': 90})
        elif camera_type == 'lidar':
            sensor_spec.update(
                {
                    'range': 10.0,
                    'rotation_frequency': 10.0,
                    'channels': 32,
                    'upper_fov': 10.0,
                    'lower_fov': -30.0,
                    'points_per_second': 56000,
                }
            )

        self.sensor_list.append(sensor_spec)
    
    def on_carla_tick(self, snapshot):
        """Update obs on carla tick
        """
        if not self.sensor_interface._new_data_buffers.empty():
            self.obs = self.sensor_interface.get_data()
        
        SensorDataProvider.update_camera_data(self.actor_config["actor_id"], self.obs)

    def __call__(self, action=None):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        timestamp = GameTime.get_time()
        control = self.run_step(self.obs, timestamp)
        control.manual_gear_shift = False
        return control
