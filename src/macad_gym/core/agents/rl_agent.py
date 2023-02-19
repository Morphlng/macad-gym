'''
rl_agent.py: Agent class for Macad-Gym, compatible with scenario_runner
__author__: Morphlng
'''

import carla
import math
from macad_gym.core.agents.macad_agent import MacadAgent


class RLAgent(MacadAgent):

    """
        Reinforcement Learning Agent to control the ego via input actions
    """

    def setup(self, actor_config):
        """
        Setup the agent parameters
        """
        super().setup(actor_config)
        self.actor = self.simulator.get_actor_by_id(actor_config["id"])

    def run_step(self, input_data, timestamp=None):
        """
        Args
            input_data: raw action given by RL algorithm. E.g.

            input_data = {
                "throttle": 0.0,
                "steer": 0.0,
                "brake": 0.0,
                "hand_brake": False,
                "reverse": False,
            }

            timestamp: not used

        Return:
            Carla Control
        """
        agent_type = self.actor_config.get("type", "vehicle")

        # space of ped actors
        if agent_type == "pedestrian":
            rotation = self.actor.get_transform().rotation
            rotation.yaw += input_data["steer"] * 10.0
            x_dir = math.cos(math.radians(rotation.yaw))
            y_dir = math.sin(math.radians(rotation.yaw))

            return carla.WalkerControl(
                speed=3.0 * input_data["throttle"],
                direction=carla.Vector3D(x_dir, y_dir, 0.0),
            )

        # TODO: Change this if different vehicle types (Eg.:vehicle_4W,
        #  vehicle_2W, etc) have different control APIs
        elif "vehicle" in agent_type:
            return carla.VehicleControl(
                throttle=input_data["throttle"],
                steer=input_data["steer"],
                brake=input_data["brake"],
                hand_brake=input_data["hand_brake"],
                reverse=input_data["reverse"],
            )

    def __call__(self, action=None):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        control = self.run_step(action)
        control.manual_gear_shift = False
        return control
