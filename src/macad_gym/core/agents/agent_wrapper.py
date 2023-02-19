#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Wrapper for autonomous agents required for tracking and checking of used sensors
"""

from __future__ import print_function

import carla

from macad_gym.core.data.sensor_interface import CallBack


class AgentWrapper(object):

    """
    Wrapper for autonomous agents required for tracking and checking of used sensors

    usage:
        agent = AutonomousAgent()
        agent = AgentWrapper(agent)
    """

    _agent = None
    _sensors_list = []

    def __init__(self, agent):
        """
        Set the autonomous agent
        """
        self._agent = agent

    def __call__(self, action=None):
        """
        Pass the call directly to the agent
        """
        return self._agent(action)

    def setup_sensors(self, vehicle, debug_mode=False):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        bp_library = vehicle.get_world().get_blueprint_library()
        for sensor_spec in self._agent.sensors():
            # These are the sensors spawned on the carla world
            bp = bp_library.find(str(sensor_spec['type']))

            if sensor_spec.get('manual_control', False):
                sensor_spec.update({
                    'x': -2.0 * (0.5 + vehicle.bounding_box.extent.x),
                    'y': 0.0,
                    'z': 2.0 * (0.5 + vehicle.bounding_box.extent.z),
                    'pitch': 8.0,
                    'yaw': 0.0,
                    'roll': 0.0,
                })
                sensor_spec["attachment_type"] = carla.AttachmentType.SpringArm

            if sensor_spec['type'].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(sensor_spec['width']))
                bp.set_attribute('image_size_y', str(sensor_spec['height']))
                bp.set_attribute('fov', str(sensor_spec['fov']))
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                 z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                 roll=sensor_spec['roll'],
                                                 yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.lidar'):
                bp.set_attribute('range', str(sensor_spec['range']))
                bp.set_attribute('rotation_frequency', str(
                    sensor_spec['rotation_frequency']))
                bp.set_attribute('channels', str(sensor_spec['channels']))
                bp.set_attribute('upper_fov', str(sensor_spec['upper_fov']))
                bp.set_attribute('lower_fov', str(sensor_spec['lower_fov']))
                bp.set_attribute('points_per_second', str(
                    sensor_spec['points_per_second']))
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                 z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                 roll=sensor_spec['roll'],
                                                 yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.other.gnss'):
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                 z=sensor_spec['z'])
                sensor_rotation = carla.Rotation()

            # create sensor
            sensor_transform = carla.Transform(
                sensor_location, sensor_rotation)
            sensor = vehicle.get_world().spawn_actor(bp, sensor_transform, attach_to=vehicle,
                                                     attachment_type=sensor_spec["attachment_type"])
            # setup callback
            sensor.listen(CallBack(sensor_spec['id'], sensor, self._agent.sensor_interface))
            self._sensors_list.append(sensor)

        # Tick once to spawn the sensors
        vehicle.get_world().tick()

    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].stop()
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        self._sensors_list = []
        self._agent.destroy()
