#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This file containts CallBack class and SensorInterface, responsible of
handling the use of sensors for the agents
"""

import copy
import logging

try:
    from queue import Queue
    from queue import Empty
except ImportError:
    from Queue import Queue
    from Queue import Empty

import numpy as np

import carla


class SensorReceivedNoData(Exception):

    """
    Exceptions thrown when the sensors used by the agent take too long to receive data
    """


class SensorDataProvider:
    """
    All sensor data will be buffered in this class

    The data can be retreived in following data structure:

    {
        'camera': {
            'actor_id': {
                'sensor_id': (raw_data, processed_data),
                ...
            },
            ...
        },
        'collision': {
            'actor_id': CollisionSensor,
            ...
        },
        'lane_invasion': {
            'actor_id': LaneInvasionSensor,
            ...
        },
        ...
    }
    """
    _global_data = {}
    _camera_data_dict = {}
    _collision_sensors = {}
    _lane_invasion_sensors = {}

    @staticmethod
    def update_camera_data(actor_id, data):
        """
        Updates the camera data

        Args:
            actor_id (str): actor id
            data (Dict): image data from sensor_interface.get_data(). E.g.

            data = {
                "sensor_id": (raw_data : carla.Image, processed_data : ndarray),
                ...
            }
        """
        if data is None:
            filter_data = None
        else:
            filter_data = {k:v for k,v in data.items() if k != "ManualControl"}

        SensorDataProvider._camera_data_dict[actor_id] = filter_data

    @staticmethod
    def update_collision_sensor(actor_id, sensor):
        """
        Updates a collision sensor
        """
        SensorDataProvider._collision_sensors[actor_id] = sensor

    @staticmethod
    def update_lane_invasion_sensor(actor_id, sensor):
        """
        Updates a lane invasion sensor
        """
        SensorDataProvider._lane_invasion_sensors[actor_id] = sensor

    @staticmethod
    def get_camera_data(actor_id):
        """
        Returns the camera data of the actor

        Returns:
            Dict: image data from sensor_interface.get_data(). E.g.

            data = {
                "sensor_id": (raw_data : carla.Image, processed_data : ndarray),
                ...
            }
        """
        return SensorDataProvider._camera_data_dict[actor_id]

    @staticmethod
    def get_collision_sensor(actor_id):
        """
        Returns:
            CollisionSensor: collision sensor of the actor
        """
        return SensorDataProvider._collision_sensors[actor_id]

    @staticmethod
    def get_lane_invasion_sensor(actor_id):
        """
        Returns:
            LaneInvasionSensor: lane invasion sensor of the actor
        """
        return SensorDataProvider._lane_invasion_sensors[actor_id]

    @staticmethod
    def get_all_data():
        """
        Returns all sensor data
        """
        return {
            'camera': SensorDataProvider._camera_data_dict,
            'collision': SensorDataProvider._collision_sensors,
            'lane_invasion': SensorDataProvider._lane_invasion_sensors,
            'global': SensorDataProvider._global_data
        }

    @staticmethod
    def cleanup():
        for actor_id, colli in SensorDataProvider._collision_sensors.items():
            if colli.sensor.is_alive:
                colli.sensor.destroy()
        for actor_id, lane in SensorDataProvider._lane_invasion_sensors.items():
            if lane.sensor.is_alive:
                lane.sensor.destroy()

        SensorDataProvider._global_data = {}
        SensorDataProvider._camera_data_dict = {}
        SensorDataProvider._collision_sensors = {}
        SensorDataProvider._lane_invasion_sensors = {}


class CallBack(object):

    """
    Class the sensors listen to in order to receive their data each frame
    """

    def __init__(self, tag, sensor, data_provider):
        """
        Initializes the call back
        """
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor)

    def __call__(self, data):
        """
        call function
        """
        if isinstance(data, carla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        """
        parses cameras
        """
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._data_provider.update_sensor(tag, array, image)

    def _parse_lidar_cb(self, lidar_data, tag):
        """
        parses lidar sensors
        """
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data)

    def _parse_radar_cb(self, radar_data, tag):
        """
        parses radar sensors
        """
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data)

    def _parse_gnss_cb(self, gnss_data, tag):
        """
        parses gnss sensors
        """
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data)

    def _parse_imu_cb(self, imu_data, tag):
        """
        parses IMU sensors
        """
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                          ], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, imu_data)


class SensorInterface(object):

    """
    Class that contains all sensor data for one agent
    """

    def __init__(self):
        """
        Initializes the class
        """
        self._sensors_objects = {}
        self._new_data_buffers = Queue()
        self._queue_timeout = 10

    def register_sensor(self, tag, sensor):
        """
        Registers the sensors
        """
        if tag in self._sensors_objects:
            raise ValueError("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor

    def update_sensor(self, tag, data, raw_data):
        """
        Updates the sensor
        """
        if tag not in self._sensors_objects:
            raise ValueError(
                "The sensor with tag [{}] has not been created!".format(tag))

        self._new_data_buffers.put((tag, raw_data, data))

    def get_data(self):
        """
        Returns the data of a sensor
        """
        try:
            data_dict = {}
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):

                sensor_data = self._new_data_buffers.get(
                    True, self._queue_timeout)

                # data_dict["sensor_id"] = (raw_data, processed_data)
                data_dict[sensor_data[0]] = ((sensor_data[1], sensor_data[2]))

        except Empty:
            raise SensorReceivedNoData(
                "A sensor took too long to send its data")

        return data_dict
