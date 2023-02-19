#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all frequently used data from CARLA via
local buffers to avoid blocking calls to CARLA
"""

from __future__ import print_function

import math
import re
from numpy import random
from six import iteritems

import carla


def calculate_velocity(actor):
    """
    Method to calculate the velocity of a actor
    """
    velocity_squared = actor.get_velocity().x**2
    velocity_squared += actor.get_velocity().y**2
    return math.sqrt(velocity_squared)


class CarlaDataProvider(object):  # pylint: disable=too-many-public-methods

    """
    This class provides access to various data of all registered actors
    It buffers the data and updates it on every CARLA tick

    Currently available data:
    - Absolute velocity
    - Location
    - Transform

    Potential additions:
    - Acceleration

    In addition it provides access to the map and the transform of all traffic lights
    """

    def __init__(self):
        self._actor_velocity_map = {}
        self._actor_location_map = {}
        self._actor_transform_map = {}
        self._traffic_light_map = {}
        self._carla_actor_pool = {}
        self._global_osc_parameters = {}
        self._client = None
        self._world = None
        self._map = None
        self._sync_flag = False
        self._spawn_points = None
        self._spawn_index = 0
        self._blueprint_library = None
        self._ego_vehicle_route = None
        self._traffic_manager_port = 8000
        self._random_seed = 2000
        self._rng = random.RandomState(self._random_seed)

    def register_actor(self, actor):
        """
        Add new actor to dictionaries
        If actor already exists, throw an exception
        """
        if actor in self._actor_velocity_map:
            raise KeyError(
                "Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
        else:
            self._actor_velocity_map[actor] = 0.0

        if actor in self._actor_location_map:
            raise KeyError(
                "Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
        else:
            self._actor_location_map[actor] = None

        if actor in self._actor_transform_map:
            raise KeyError(
                "Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
        else:
            self._actor_transform_map[actor] = None

    def update_osc_global_params(self, parameters):
        """
        updates/initializes global osc parameters.
        """
        self._global_osc_parameters.update(parameters)

    def get_osc_global_param_value(self, ref):
        """
        returns updated global osc parameter value.
        """
        return self._global_osc_parameters.get(ref.replace("$", ""))

    def register_actors(self, actors):
        """
        Add new set of actors to dictionaries
        """
        for actor in actors:
            self.register_actor(actor)

    def on_carla_tick(self):
        """
        Callback from CARLA
        """
        for actor in self._actor_velocity_map:
            if actor is not None and actor.is_alive:
                self._actor_velocity_map[actor] = calculate_velocity(actor)

        for actor in self._actor_location_map:
            if actor is not None and actor.is_alive:
                self._actor_location_map[actor] = actor.get_location()

        for actor in self._actor_transform_map:
            if actor is not None and actor.is_alive:
                self._actor_transform_map[actor] = actor.get_transform()

        world = self._world
        if world is None:
            print("WARNING: self couldn't find the world")

    def get_velocity(self, actor):
        """
        returns the absolute velocity for the given actor
        """
        for key in self._actor_velocity_map:
            if key.id == actor.id:
                return self._actor_velocity_map[key]

        # We are intentionally not throwing here
        # This may cause exception loops in py_trees
        print('{}.get_velocity: {} not found!' .format(__name__, actor))
        return 0.0

    def get_location(self, actor):
        """
        returns the location for the given actor
        """
        for key in self._actor_location_map:
            if key.id == actor.id:
                return self._actor_location_map[key]

        # We are intentionally not throwing here
        # This may cause exception loops in py_trees
        print('{}.get_location: {} not found!' .format(__name__, actor))
        return None

    def get_transform(self, actor):
        """
        returns the transform for the given actor
        """
        for key in self._actor_transform_map:
            if key.id == actor.id:
                return self._actor_transform_map[key]

        # We are intentionally not throwing here
        # This may cause exception loops in py_trees
        print('{}.get_transform: {} not found!' .format(__name__, actor))
        return None

    def set_client(self, client):
        """
        Set the CARLA client
        """
        self._client = client

    def get_client(self):
        """
        Get the CARLA client
        """
        return self._client

    def set_world(self, world):
        """
        Set the world and world settings
        """
        self._world = world
        self._sync_flag = world.get_settings().synchronous_mode
        self._map = world.get_map()
        self._blueprint_library = world.get_blueprint_library()
        self.generate_spawn_points()
        self.prepare_map()

    def get_world(self):
        """
        Return world
        """
        return self._world

    def get_map(self, world=None):
        """
        Get the current map
        """
        if self._map is None:
            if world is None:
                if self._world is None:
                    raise ValueError(
                        "class member \'world'\' not initialized yet")
                else:
                    self._map = self._world.get_map()
            else:
                self._map = world.get_map()

        return self._map

    def is_sync_mode(self):
        """
        @return true if syncronuous mode is used
        """
        return self._sync_flag

    def find_weather_presets(self):
        """
        Get weather presets from CARLA
        """
        rgx = re.compile(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

        def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters)
                   if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

    def prepare_map(self):
        """
        This function set the current map and loads all traffic lights for this map to
        _traffic_light_map
        """
        if self._map is None:
            self._map = self._world.get_map()

        # Parse all traffic lights
        self._traffic_light_map.clear()
        for traffic_light in self._world.get_actors().filter('*traffic_light*'):
            if traffic_light not in self._traffic_light_map.keys():
                self._traffic_light_map[traffic_light] = traffic_light.get_transform(
                )
            else:
                raise KeyError(
                    "Traffic light '{}' already registered. Cannot register twice!".format(traffic_light.id))

    def annotate_trafficlight_in_group(self, traffic_light):
        """
        Get dictionary with traffic light group info for a given traffic light
        """
        dict_annotations = {'ref': [], 'opposite': [], 'left': [], 'right': []}

        # Get the waypoints
        ref_location = self.get_trafficlight_trigger_location(
            traffic_light)
        ref_waypoint = self.get_map().get_waypoint(ref_location)
        ref_yaw = ref_waypoint.transform.rotation.yaw

        group_tl = traffic_light.get_group_traffic_lights()

        for target_tl in group_tl:
            if traffic_light.id == target_tl.id:
                dict_annotations['ref'].append(target_tl)
            else:
                # Get the angle between yaws
                target_location = self.get_trafficlight_trigger_location(
                    target_tl)
                target_waypoint = self.get_map().get_waypoint(target_location)
                target_yaw = target_waypoint.transform.rotation.yaw

                diff = (target_yaw - ref_yaw) % 360

                if diff > 330:
                    continue
                elif diff > 225:
                    dict_annotations['right'].append(target_tl)
                elif diff > 135.0:
                    dict_annotations['opposite'].append(target_tl)
                elif diff > 30:
                    dict_annotations['left'].append(target_tl)

        return dict_annotations

    def get_trafficlight_trigger_location(self, traffic_light):    # pylint: disable=invalid-name
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """
        def rotate_point(point, angle):
            """
            rotate a given point by a given angle
            """
            x_ = math.cos(math.radians(angle)) * point.x - \
                math.sin(math.radians(angle)) * point.y
            y_ = math.sin(math.radians(angle)) * point.x - \
                math.cos(math.radians(angle)) * point.y

            return carla.Vector3D(x_, y_, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(
            traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)

    def update_light_states(self, ego_light, annotations, states, freeze=False, timeout=1000000000):
        """
        Update traffic light states
        """
        reset_params = []

        for state in states:
            relevant_lights = []
            if state == 'ego':
                relevant_lights = [ego_light]
            else:
                relevant_lights = annotations[state]
            for light in relevant_lights:
                prev_state = light.get_state()
                prev_green_time = light.get_green_time()
                prev_red_time = light.get_red_time()
                prev_yellow_time = light.get_yellow_time()
                reset_params.append({'light': light,
                                     'state': prev_state,
                                     'green_time': prev_green_time,
                                     'red_time': prev_red_time,
                                     'yellow_time': prev_yellow_time})

                light.set_state(states[state])
                if freeze:
                    light.set_green_time(timeout)
                    light.set_red_time(timeout)
                    light.set_yellow_time(timeout)

        return reset_params

    def reset_lights(self, reset_params):
        """
        Reset traffic lights
        """
        for param in reset_params:
            param['light'].set_state(param['state'])
            param['light'].set_green_time(param['green_time'])
            param['light'].set_red_time(param['red_time'])
            param['light'].set_yellow_time(param['yellow_time'])

    def get_next_traffic_light(self, actor, use_cached_location=True):
        """
        returns the next relevant traffic light for the provided actor
        """

        if not use_cached_location:
            location = actor.get_transform().location
        else:
            location = self.get_location(actor)

        waypoint = self.get_map().get_waypoint(location)
        # Create list of all waypoints until next intersection
        list_of_waypoints = []
        while waypoint and not waypoint.is_intersection:
            list_of_waypoints.append(waypoint)
            waypoint = waypoint.next(2.0)[0]

        # If the list is empty, the actor is in an intersection
        if not list_of_waypoints:
            return None

        relevant_traffic_light = None
        distance_to_relevant_traffic_light = float("inf")

        for traffic_light in self._traffic_light_map:
            if hasattr(traffic_light, 'trigger_volume'):
                tl_t = self._traffic_light_map[traffic_light]
                transformed_tv = tl_t.transform(
                    traffic_light.trigger_volume.location)
                distance = carla.Location(transformed_tv).distance(
                    list_of_waypoints[-1].transform.location)

                if distance < distance_to_relevant_traffic_light:
                    relevant_traffic_light = traffic_light
                    distance_to_relevant_traffic_light = distance

        return relevant_traffic_light

    def set_ego_vehicle_route(self, route):
        """
        Set the route of the ego vehicle

        @todo extend ego_vehicle_route concept to support multi ego_vehicle scenarios
        """
        self._ego_vehicle_route = route

    def get_ego_vehicle_route(self):
        """
        returns the currently set route of the ego vehicle
        Note: Can be None
        """
        return self._ego_vehicle_route

    def generate_spawn_points(self):
        """
        Generate spawn points for the current map
        """
        spawn_points = list(self.get_map(
            self._world).get_spawn_points())
        self._rng.shuffle(spawn_points)
        self._spawn_points = spawn_points
        self._spawn_index = 0

    def create_blueprint(self, model, rolename='scenario', color=None, actor_category="car", safe=False):
        """
        Function to setup the blueprint of an actor given its model and other relevant parameters
        """

        _actor_blueprint_categories = {
            'car': 'vehicle.tesla.model3',
            'van': 'vehicle.volkswagen.t2',
            'truck': 'vehicle.carlamotors.carlacola',
            'trailer': '',
            'semitrailer': '',
            'bus': 'vehicle.volkswagen.t2',
            'motorbike': 'vehicle.kawasaki.ninja',
            'bicycle': 'vehicle.diamondback.century',
            'train': '',
            'tram': '',
            'pedestrian': 'walker.pedestrian.0001',
        }

        # Set the model
        try:
            blueprints = self._blueprint_library.filter(model)
            blueprints_ = []
            if safe:
                for bp in blueprints:
                    if bp.id.endswith('firetruck') or bp.id.endswith('ambulance') \
                            or int(bp.get_attribute('number_of_wheels')) != 4:
                        # Two wheeled vehicles take much longer to render + bicicles shouldn't behave like cars
                        continue
                    blueprints_.append(bp)
            else:
                blueprints_ = blueprints

            blueprint = self._rng.choice(blueprints_)
        except ValueError:
            # The model is not part of the blueprint library. Let's take a default one for the given category
            bp_filter = "vehicle.*"
            new_model = _actor_blueprint_categories[actor_category]
            if new_model != '':
                bp_filter = new_model
            print("WARNING: Actor model {} not available. Using instead {}".format(
                model, new_model))
            blueprint = self._rng.choice(
                self._blueprint_library.filter(bp_filter))

        # Set the color
        if color:
            if not blueprint.has_attribute('color'):
                print(
                    "WARNING: Cannot set Color ({}) for actor {} due to missing blueprint attribute".format(
                        color, blueprint.id))
            else:
                default_color_rgba = blueprint.get_attribute(
                    'color').as_color()
                default_color = '({}, {}, {})'.format(
                    default_color_rgba.r, default_color_rgba.g, default_color_rgba.b)
                try:
                    blueprint.set_attribute('color', color)
                except ValueError:
                    # Color can't be set for this vehicle
                    print("WARNING: Color ({}) cannot be set for actor {}. Using instead: ({})".format(
                        color, blueprint.id, default_color))
                    blueprint.set_attribute('color', default_color)
        else:
            if blueprint.has_attribute('color') and rolename != 'hero':
                color = self._rng.choice(
                    blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

        # Make pedestrians mortal
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')

        # Set the rolename
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', rolename)

        return blueprint

    def handle_actor_batch(self, batch, tick=True):
        """
        Forward a CARLA command batch to spawn actors to CARLA, and gather the responses.
        Returns list of actors on success, none otherwise
        """
        sync_mode = self.is_sync_mode()
        actors = []

        if self._client:
            responses = self._client.apply_batch_sync(
                batch, sync_mode and tick)
        else:
            raise ValueError("class member \'client'\' not initialized yet")

        # Wait (or not) for the actors to be spawned properly before we do anything
        if not tick:
            pass
        elif sync_mode:
            self._world.tick()
        else:
            self._world.wait_for_tick()

        actor_ids = [r.actor_id for r in responses if not r.error]
        for r in responses:
            if r.error:
                print("WARNING: Not all actors were spawned")
                break
        actors = list(self._world.get_actors(actor_ids))
        return actors

    def request_new_actor(self, model, spawn_point, attach_to=None, rolename='scenario', autopilot=False,
                          random_location=False, color=None, actor_category="car",
                          safe_blueprint=False, blueprint=None, tick=True):
        """
        This method tries to create a new actor, returning it if successful (None otherwise).

        Args:
            model (str): model of the actor to be spawned, e.g. 'vehicle.tesla.model3'
            spawn_point (carla.Transform): spawn point of the actor
            attach_to (carla.Actor): attach the new actor to an existing actor (e.g. WalkerController attch to a pedestrian)
            autopilot (bool): if True, the actor will be spawned with autopilot enabled
            safe_blueprint (bool): if True, the blueprint will be filtered to avoid spawning firetrucks, ambulances, etc.
            blueprint (carla.Blueprint): blueprint to be used for spawning the actor. If None, a new blueprint will be created
            tick (bool): if True, the world will be ticked after spawning the actor

            Other arguments are the same as in create_blueprint.
        """
        if blueprint is None:
            blueprint = self.create_blueprint(
                model, rolename, color, actor_category, safe_blueprint)

        if random_location:
            actor = None
            while not actor:
                spawn_point = self._rng.choice(
                    self._spawn_points)
                actor = self._world.try_spawn_actor(
                    blueprint, spawn_point, attach_to)

        else:
            # slightly lift the actor to avoid collisions with ground when spawning the actor
            # DO NOT USE spawn_point directly, as this will modify spawn_point permanently
            _spawn_point = carla.Transform(
                carla.Location(), spawn_point.rotation)
            _spawn_point.location.x = spawn_point.location.x
            _spawn_point.location.y = spawn_point.location.y
            _spawn_point.location.z = spawn_point.location.z + 0.2
            actor = self._world.try_spawn_actor(
                blueprint, _spawn_point, attach_to)

        if actor is None:
            print("WARNING: Cannot spawn actor {} at position {}".format(
                model, spawn_point.location))
            return None

        # De/activate the autopilot of the actor if it belongs to vehicle
        if actor.type_id.startswith('vehicle.'):
            actor.set_autopilot(
                autopilot, self._traffic_manager_port)
        elif autopilot:
            print("WARNING: Tried to set the autopilot of a non vehicle actor")

        # Wait for the actor to be spawned properly before we do anything
        if not tick:
            pass
        elif self.is_sync_mode():
            self._world.tick()
        else:
            self._world.wait_for_tick()

        self._carla_actor_pool[actor.id] = actor
        self.register_actor(actor)
        return actor

    def request_new_actors(self, actor_list, safe_blueprint=False, tick=True):
        """
        This method tries to series of actor in batch. If this was successful,
        the new actors are returned, None otherwise.

        param:
        - actor_list: list of ActorConfigurationData
        """

        SpawnActor = carla.command.SpawnActor                      # pylint: disable=invalid-name
        PhysicsCommand = carla.command.SetSimulatePhysics          # pylint: disable=invalid-name
        FutureActor = carla.command.FutureActor                    # pylint: disable=invalid-name
        ApplyTransform = carla.command.ApplyTransform              # pylint: disable=invalid-name
        SetAutopilot = carla.command.SetAutopilot                  # pylint: disable=invalid-name
        SetVehicleLightState = carla.command.SetVehicleLightState  # pylint: disable=invalid-name

        batch = []

        self.generate_spawn_points()

        for actor in actor_list:

            # Get the blueprint
            blueprint = self.create_blueprint(
                actor.model, actor.rolename, actor.color, actor.category, safe_blueprint)

            # Get the spawn point
            transform = actor.transform
            if actor.random_location:
                if self._spawn_index >= len(self._spawn_points):
                    print("No more spawn points to use")
                    break
                else:
                    _spawn_point = self._spawn_points[
                        self._spawn_index]  # pylint: disable=unsubscriptable-object
                    self._spawn_index += 1

            else:
                _spawn_point = carla.Transform()
                _spawn_point.rotation = transform.rotation
                _spawn_point.location.x = transform.location.x
                _spawn_point.location.y = transform.location.y
                if blueprint.has_tag('walker'):
                    # On imported OpenDRIVE maps, spawning of pedestrians can fail.
                    # By increasing the z-value the chances of success are increased.
                    map_name = self._map.name.split("/")[-1]
                    if not map_name.startswith('OpenDrive'):
                        _spawn_point.location.z = transform.location.z + 0.2
                    else:
                        _spawn_point.location.z = transform.location.z + 0.8
                else:
                    _spawn_point.location.z = transform.location.z + 0.2

            # Get the command
            command = SpawnActor(blueprint, _spawn_point)
            command.then(SetAutopilot(FutureActor, actor.autopilot,
                         self._traffic_manager_port))

            if actor.args is not None and 'physics' in actor.args and actor.args['physics'] == "off":
                command.then(ApplyTransform(FutureActor, _spawn_point)).then(
                    PhysicsCommand(FutureActor, False))
            elif actor.category == 'misc':
                command.then(PhysicsCommand(FutureActor, True))
            if actor.args is not None and 'lights' in actor.args and actor.args['lights'] == "on":
                command.then(SetVehicleLightState(
                    FutureActor, carla.VehicleLightState.All))

            batch.append(command)

        actors = self.handle_actor_batch(batch, tick)
        for actor in actors:
            if actor is None:
                continue
            self._carla_actor_pool[actor.id] = actor
            self.register_actor(actor)

        return actors

    def request_new_batch_actors(self, model, amount, spawn_points, autopilot=False,
                                 random_location=False, rolename='scenario',
                                 safe_blueprint=False, tick=True):
        """
        Simplified version of "request_new_actors". This method also create several actors in batch.

        Instead of needing a list of ActorConfigurationData, an "amount" parameter is used.
        This makes actor spawning easier but reduces the amount of configurability.

        Some parameters are the same for all actors (rolename, autopilot and random location)
        while others are randomized (color)
        """

        SpawnActor = carla.command.SpawnActor      # pylint: disable=invalid-name
        SetAutopilot = carla.command.SetAutopilot  # pylint: disable=invalid-name
        FutureActor = carla.command.FutureActor    # pylint: disable=invalid-name

        self.generate_spawn_points()

        batch = []

        for i in range(amount):
            # Get vehicle by model
            blueprint = self.create_blueprint(
                model, rolename, safe=safe_blueprint)

            if random_location:
                if self._spawn_index >= len(self._spawn_points):
                    print("No more spawn points to use. Spawned {} actors out of {}".format(
                        i + 1, amount))
                    break
                else:
                    spawn_point = self._spawn_points[
                        self._spawn_index]  # pylint: disable=unsubscriptable-object
                    self._spawn_index += 1
            else:
                try:
                    spawn_point = spawn_points[i]
                except IndexError:
                    print(
                        "The amount of spawn points is lower than the amount of vehicles spawned")
                    break

            if spawn_point:
                batch.append(SpawnActor(blueprint, spawn_point).then(
                    SetAutopilot(FutureActor, autopilot,
                                 self._traffic_manager_port)))

        actors = self.handle_actor_batch(batch, tick)
        for actor in actors:
            if actor is None:
                continue
            self._carla_actor_pool[actor.id] = actor
            self.register_actor(actor)

        return actors

    def get_actors(self):
        """
        Return list of actors and their ids

        Note: iteritems from six is used to allow compatibility with Python 2 and 3
        """
        return iteritems(self._carla_actor_pool)

    def actor_id_exists(self, actor_id):
        """
        Check if a certain id is still at the simulation
        """
        if actor_id in self._carla_actor_pool:
            return True

        return False

    def get_hero_actor(self):
        """
        Get the actor object of the hero actor if it exists, returns none otherwise.
        """
        for actor_id in self._carla_actor_pool:
            if self._carla_actor_pool[actor_id].attributes['role_name'] == 'hero':
                return self._carla_actor_pool[actor_id]
        return None

    def get_actor_by_id(self, actor_id):
        """
        Get an actor from the pool by using its ID. If the actor
        does not exist, None is returned.
        """
        if actor_id in self._carla_actor_pool:
            return self._carla_actor_pool[actor_id]

        print("Non-existing actor id {}".format(actor_id))
        return None

    def remove_actor_by_id(self, actor_id):
        """
        Remove an actor from the pool using its ID
        """
        if actor_id in self._carla_actor_pool:
            print(
                f"Removing actor id {actor_id}, final transform {self._carla_actor_pool[actor_id].get_transform()}.")

            self._carla_actor_pool[actor_id].destroy()
            self._carla_actor_pool[actor_id] = None
            self._carla_actor_pool.pop(actor_id)
        else:
            print("Trying to remove a non-existing actor id {}".format(actor_id))

    def remove_actors_in_surrounding(self, location, distance):
        """
        Remove all actors from the pool that are closer than distance to the
        provided location
        """
        for actor_id in self._carla_actor_pool.copy():
            if self._carla_actor_pool[actor_id].get_location().distance(location) < distance:
                self._carla_actor_pool[actor_id].destroy()
                self._carla_actor_pool.pop(actor_id)

        # Remove all keys with None values
        self._carla_actor_pool = dict(
            {k: v for k, v in self._carla_actor_pool.items() if v})

    def get_traffic_manager_port(self):
        """
        Get the port of the traffic manager.
        """
        return self._traffic_manager_port

    def set_traffic_manager_port(self, tm_port):
        """
        Set the port to use for the traffic manager.
        """
        self._traffic_manager_port = tm_port

    def cleanup(self, completely=False):
        """
        Cleanup and remove all entries from all dictionaries
        """
        DestroyActor = carla.command.DestroyActor       # pylint: disable=invalid-name
        batch = []

        for actor_id in self._carla_actor_pool.copy():
            actor = self._carla_actor_pool[actor_id]
            if actor is not None and actor.is_alive:
                if isinstance(actor, carla.WalkerAIController):
                    actor.stop()
                batch.append(DestroyActor(actor))

        if self._client:
            try:
                self._client.apply_batch_sync(batch)
            except RuntimeError as e:
                if "time-out" in str(e):
                    pass
                else:
                    raise e

        self.reset(completely)

    def reset(self, completely=False):
        """
        Reset the data provider
        """
        self._actor_velocity_map.clear()
        self._actor_location_map.clear()
        self._actor_transform_map.clear()

        self._ego_vehicle_route = None
        self._carla_actor_pool = {}
        self._spawn_index = 0
        self._rng = random.RandomState(self._random_seed)

        if completely:
            self._map = None
            self._world = None
            self._sync_flag = False
            self._client = None
            self._spawn_points = None
            self._traffic_light_map.clear()
