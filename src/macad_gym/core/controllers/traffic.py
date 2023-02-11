import random
import logging
import carla
from macad_gym.core.data.carla_data_provider import CarlaDataProvider
from macad_gym.core.data.simulator import Simulator

# TODO make the seed user configurable
random.seed(10)
logger = logging.getLogger(__name__)


def apply_traffic(num_vehicles, num_pedestrians, percentagePedestriansRunning=0.0, percentagePedestriansCrossing=0.0, safe=False):
    # --------------
    # Spawn vehicles
    # --------------
    world = Simulator.get_world()
    traffic_manager = Simulator.get_traffic_manager()

    spawn_points = CarlaDataProvider._spawn_points
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
        vehicle = CarlaDataProvider.request_new_actor(
            "vehicle", transform, rolename="autopilot", autopilot=True, safe_blueprint=safe)
        if vehicle is not None:
            vehicles_list.append(vehicle)
        else:
            failed_v += 1

    logger.info("{}/{} vehicles correctly spawned.".format(num_vehicles-failed_v, num_vehicles))

    # -------------
    # Spawn Walkers
    # -------------
    blueprints = CarlaDataProvider._blueprint_library.filter(
        "walker.pedestrian.*")
    pedestrian_controller_bp = CarlaDataProvider._blueprint_library.find(
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
                speed = pedestrian_bp.get_attribute('speed').recommended_values[1]  # walking
            else:
                speed = pedestrian_bp.get_attribute('speed').recommended_values[2]  # running
        else:
            speed = 0.0
        pedestrian = CarlaDataProvider.request_new_actor(
            "walker.pedestrian", spawn_point, actor_category="pedestrian", blueprint=pedestrian_bp)
        if pedestrian is not None:
            controller = CarlaDataProvider.request_new_actor(
                "controller.ai.walker", carla.Transform(), attach_to=pedestrian, blueprint=pedestrian_controller_bp)
            if controller is not None:
                pedestrians_list.append(pedestrian)
                controllers_list.append(controller)
                pedestrians_speed.append(speed)
            else:
                CarlaDataProvider.remove_actor_by_id(pedestrian.id)
                failed_p += 1
        else:
            failed_p += 1

    logger.info("{}/{} pedestrians correctly spawned.".format(num_pedestrians-failed_p, num_pedestrians))
    Simulator.tick()

    # Initialize each controller and set target to walk
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i, controller in enumerate(controllers_list):
        controller.start()  # start walker
        controller.go_to_location(world.get_random_location_from_navigation())  # set walk to random point
        controller.set_max_speed(float(pedestrians_speed[int(i / 2)]))  # max speed

    traffic_manager.global_percentage_speed_difference(30.0)

    return vehicles_list, (pedestrians_list, controllers_list)
