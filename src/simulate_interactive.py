import copy
import os
import numpy as np
import yaml
import traceback

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from sumocr.interface.sumo_simulation import SumoSimulation
from sumocr.maps.sumo_scenario import ScenarioWrapper
from our_sumo_config import CRSumoConfig as Conf

from sensor import Sensor
from occlusion_tracker import Occlusion_tracker
from planner import Planner
from visualizer import Visualizer
from utilities import add_no_stop_zone_DEU_Ffb, add_building_DEU_Ffb, create_planning_problem_DEU_Ffb, simulation_log


def simulate_interactive(scenario_path):
    with open(scenario_path + '/configuration.yaml', 'r') as file:
        configuration = yaml.load(file, Loader=yaml.FullLoader)
    config = Conf(configuration)

    cr_file = os.path.abspath(
        os.path.join(scenario_path,
                     config.scenario_name + '.cr.xml'))

    output_folder = os.path.dirname(cr_file)
    print("Reading file:", cr_file, " Outputing to folder:", output_folder)

    scenario, _ = CommonRoadFileReader(cr_file).open()
    planning_problems = create_planning_problem_DEU_Ffb(configuration)
    initial_state = planning_problems.find_planning_problem_by_id(0).initial_state
    goal_region = planning_problems.find_planning_problem_by_id(0).goal
    early_goal_shape = ShapelyPolygon([(55,7),(60,8),(60,4),(55,3),(55,7)])

    config.scenarios_path = None
    wrapper = ScenarioWrapper.init_from_scenario(config, scenario_path, cr_map_file=cr_file)
    sumo_sim = SumoSimulation()
    try:
        sumo_sim.initialize(config, wrapper, planning_problems)
    except:
        sumo_sim.stop()
    planner = Planner(initial_state,
                      vehicle_shape=Rectangle(configuration.get('vehicle_length'),
                                              configuration.get('vehicle_width')),
                      goal_point=goal_region.state_list[0].position.center,
                      reference_speed=configuration.get('reference_speed'),
                      max_acceleration=configuration.get('max_acceleration'),
                      max_deceleration=configuration.get('max_deceleration'),
                      time_horizon=configuration.get('planning_horizon'))
    sensor = Sensor(initial_state.position,
                    field_of_view=configuration.get(
                        'field_of_view_degrees')*2*np.pi/360,
                    min_resolution=configuration.get('min_resolution'),
                    view_range=configuration.get('view_range'))
    occ_track = Occlusion_tracker(scenario,
                                  min_vel=configuration.get('min_velocity'),
                                  max_vel=configuration.get('max_velocity'),
                                  min_shadow_area=configuration.get(
                                      'min_shadow_area'),
                                  prediction_horizon=configuration.get(
                                      'prediction_horizon'),
                                  tracking_enabled=configuration.get('tracking_enabled'))
    previous_plan = None
    previous_trajectory = None
    sensor_views = []
    percieved_scenarios = []
    sensor_range_shapes = []
    goal_reached = False
    simulation_failed = False

    for sumo_time_step in range(config.simulation_steps):
        if sumo_time_step == 1:
            occ_track.reset(new_time=sumo_time_step)
        try:
            # Get scenario and ego vehicle from SUMO and building that gets removed
            commonroad_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
            ego_vehicle = list(sumo_sim.ego_vehicles.values())[0]
            #add_building_DEU_Ffb(commonroad_scenario)

            # Construct percieved scenario
            percieved_scenario = copy.deepcopy(commonroad_scenario)
            for obstacle in percieved_scenario.obstacles:
                percieved_scenario.remove_obstacle(obstacle)
            # Update the time step, sumo resets it. Needs to be done after removing obstacles.
            for obstacle in commonroad_scenario.dynamic_obstacles:
                obstacle.initial_state.time_step = ego_vehicle.current_state.time_step

            # Update the sensor and get the sensor view and the list of observed obstacles
            sensor.update(ego_vehicle.current_state)
            sensor_view = sensor.get_sensor_view(commonroad_scenario)
            observed_obstacles, _ = sensor.get_observed_obstacles(sensor_view, commonroad_scenario)
            percieved_scenario.add_objects(observed_obstacles)

            # Update the tracker with the new sensor view and get the prediction for the shadows
            occ_track.update(sensor_view, ego_vehicle.current_state.time_step)
            shadow_obstacles = occ_track.get_dynamic_obstacles(percieved_scenario)
            percieved_scenario.add_objects(shadow_obstacles)

            # Update the planner and plan a trajectory
            add_no_stop_zone_DEU_Ffb(percieved_scenario, ego_vehicle.current_state.time_step + configuration.get('planning_horizon'))
            planner.update(ego_vehicle.current_state)
            collision_free_trajectory = planner.plan(percieved_scenario)
            if collision_free_trajectory:
                ego_trajectory = copy.deepcopy(collision_free_trajectory.trajectory.state_list)
                # Planned ego trajectory starts at t=1 in SUMO
                for time_from_current, state in enumerate(ego_trajectory):
                    state.time_step = 1 + time_from_current
                ego_vehicle.set_planned_trajectory(ego_trajectory)
                previous_plan = ego_trajectory
            elif previous_plan:
                # Use previous plan, pop first state and append the last again
                previous_plan.pop(0)
                for time_from_current, state in enumerate(previous_plan):
                    state.time_step = 1 + time_from_current
                last_state = copy.deepcopy(previous_plan[-1])
                last_state.time_step = 1 + last_state.time_step
                previous_plan.append(last_state)
                ego_vehicle.set_planned_trajectory(previous_plan)
            else:
                print(sumo_time_step, "no collision free trajectory found!")

            commonroad_ego_vehicle = ego_vehicle.get_dynamic_obstacle()
            commonroad_ego_vehicle.initial_state = ego_vehicle.current_state
            if collision_free_trajectory:
                commonroad_ego_vehicle.prediction = collision_free_trajectory
                previous_trajectory = collision_free_trajectory
            else:
                commonroad_ego_vehicle.prediction = previous_trajectory
            percieved_scenario.add_objects(commonroad_ego_vehicle)
            percieved_scenarios.append(percieved_scenario)
            sensor_views.append(sensor_view)
            sensor_range_shape = ShapelyPolygon(sensor.view_vertices([]))
            sensor_range_shapes.append(sensor_range_shape.buffer(0))

            sumo_sim.simulate_step()
            if early_goal_shape.intersects(ShapelyPoint(ego_vehicle.current_state.position)):
                print("Reached goal at time step: ", sumo_time_step)
                goal_reached = True
                break
        except:
            print("Simulate: Failed at time step " + str(sumo_time_step) + "in " + scenario_path)
            print(traceback.format_exc())
            simulation_failed = True
            break

    sumo_sim.stop()
    print("Done simulating")

    viz = Visualizer()
    for ego_vehicle in sumo_sim.ego_vehicles.values():
        viz.save_animation(percieved_scenarios,
                           sensor_views,
                           commonroad_ego_vehicle.obstacle_id,
                           scenario_path + "/" + config.scenario_name + ".mp4")

    simulation_log(percieved_scenarios, sensor_range_shapes, output_folder, goal_reached, simulation_failed, sumo_time_step)
    return commonroad_ego_vehicle, percieved_scenarios, sensor_views
