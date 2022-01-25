import copy
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle

from planner import Planner
from sensor import Sensor
from occlusion_tracker import Occlusion_tracker

import yaml

from utilities import add_no_stop_zone_DEU_Ffb

# Create new scenario with new vehicles at next time step
def step_scenario(scenario):
    new_scenario = copy.deepcopy(scenario)
    for vehicle in scenario.dynamic_obstacles:
        new_scenario.remove_obstacle(vehicle)
        if len(vehicle.prediction.trajectory.state_list) > 1:
            stepped_vehicle = step_vehicle(vehicle)
            new_scenario.add_objects(stepped_vehicle)
    return new_scenario


def step_vehicle(vehicle):
    initial_state = vehicle.prediction.trajectory.state_list.pop(0)
    trajectory = Trajectory(1 + initial_state.time_step,
                            vehicle.prediction.trajectory.state_list)
    return DynamicObstacle(vehicle.obstacle_id,
                           vehicle.obstacle_type,
                           vehicle.obstacle_shape,
                           initial_state,
                           TrajectoryPrediction(trajectory, vehicle.obstacle_shape))


def step_simulation(scenario, configuration):
    driven_state_list = []
    percieved_scenarios = []
    sensor_views = []

    ego_shape = Rectangle(configuration.get('vehicle_length'),
                          configuration.get('vehicle_width'))
    ego_initial_state = State(position=np.array([configuration.get('initial_state_x'),
                                                 configuration.get('initial_state_y')]),
                              orientation=configuration.get(
                                  'initial_state_orientation'),
                              velocity=configuration.get(
                                  'initial_state_velocity'),
                              time_step=0)
    ego_vehicle = DynamicObstacle(scenario.generate_object_id(),
                                  ObstacleType.CAR, ego_shape,
                                  ego_initial_state)

    sensor = Sensor(ego_vehicle.initial_state.position,
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

    planner = Planner(ego_vehicle.initial_state,
                      vehicle_shape=ego_vehicle.obstacle_shape,
                      goal_point=[configuration.get('goal_point_x'),
                                  configuration.get('goal_point_y')],
                      reference_speed=configuration.get('reference_speed'),
                      max_acceleration=configuration.get('max_acceleration'),
                      max_deceleration=configuration.get('max_deceleration'),
                      time_horizon=configuration.get('planning_horizon'))
    simulation_steps = configuration.get('simulation_duration')
    for step in range(simulation_steps+1):
        # Start with an empty percieved scenario
        percieved_scenario = copy.deepcopy(scenario)
        for obstacle in percieved_scenario.obstacles:
            percieved_scenario.remove_obstacle(obstacle)

        # Update the sensor and get the sensor view and the list of observed obstacles
        sensor.update(ego_vehicle.initial_state) # initial_state is current state
        sensor_view = sensor.get_sensor_view(scenario)
        observed_obstacles, _ = sensor.get_observed_obstacles(sensor_view, scenario)
        percieved_scenario.add_objects(observed_obstacles)

        # Update the tracker with the new sensor view and get the prediction for the shadows
        occ_track.update(sensor_view, ego_vehicle.initial_state.time_step)
        shadow_obstacles = occ_track.get_dynamic_obstacles(percieved_scenario)
        percieved_scenario.add_objects(shadow_obstacles)

        # Update the planner and plan a trajectory
        #if
        add_no_stop_zone_DEU_Ffb(percieved_scenario, step + configuration.get('planning_horizon'), configuration.get('safety_margin'))
        planner.update(ego_vehicle.initial_state)
        collision_free_trajectory = planner.plan(percieved_scenario)
        if collision_free_trajectory:
            ego_vehicle.prediction = collision_free_trajectory
        # else, if no trajectory found, keep previous collision free trajectory

        # Add the ego vehicle to the perceived scenario
        percieved_scenario.add_objects(ego_vehicle)

        percieved_scenarios.append(percieved_scenario)
        sensor_views.append(sensor_view)
        driven_state_list.append(ego_vehicle.initial_state)

        ego_vehicle = step_vehicle(ego_vehicle)
        scenario = step_scenario(scenario)

    # Set initial_state to initial state and not current
    ego_vehicle.initial_state = driven_state_list.pop(0)
    driven_trajectory = Trajectory(0, driven_state_list)
    driven_trajectory_pred = TrajectoryPrediction(
        driven_trajectory, ego_vehicle.obstacle_shape)
    ego_vehicle.prediction = driven_trajectory_pred
    return ego_vehicle, percieved_scenarios, sensor_views
