import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry import GeometryCollection as ShapelyGeometryCollection
from shapely.geometry import MultiPoint
from shapely.geometry import LineString
from shapely.geometry import Point

from commonroad.geometry.shape import Circle, Rectangle, Polygon as CommonRoadPolygon
from commonroad.prediction.prediction import Occupancy, SetBasedPrediction
from commonroad.scenario.trajectory import State
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle, StaticObstacle
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem, GoalRegion
from commonroad.common.util import Interval, AngleInterval
from commonroad.scenario.scenario import Lanelet
from commonroad_dc.collision.visualization.draw_dispatch import draw_object

import yaml

from datetime import datetime


def ShapelyPolygon2Polygon(shapely_polygon):
    assert isinstance(shapely_polygon, ShapelyPolygon)
    assert hasattr(shapely_polygon, 'exterior')
    assert hasattr(shapely_polygon.exterior, 'xy')
    vertices = np.array(list(zip(*shapely_polygon.exterior.xy)))
    return CommonRoadPolygon(vertices)


def Lanelet2ShapelyPolygon(lanelet):
    assert isinstance(lanelet, Lanelet)
    right = lanelet.right_vertices
    left = np.flip(lanelet.left_vertices, axis=0)
    lanelet_boundary = np.concatenate((right, left, np.array([right[0]])))

    lanelet_shapely = ShapelyPolygon(lanelet_boundary)
    if not lanelet_shapely.is_valid:
        lanelet_shapely = lanelet_shapely.buffer(0)
        if not lanelet_shapely.is_valid:
            print("Note: Shape of lanelet", lanelet.lanelet_id,
                  "is not valid, creating valid shape with convex hull of lane boundary.")
            lanelet_shapely = MultiPoint(lanelet_boundary).convex_hull
            assert lanelet_shapely.is_valid, "Failed to convert lanelet to polygon"
    return lanelet_shapely


def polygon_diff(polygonA, polygonB):
    difference_undef = polygonA.difference(polygonB)
    polygon_list = filter_polygons(difference_undef)
    return polygon_list


def polygon_intersection(polygonA, polygonB):
    intersection_undef = polygonA.intersection(polygonB)
    polygon_list = filter_polygons(intersection_undef)
    return polygon_list


def polygon_union(polygons):
    current_polygon = ShapelyPolygon()
    for polygon in polygons:
        current_polygon = current_polygon.union(polygon)
    polygon_list = filter_polygons(current_polygon)
    return polygon_list


def filter_polygons(input):
    polygonEmpty = ShapelyPolygon()
    polygon_list = []
    if isinstance(input, ShapelyPolygon):
        if input != polygonEmpty:
            assert input.is_valid
            polygon_list.append(input)
    elif isinstance(input, ShapelyMultiPolygon):
        for polygon in input.geoms:
            if polygon != polygonEmpty:
                assert polygon.is_valid
                polygon_list.append(polygon)
    elif isinstance(input, ShapelyGeometryCollection):
        for element in input.geoms:
            if isinstance(element, ShapelyPolygon):
                if element != polygonEmpty:
                    assert element.is_valid
                    polygon_list.append(element)
    return polygon_list


def cut_line(line, bottom_dis, top_dis):
    # Check that the two distances are in the right order
    assert bottom_dis <= top_dis

    # Check that the distances are inside the limits
    assert bottom_dis >= 0.0
    if top_dis > line.length:
        top_dis = line.length

    # Line cutting algorithm
    points = []
    points.append(line.interpolate(bottom_dis).coords[0])
    for edge in line.coords:
        edge_dis = line.project(Point(edge))
        if bottom_dis < edge_dis:
            if edge_dis < top_dis:
                points.append(edge)
            else:
                break
    points.append(line.interpolate(top_dis).coords[0])
    return points


def add_building_DEU_Ffb(scenario):
    scenario.add_objects(StaticObstacle(scenario.generate_object_id(),
                                        ObstacleType.BUILDING,
                                        Rectangle(10, 15),
                                        State(position=np.array([83.6288, -11.5553]),
                                              orientation=0, time_step=0)))


def add_no_stop_zone_DEU_Ffb(scenario, planning_horizon, safety_margin=5):
    lanelet_1 = scenario.lanelet_network.find_lanelet_by_id(
        49602).convert_to_polygon()
    lanelet_2 = scenario.lanelet_network.find_lanelet_by_id(
        49600).convert_to_polygon()
    lanelet_3 = scenario.lanelet_network.find_lanelet_by_id(
        49598).convert_to_polygon()
    lanelet_4 = scenario.lanelet_network.find_lanelet_by_id(
        49596).convert_to_polygon()
    horizontal_lanes = polygon_union(
        [lanelet_1._shapely_polygon, lanelet_2._shapely_polygon])
    vertical_lanes = polygon_union(
        [lanelet_3._shapely_polygon, lanelet_4._shapely_polygon])
    no_stop_shapely = polygon_intersection(
        horizontal_lanes[0], vertical_lanes[0])
    no_stop_polygon = ShapelyPolygon2Polygon(no_stop_shapely[0].convex_hull.buffer(safety_margin))

    dummy_state = State(position=np.array(
        [0, 0]), orientation=0, velocity=0, time_step=planning_horizon)
    occupancy = Occupancy(planning_horizon, no_stop_polygon)
    prediction = SetBasedPrediction(planning_horizon, [occupancy])

    no_stop_object = DynamicObstacle(scenario.generate_object_id(),
                                     ObstacleType.ROAD_BOUNDARY,
                                     no_stop_polygon,
                                     dummy_state,
                                     prediction)

    scenario.add_objects(no_stop_object)


def create_planning_problem_DEU_Ffb(configuration, planning_id=0):
    initial_state = State(position=np.array([configuration.get('initial_state_x'),
                                             configuration.get('initial_state_y')]),
                          orientation=configuration.get(
                              'initial_state_orientation'),
                          velocity=configuration.get('initial_state_velocity'),
                          time_step=0,
                          yaw_rate=0,
                          slip_angle=0)

    goal_state = State(position=Circle(2, np.array([configuration.get('goal_point_x'),
                                           configuration.get('goal_point_y')])),
                       orientation=AngleInterval(-np.pi, np.pi-0.0000000000001),
                       velocity=Interval(0, 20),
                       time_step=Interval(0, configuration.get('simulation_duration')))

    goal_region = GoalRegion([goal_state])
    planning_problem = PlanningProblem(planning_id, initial_state, goal_region)
    return PlanningProblemSet([planning_problem])

def clamp(x):
    return max(0, min(x, 255))

def rgb2hex(r,g,b):
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))

def simulation_log(percieved_scenarios, sensor_range_shapes, directory, goal_reached, simulation_failed, Simulation_length):
    # Calculate the average occluded area
    occluded_areas = []
    for scenario, sensor_range_shape in zip(percieved_scenarios, sensor_range_shapes):
        occluded_areas.append(calculate_occluded_area(scenario, sensor_range_shape))
    average_occluded_area = sum(occluded_areas)/len(occluded_areas)

    # Save the logs in a file
    dict_file = {'Time': datetime.now(),
                 'Average occluded area': average_occluded_area,
                 'Goal reached': goal_reached,
                 'Simulation failed': simulation_failed,
                 'Simulation length': Simulation_length,
                 'List of occluded areas': occluded_areas}
    with open(directory + '/simulation_log.yaml', 'w') as file:
        documents = yaml.dump(dict_file, file, sort_keys=False)

def calculate_occluded_area(scenario, sensor_range_shape):
    # Find the shadow objects
    shadows = scenario.obstacles_by_role_and_type(obstacle_type = ObstacleType.UNKNOWN)

    # Get all the polygons
    polygons = []
    for shadow in shadows:
        shadow_polygon = ShapelyPolygon(shadow.obstacle_shape.vertices)
        intersection = polygon_intersection(shadow_polygon, sensor_range_shape)
        if intersection:
            polygons.append(intersection[0])

    # Combine all the polygons
    polygon_list = polygon_union(polygons)

    # Add all the areas
    total_area = 0
    for polygon in polygon_list:
        total_area = total_area + polygon.area

    # Return the area
    return total_area

def scenario_generation_log(directory, traffic_density, min_initial_vel, max_initial_vel, distance_to_intersection):
    dict_file = {'Time': datetime.now(),
                 'Traffic density': traffic_density,
                 'Minimum initial velocity': min_initial_vel,
                 'Maximum initial velicity': max_initial_vel,
                 'Distance to intersection': distance_to_intersection}
    with open(directory + '/scenario_generation_log.yaml', 'w') as file:
        documents = yaml.dump(dict_file, file, sort_keys=False)
