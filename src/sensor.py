
import numpy as np
import math

import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.pycrcc import CollisionObject
from commonroad.geometry.shape import Polygon
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad.scenario.obstacle import StaticObstacle
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker, create_collision_object

import matplotlib.pyplot as plt

from shapely.geometry import Polygon as ShapelyPolygon
from utilities import ShapelyPolygon2Polygon, polygon_union


class Sensor:
    def __init__(self,
                 position,
                 orientation=0,
                 field_of_view=2*np.pi,
                 min_resolution=1,
                 view_range=50):
        assert len(position) == 2
        assert field_of_view <= 2*np.pi
        self.position = position
        self.orientation = orientation
        self.field_of_view = field_of_view
        self.min_resolution = min_resolution
        self.range = view_range
        self.angle_resolution = math.atan(min_resolution/view_range)

    def update(self, state):
        assert isinstance(state, State)
        self.position = state.position
        self.orientation = state.orientation

    def get_extended_sensor_view_and_obstacles(self, scenario):
        sensor_view = self.get_sensor_view(scenario)
        [observed_obstacles, observed_shapes] = self.get_observed_obstacles(sensor_view, scenario)
        extended_sensor_view = sensor_view
        for observed_shape in observed_shapes:
            unions = polygon_union([extended_sensor_view, observed_shape])
            extended_sensor_view = unions[0] # This has to be fixed
        return [extended_sensor_view, observed_obstacles]

    def get_sensor_view(self, scenario):
        assert isinstance(scenario, Scenario)
        obstacles = []
        for obstacle in scenario.obstacles:
            obstacle_shape = obstacle.occupancy_at_time(
                obstacle.initial_state.time_step).shape
            collision_object = create_collision_object(obstacle_shape)
            obstacles.append(collision_object)

        sensor_view = ShapelyPolygon(self.view_vertices(obstacles))
        if not sensor_view.is_valid:
            sensor_view = sensor_view.buffer(0)
        assert sensor_view.is_valid
        return sensor_view

    def get_observed_obstacles(self, sensor_view, scenario):
        assert isinstance(sensor_view, ShapelyPolygon)
        assert isinstance(scenario, Scenario)

        observed_obstacles = []
        observed_shapes = []
        for obstacle in scenario.obstacles:
            obstacle_shapely = obstacle.occupancy_at_time(
                obstacle.initial_state.time_step).shape._shapely_polygon
            if sensor_view.overlaps(obstacle_shapely):
                observed_obstacles.append(obstacle)
                observed_shapes.append(obstacle_shapely)

        return [observed_obstacles, observed_shapes]

    def view_vertices(self, obstacles):
        assert all(isinstance(ob, CollisionObject) for ob in obstacles)
        vertices = []

        has_omnidirectional_view = self.field_of_view > 2*np.pi - self.angle_resolution
        if not has_omnidirectional_view:
            vertices.append(self.position)

        num_ray_angles = int(self.field_of_view/self.angle_resolution) + 1
        ray_angles = self.orientation + np.linspace(-self.field_of_view/2,
                                                    self.field_of_view/2,
                                                    num_ray_angles)

        cc = pycrcc.CollisionChecker()
        for obstacle in obstacles:
            cc.add_collision_object(obstacle)

        for angle in ray_angles:
            ray_end = [self.position[0] + self.range*np.cos(angle),
                       self.position[1] + self.range*np.sin(angle)]

            ray_hits = cc.raytrace(self.position[0], self.position[1],
                                   ray_end[0], ray_end[1], False)

            if not ray_hits:
                vertices.append(ray_end)
            else:
                closest_hit = ray_end
                closest_hit_distance = self.range
                for ray_hit in ray_hits:
                    hit_in = ray_hit[0:2]
                    hit_out = ray_hit[2:]
                    distance_to_hit_in = np.hypot(hit_in[0] - self.position[0],
                                                  hit_in[1] - self.position[1])
                    distance_to_hit_out = np.hypot(hit_out[0] - self.position[0],
                                                   hit_out[1] - self.position[1])
                    if (distance_to_hit_in < closest_hit_distance):
                        closest_hit_distance = distance_to_hit_in
                        closest_hit = hit_in
                    if (distance_to_hit_out < closest_hit_distance):
                        closest_hit_distance = distance_to_hit_out
                        closest_hit = hit_out
                vertices.append(closest_hit)

        vertices.append(vertices[0])

        return np.array(vertices)
