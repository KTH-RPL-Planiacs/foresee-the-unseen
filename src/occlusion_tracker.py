
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import LinearRing
from shapely.ops import substring

from commonroad.scenario.scenario import Scenario, Lanelet
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import State
from commonroad.prediction.prediction import SetBasedPrediction, Occupancy
from commonroad_dc.collision.visualization.draw_dispatch import draw_object

from utilities import Lanelet2ShapelyPolygon, ShapelyPolygon2Polygon, polygon_intersection, polygon_diff, polygon_union, cut_line

import time

class Shadow:
    def __init__(self,
                 polygon,
                 lane):
        self.polygon = polygon
        self.lane = lane
        self.center_line = ShapelyLineString(self.lane.center_vertices)
        self.right_line = ShapelyLineString(self.lane.right_vertices)
        self.left_line = ShapelyLineString(self.lane.left_vertices)
        self.lane_shapely = Lanelet2ShapelyPolygon(lane)

    def expand(self, dist):
        self.polygon = self.__get_next_occ(self.polygon, dist)

    def get_occupancy_set(self, time_step, dt, max_vel, prediction_horizon, planning_horizon = 100):
        dist = dt*max_vel
        occupancy_set = []
        #pred_polygon_shapely = self.polygon

        # Calculate the right and left projections
        right_projections = []
        left_projections = []
        for edge in self.polygon.exterior.coords:
            right_projections.append(self.right_line.project(ShapelyPoint(edge)))
            left_projections.append(self.left_line.project(ShapelyPoint(edge)))

        # Calculate the edges of the current shadow
        bottom_right = min(right_projections)
        bottom_left = min(left_projections)
        top_right = max(right_projections)
        top_left = max(left_projections)

        for i in range(prediction_horizon):
            # Extend the top edges without overpasing the length of the lane
            new_top_right = max(top_right + dist, self.right_line.project(self.left_line.interpolate(top_left + dist)))
            new_top_left = max(top_left + dist, self.left_line.project(self.right_line.interpolate(top_right + dist)))
            top_right = new_top_right
            top_left = new_top_left
            top_right = min(top_right, self.right_line.length)
            top_left = min(top_left, self.left_line.length)

            pred_polygon_shapely = self.__build_polygon(bottom_right, bottom_left, top_right, top_left)
            pred_polygon = ShapelyPolygon2Polygon(pred_polygon_shapely)
            occupancy = Occupancy(time_step+i+1, pred_polygon)
            occupancy_set.append(occupancy)

        # Populate the rest of the planning horizon with the last prediction
        for i in range(prediction_horizon, planning_horizon):
            occupancy = Occupancy(time_step+i+1, pred_polygon)
            occupancy_set.append(occupancy)

        return occupancy_set

    def __get_next_occ(self, poly, dist):
        smallest_projection = 999999
        for edge in poly.exterior.coords:
            projection = self.center_line.project(ShapelyPoint(edge))
            if projection < smallest_projection:
                smallest_projection = projection
            if smallest_projection <= 0:
                break
        poly = poly.buffer(dist, join_style=1)
        intersection = polygon_intersection(poly, self.lane_shapely)
        poly = intersection[0] #This has to be fixed

        if smallest_projection > 0:
            sub_center_line = substring(self.center_line, 0, smallest_projection)
            left_side = sub_center_line.parallel_offset(2.8, 'left')
            right_side = sub_center_line.parallel_offset(2.8, 'right')
            Area_to_substract = ShapelyPolygon(np.concatenate((np.array(left_side.coords), np.array(right_side.coords))))
            diff = polygon_diff(poly, Area_to_substract)
            poly = diff[0] #This has to be fixed

        return poly

    def __build_polygon(self, bottom_right, bottom_left, top_right, top_left):
        # Cut the left and right lines with the top and bottom points
        right_side = cut_line(self.right_line, bottom_right, top_right)
        left_side = cut_line(self.left_line, bottom_left, top_left)

        # Build the polygon
        left_side.reverse()
        shadow_boundary = right_side + left_side + [right_side[0]]
        shadow_shapely = ShapelyPolygon(shadow_boundary)

        shadow_shapely = shadow_shapely.buffer(0)

        assert shadow_shapely.is_valid
        assert not shadow_shapely.is_empty
        if not isinstance(shadow_shapely, ShapelyPolygon):#, "shadow_boundary: " + str(shadow_boundary)
            print(type(shadow_shapely))
            print("Not instance")
            assert LinearRing(shadow_boundary).is_valid
        return shadow_shapely

class Occlusion_tracker:
    def __init__(self,
                 scenario,
                 initial_time_step=0,
                 min_vel=0,
                 max_vel=1,
                 #min_acc=-1,
                 #max_acc=1,
                 min_shadow_area = 1,
                 initial_sensor_view = ShapelyPolygon(),
                 prediction_horizon = 10,
                 tracking_enabled = True):
        self.time_step = initial_time_step
        self.dt = scenario.dt
        self.min_vel = min_vel
        self.max_vel = max_vel
        #self.min_acc = min_acc
        #self.max_acc = max_acc
        self.min_shadow_area = min_shadow_area
        self.shadows = []
        self.prediction_horizon = prediction_horizon
        self.tracking_enabled = tracking_enabled

        ## Find the initial lanelets
        initial_lanelets = []
        for lanelet in scenario.lanelet_network.lanelets:
            if lanelet.predecessor == []:
                initial_lanelets.append(lanelet)

        ## Generate lanes (Collection of lanelets from start to end of the scenario)
        lanes = []
        for lanelet in initial_lanelets:
            current_lanes, _ = Lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, scenario.lanelet_network, max_length=500)
            for lane in current_lanes:
                lanes.append(lane)
        self.lanes = lanes

        # Calculate the first "view"
        for lane in self.lanes:
            lanelet_shapely = Lanelet2ShapelyPolygon(lane)
            shadow_polygons = polygon_diff(lanelet_shapely, initial_sensor_view)
            for shadow_polygon in shadow_polygons:
                if shadow_polygon.area >= self.min_shadow_area:
                    current_shadow = Shadow(shadow_polygon, lane)
                    self.shadows.append(current_shadow)

        # Calculate the first occluded area
        self.accumulated_occluded_area = 0

        #plt.figure()
        #plot(scenario, shadows=self.shadows)
        #plt.show()

    def update(self, sensor_view, new_time):
        if self.tracking_enabled == True:
            self.update_tracker(sensor_view, new_time)
        else:
            self.reset(sensor_view, new_time)

    def update_tracker(self, sensor_view, new_time):
        assert(new_time>=self.time_step)
        time_diff = new_time - self.time_step
        # Update the time
        self.time_step = new_time
        # Expand all the shadows
        for shadow in self.shadows:
            shadow.expand(self.dt*self.max_vel*time_diff)

        # Intersect them with the current sensorview
        new_shadows = []
        for shadow in self.shadows:
            intersections = polygon_diff(shadow.polygon, sensor_view)
            if not intersections:
                pass
            else:
                for intersection in intersections:
                    assert intersection.is_valid
                    assert not intersection.is_empty
                    if intersection.area >= self.min_shadow_area:
                        new_shadows.append(Shadow(intersection, shadow.lane))
        self.shadows = new_shadows

        # Update the accumulated occluded area
        self.accumulated_occluded_area = self.accumulated_occluded_area + self.get_currently_occluded_area()

    def reset(self, sensor_view=ShapelyPolygon(), new_time=0):
        # Update the time
        self.time_step = new_time
        # Reset all the shadows
        new_shadows = []
        for lane in self.lanes:
            lanelet_shapely = Lanelet2ShapelyPolygon(lane)
            shadow_polygons = polygon_diff(lanelet_shapely, sensor_view)
            for shadow_polygon in shadow_polygons:
                if shadow_polygon.area >= self.min_shadow_area:
                    #print(shadow_polygon.area)
                    #plt.figure()
                    #plot(shapelyPolygons=[shadow_polygon])
                    #plt.show()
                    current_shadow = Shadow(shadow_polygon, lane)
                    new_shadows.append(current_shadow)
        self.shadows = new_shadows

        # Update the accumulated occluded area
        self.accumulated_occluded_area = self.accumulated_occluded_area + self.get_currently_occluded_area()

    def get_dynamic_obstacles(self, scenario):
        dynamic_obstacles = []
        for shadow in self.shadows:
            occupancies = []
            occupancy_set = shadow.get_occupancy_set(self.time_step, self.dt, self.max_vel, self.prediction_horizon)
            obstacle_id = scenario.generate_object_id()
            obstacle_type = ObstacleType.UNKNOWN
            obstacle_shape = ShapelyPolygon2Polygon(shadow.polygon)
            obstacle_initial_state = State(position = np.array([0,0]),
                                           velocity = self.max_vel,
                                           orientation = 0,
                                           time_step = self.time_step)
            obstacle_prediction = SetBasedPrediction(self.time_step+1, occupancy_set)
            dynamic_obstacle = DynamicObstacle(obstacle_id,
                            obstacle_type,
                            obstacle_shape,
                            obstacle_initial_state,
                            obstacle_prediction)
            dynamic_obstacles.append(dynamic_obstacle)
        return dynamic_obstacles

    def get_currently_occluded_area(self):
        # Get all the shadow polygons:
        list_of_shadows = []
        for shadow in self.shadows:
            list_of_shadows.append(shadow.polygon)

        # Calculate the union:
        polygon_list = polygon_union(list_of_shadows)

        # Add up all the areas:
        currently_occluded_area = 0
        for polygon in polygon_list:
            currently_occluded_area = currently_occluded_area + polygon.area

        # Return the currently occluded area:
        return currently_occluded_area
