import matplotlib.pyplot as plt
from matplotlib import animation
import yaml
from commonroad_dc.collision.visualization.draw_dispatch import draw_object
from commonroad.scenario.obstacle import ObstacleType

from utilities import ShapelyPolygon2Polygon, rgb2hex


class Visualizer:
    def __init__(self):
        self.animation = None
        self.fig = None

    def _update(self, time_step, scenarios, sensor_views, ego_id):
        plt.clf()
        scenario = scenarios[time_step]
        ego_vehicle = scenario.obstacle_by_id(ego_id)
        self.plot(scenario=scenario,
                  ego_vehicle=ego_vehicle,
                  sensor_view=sensor_views[time_step],
                  time_begin=time_step)
        plt.autoscale()
        plt.axis('equal')
        plt.xlim(0, 100)
        plt.ylim(-50, 50)

    def _animate_scenarios(self, scenarios, sensor_views, ego_id):
        self.fig = plt.figure(figsize=(10, 10))
        self.animation = animation.FuncAnimation(self.fig,
                                                 self._update,
                                                 fargs=[scenarios,
                                                        sensor_views,
                                                        ego_id],
                                                 frames=len(scenarios),
                                                 interval=round(1000*scenarios[0].dt))
        # plt.close(self.fig)

    def save_animation(self, scenarios, sensor_views, ego_id, file_name='videos/animation.mp4'):
        self._animate_scenarios(scenarios, sensor_views, ego_id)
        self.animation.save(file_name)

    def save_snapshot_plot(self, scenario, sensor_view, ego_id, file_name='figs/snapshot.png', xylimits=[40,100,-20, 40], xticks=None, yticks=None, fig_width=6):
        yx_ratio = (xylimits[3]-xylimits[2])/(xylimits[1]-xylimits[0])
        plt.figure(figsize=(fig_width, fig_width*yx_ratio))
        ego_vehicle = scenario.obstacle_by_id(ego_id)
        self.plot(scenario=scenario,
                  sensor_view=sensor_view,
                  ego_vehicle=ego_vehicle,
                  time_begin=ego_vehicle.initial_state.time_step)
        plt.axis('scaled')
        plt.xlim(xylimits[0], xylimits[1])
        plt.ylim(xylimits[2],xylimits[3])
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.tight_layout()
        plt.savefig(file_name)

    def save_velocity_comparison_plot(self, vehicles, labels, file_name='figs/velocity_plot.png'):
        plt.figure(figsize=(6, 2))
        for idx, vehicle in enumerate(vehicles):
            velocities = [round(vehicle.initial_state.velocity, 2)]
            time = [round(vehicle.initial_state.time_step/10, 2)]
            for state in vehicle.prediction.trajectory.state_list:
                velocities.append(round(state.velocity, 2))
                time.append(round(state.time_step/10, 2))
            plt.plot(time, velocities, label=labels[idx])
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.ylim(0, 10)
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(file_name)

    def save_occlusion_comparison_plot(self, simulation_log_1, simulation_log_2, labels, file_name='figs/occ_area_comparison.png'):
        with open(simulation_log_1) as file:
            simulation_log_1 = yaml.load(file, Loader=yaml.FullLoader)
        occ_areas_1 = simulation_log_1['List of occluded areas']

        with open(simulation_log_2) as file:
            simulation_log_2 = yaml.load(file, Loader=yaml.FullLoader)
        occ_areas_2 = simulation_log_2['List of occluded areas']

        time = []
        i=1
        for element in occ_areas_1:
            time.append(i/10)
            i = i+1

        plt.figure(figsize=(6, 2))
        plt.plot(time, occ_areas_1, label=labels[0])
        plt.plot(time, occ_areas_2, label=labels[1])
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Occluded area [m^2]')
        plt.tight_layout()
        plt.savefig(file_name)


    def draw_shadows(self, shadows, time_begin, time_horizon):
        # We use red
        R = int(255)
        G = int(0)
        B = int(0)

        # Calculate the hex
        color = rgb2hex(100,0,0)

        # Draw the first location of the shadows
        shadow_draw_params = {'time_begin': time_begin,
                              'time_end':time_begin + 1,
                              'dynamic_obstacle': {'shape': {'opacity': 1,
                                                             'facecolor': color,
                                                             'edgecolor': color},
                                                   'draw_shape': True,
                                                   'occupancy': {'draw_occupancies': -1}
                                                   }
                              }
        draw_object(shadows, draw_params=shadow_draw_params)

        # Draw the shadow predictions
        for i in reversed(range(time_horizon)):
            tint_factor = 0.005**(1/(i+1))
            Ri = int(R + (255 - R) * tint_factor)
            Gi = int(G + (255 - G) * tint_factor)
            Bi = int(B + (255 - G) * tint_factor)
            color = rgb2hex(Ri,Gi,Bi)
            shadow_predictions_draw_params = {'time_begin': time_begin+i,
                                              'time_end':time_begin+i+1,
                                              'dynamic_obstacle': {'draw_shape': False,
                                                                   'occupancy': {'draw_occupancies': 1,
                                                                                 'shape': {'opacity': 1,
                                                                                           'facecolor': color,
                                                                                           'edgecolor': color,
                                                                                           'zorder': 1}
                                                                                 }
                                                                   }
                                              }
            draw_object(shadows, draw_params=shadow_predictions_draw_params)

    def plot(self,
             scenario=None,
             time_begin=0,
             time_end=500,
             ego_vehicle=None,
             obstacles=None,
             lanes=None,
             polygons=None,
             shapelyPolygons=None,
             shadows=None,
             sensor_view=None,
             goal_region=None):

        scenario_draw_params = {'time_begin': time_begin,
                                'time_end': time_end,
                                'scenario': {'static_obstacle': {'opacity': 1,
                                                                 'facecolor': '#808080',
                                                                 'edgecolor': '#000000',
                                                                 'zorder': 1},
                                             'dynamic_obstacle': {'shape': {'opacity': 1,
                                                                            'facecolor': '#ffff00',
                                                                            'edgecolor': '#000000',
                                                                            'zorder': 100},
                                                                  'occupancy': {'draw_occupancies': 1, #0=set, 1=set&traj, 2=None
                                                                                'shape': {'opacity': 0.2,
                                                                                          'facecolor': '#ffff00',
                                                                                          'edgecolor': '#ffff00',
                                                                                          'zorder': 100}},
                                                                  'trajectory': {'draw_trajectory': False}
                                                                  },
                                             'lanelet_network': {'lanelet': {'left_bound_color': '#000000',
                                                                             'right_bound_color': '#000000',
                                                                             'draw_center_bound': False,
                                                                             'fill_lanelet': False,
                                                                             'draw_start_and_direction': False}}
                                             }
                               }

        if sensor_view is not None:
            draw_params = {'shape': {'opacity': 1,
                                     'facecolor': '#E5E5FF',
                                     'edgecolor': '#E5E5FF',
                                     'zorder': 1}}
            draw_object(ShapelyPolygon2Polygon(
                sensor_view), draw_params=draw_params)
        if scenario is not None:
            if ego_vehicle is not None:
                scenario.remove_obstacle(ego_vehicle)
                draw_object(ego_vehicle, draw_params={'time_begin': time_begin,
                                                      'time_end': time_end,
                                                      'dynamic_obstacle': {'shape': {'opacity': 1,
                                                                                     'facecolor': '#0000ff',
                                                                                     'edgecolor': '#000000'},
                                                                           'occupancy': {'draw_occupancies': 1,
                                                                                         'shape': {'opacity': 0.2,
                                                                                                   'facecolor': '#0000ff',
                                                                                                   'edgecolor': '#0000ff'}},
                                                                           'trajectory': {'draw_trajectory': False}
                                                                           }})
            shadow_obstacles = scenario.obstacles_by_role_and_type(
                obstacle_type=ObstacleType.UNKNOWN)
            scenario.remove_obstacle(shadow_obstacles)
            draw_object(scenario, draw_params=scenario_draw_params)
            self.draw_shadows(shadow_obstacles, time_begin, 20)

            scenario.add_objects(shadow_obstacles)
            if ego_vehicle is not None:
                scenario.add_objects(ego_vehicle)
        if obstacles is not None:
            for obstacle in obstacles:
                draw_params = {
                    'shape': {'opacity': 0.2, 'facecolor': '#1d7eea'}}
                draw_object(ShapelyPolygon2Polygon(
                    obstacle), draw_params=draw_params)
        if lanes is not None:
            for lane in lanes:
                draw_params = {
                    'shape': {'opacity': 0.2, 'facecolor': '#1d7eea'}}
                draw_object(lane.convert_to_polygon(), draw_params=draw_params)
        if polygons is not None:
            for polygon in polygons:
                draw_params = {
                    'shape': {'opacity': 0.5, 'facecolor': '#ffff00'}}
                draw_object(polygon, draw_params=draw_params)
        if shapelyPolygons is not None:
            for shapelyPolygon in shapelyPolygons:
                draw_params = {
                    'shape': {'opacity': 0.5, 'facecolor': '#ffff00'}}
                draw_object(ShapelyPolygon2Polygon(
                    shapelyPolygon), draw_params=draw_params)
        if shadows is not None:
            for shadow in shadows:
                draw_params = {
                    'shape': {'opacity': 0.5, 'facecolor': '#ffff00'}}
                draw_object(ShapelyPolygon2Polygon(
                    shadow.polygon), draw_params=draw_params)
        if goal_region is not None:
            draw_params = {
                "goal_region": {
                    "draw_shape": False,
                    "shape": {
                        "circle": {
                            "opacity": 1.0,
                            "linewidth": 0.5,
                            "facecolor": "#00ff00",
                            "edgecolor": "#302404",
                            "zorder": 15
                            }
                        }
                    }
                }
            draw_object(goal_region, draw_params=draw_params)

    def plot_show(self,
                  scenario=None,
                  time_begin=0,
                  ego_vehicle=None,
                  obstacles=None,
                  lanes=None,
                  polygons=None,
                  shapelyPolygons=None,
                  shadows=None,
                  sensor_view=None,
                  goal_region=None):

        plt.figure(figsize=(10, 10))
        self.plot(scenario=scenario,
                  time_begin=time_begin,
                  ego_vehicle=ego_vehicle,
                  obstacles=obstacles,
                  lanes=lanes,
                  polygons=polygons,
                  shapelyPolygons=shapelyPolygons,
                  shadows=shadows,
                  sensor_view=sensor_view,
                  goal_region=goal_region)
        plt.xlim(0, 100)
        plt.ylim(-50, 50)
        plt.show()
