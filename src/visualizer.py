import matplotlib.pyplot as plt
from matplotlib import animation
import yaml
from commonroad.scenario.obstacle import ObstacleType
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams, DynamicObstacleParams, ShapeParams
from commonroad.geometry.shape import Polygon

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


    def draw_shadows(self, rnd, shadows, time_begin, time_horizon):
        # We use red
        R = int(255)
        G = int(0)
        B = int(0)

        # Draw the first location of the shadows
        draw_params = DynamicObstacleParams.load(file_path="src/draw_params/shadow.yaml", validate_types=False)
        draw_params.time_begin = time_begin
        draw_params.time_end = time_begin + 1
        for shadow in shadows:
            shadow.draw(rnd, draw_params=draw_params)

        # Draw the shadow predictions
        draw_params = DynamicObstacleParams.load(file_path="src/draw_params/shadow_prediction.yaml", validate_types=False)
        for i in reversed(range(time_horizon)):
            tint_factor = 0.005**(1/(i+1))
            Ri = int(R + (255 - R) * tint_factor)
            Gi = int(G + (255 - G) * tint_factor)
            Bi = int(B + (255 - G) * tint_factor)
            color = rgb2hex(Ri,Gi,Bi)

            draw_params.time_begin = time_begin+i
            draw_params.time_end = time_begin+i+1
            draw_params.occupancy.shape.facecolor = color
            draw_params.occupancy.shape.edgecolor = color
            for shadow in shadows:
                shadow.draw(rnd, draw_params=draw_params)

    def plot(self,
             scenario=None,
             time_begin=0,
             time_end=500,
             ego_vehicle=None,
             sensor_view=None):
        
        draw_params = MPDrawParams().load(file_path="src/draw_params/scenario.yaml")
        draw_params.time_begin = time_begin
        draw_params.time_end = time_end

        # Set global draw params for drawing
        rnd = MPRenderer(figsize=(8,8))
        rnd.draw_params = draw_params

        if sensor_view is not None:
            # Draw params can be overwritten when rendering specific objects
            draw_params = ShapeParams.load(file_path="src/draw_params/sensor_view.yaml", validate_types=False)
            ShapelyPolygon2Polygon(sensor_view).draw(rnd, draw_params=draw_params)
            
        if scenario is not None:
            if ego_vehicle is not None:
                scenario.remove_obstacle(ego_vehicle)

                draw_params = DynamicObstacleParams.load(file_path="src/draw_params/ego_vehicle.yaml", validate_types=False)
                draw_params.time_begin = time_begin
                draw_params.time_end = time_end
                ego_vehicle.draw(rnd, draw_params=draw_params)
                
            shadow_obstacles = scenario.obstacles_by_role_and_type(
                obstacle_type=ObstacleType.UNKNOWN)
            scenario.remove_obstacle(shadow_obstacles)

            self.draw_shadows(rnd, shadow_obstacles, time_begin, 20)
            scenario.draw(rnd)

            scenario.add_objects(shadow_obstacles)
            if ego_vehicle is not None:
                scenario.add_objects(ego_vehicle)

        rnd.render()

    def plot_show(self,
                  scenario=None,
                  time_begin=0,
                  ego_vehicle=None,
                  sensor_view=None):

        plt.figure(figsize=(10, 10))
        self.plot(scenario=scenario,
                  time_begin=time_begin,
                  ego_vehicle=ego_vehicle,
                  sensor_view=sensor_view)
        plt.xlim(0, 100)
        plt.ylim(-50, 50)
        plt.show()
