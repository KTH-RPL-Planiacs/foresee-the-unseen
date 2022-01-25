from typing import Dict, Union
from commonroad.scenario.obstacle import ObstacleType
from commonroad.common.util import Interval
from sumocr.sumo_config.default import DefaultConfig

class CRSumoConfig(DefaultConfig):

    def __init__(self, yaml_conf):
        self.scenario_name = yaml_conf.get('scenario_name')
        self.simulation_steps = yaml_conf.get('simulation_duration')
        self.ego_veh_width = yaml_conf.get('vehicle_width')
        self.ego_veh_length = yaml_conf.get('vehicle_length')
        self.presimulation_steps = 2  # number of time steps before simulation with ego vehicle starts

        # vehicle attributes
        self.veh_params: Dict[str, Dict[ObstacleType, Union[Interval, int, float]]] = {
            # maximum length
            'length': {
                ObstacleType.CAR: 5.0,
                ObstacleType.TRUCK: 7.5,
                ObstacleType.BUS: 12.4,
                ObstacleType.BICYCLE: 2.,
                ObstacleType.PEDESTRIAN: 0.415
            },
            # maximum width
            'width': {
                ObstacleType.CAR: 2.0,
                ObstacleType.TRUCK: 2.6,
                ObstacleType.BUS: 2.7,
                ObstacleType.BICYCLE: 0.68,
                ObstacleType.PEDESTRIAN: 0.678
            },
            'minGap': {
                ObstacleType.CAR: 2.5,
                ObstacleType.TRUCK: 2.5,
                ObstacleType.BUS: 2.5,
                # default 0.5
                ObstacleType.BICYCLE: 1.,
                ObstacleType.PEDESTRIAN: 0.25
            },
            # the following values cannot be set of pedestrians
            'accel': {
                # default 2.9 m/s²
                ObstacleType.CAR: Interval(2, 2.9),
                # default 1.3
                ObstacleType.TRUCK: Interval(1, 1.5),
                # default 1.2
                ObstacleType.BUS: Interval(1, 1.4),
                # default 1.2
                ObstacleType.BICYCLE: Interval(1, 1.4),
            },
            'decel': {
                # default 7.5 m/s²
                ObstacleType.CAR: Interval(4, 6.5),
                # default 4
                ObstacleType.TRUCK: Interval(3, 4.5),
                # default 4
                ObstacleType.BUS: Interval(3, 4.5),
                # default 3
                ObstacleType.BICYCLE: Interval(2.5, 3.5),
            },
            'maxSpeed': {
                # default 180/3.6 m/s
                ObstacleType.CAR: yaml_conf.get('max_velocity'),
                # default 130/3.6
                ObstacleType.TRUCK: yaml_conf.get('max_velocity'),
                # default 85/3.6
                ObstacleType.BUS: yaml_conf.get('max_velocity'),
                # default 85/3.6
                ObstacleType.BICYCLE: yaml_conf.get('max_velocity'),
            }
        }
        # vehicle behavior
        """
        'lcStrategic': eagerness for performing strategic lane changing. Higher values result in earlier lane-changing. sumo_default: 1.0
        'lcSpeedGain': eagerness for performing lane changing to gain speed. Higher values result in more lane-changing. sumo_default: 1.0
        'lcCooperative': willingness for performing cooperative lane changing. Lower values result in reduced cooperation. sumo_default: 1.0
        'sigma': [0-1] driver imperfection (0 denotes perfect driving. sumo_default: 0.5
        'speedDev': [0-1] deviation of the speedFactor. sumo_default 0.1
        'speedFactor': [0-1] The vehicles expected multiplicator for lane speed limits. sumo_default 1.0
        'lcImpatience': [-1-1] dynamic factor for modifying lcAssertive and lcPushy. sumo_default 0.0
        'impatience': [0-1] Willingness of drivers to impede vehicles with higher priority. sumo_default 0.0
        """
        # Default sumo values. These values does not seem to be used anyway?
        self.driving_params = {
            'lcStrategic': 1.0,
            'lcSpeedGain': 1.0,
            'lcCooperative': 1.0,
            'sigma': 0.5,
            'speedDev': 0.1,
            'speedFactor': 1.0,
            'lcImpatience': 0,
            'impatience': 0
        }