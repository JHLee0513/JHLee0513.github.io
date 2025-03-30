# Driving simluations for Autonomous Driving

## CARLA

[[Github]](https://github.com/carla-simulator/carla) [[Relevant presentation of CARLA by Vladlen Koltun]](https://www.youtube.com/watch?v=XmtTjqimW3g&t=2367s)

CARLA has been the go to simulation benchmark for a while, and the vast majority of works in autonomous driving has conducted experiments using the platform.

Advatanges:
* Allows for parallel computing for efficient training & testing. Simulation also may runder faster than 1x for faster data collection & testing.
* Many prior work have benchmark results, allowing for comparison against prior work as checking progress

Disadvantages:
* Sim2real domain gap exists both visually & behaviorally
* With version changes, benchmark results from prior work cannot be compared with test results on the newer version

## NuPlan
[[Website]](https://www.nuscenes.org/nuplan) [[Github]](https://github.com/motional/nuplan-devkit)

NuPlan takes a more data-driven approach, allowing experiments to be conducted by playing back recorded logs of the data from real vehicles.

Advantages:
* Allows for parallel computing for efficient training & testing. Simulation also may runder faster than 1x for faster testing.
* Real datasets are used


Disadvantages:
* Does not provide sensor data, hence perception is as given (may not realistically simulate deployment of perception systems on real scenarios)
* Sim2real domain gap (perfect vision, behavioral domain gap)
* Evaluation with traffic agents are somehwat elementary, the simulator only supports log playback or basic reactive agents.

### Running nuPlan

## Waymax
[[Paper]](https://arxiv.org/pdf/2310.08710.pdf) [[Github]](https://github.com/waymo-research/waymax)


* Data-driven simulator that uses the Waymo Open Motion Dataset (Dataset is quite large, probably best to use with cloud?)
* Simulator entirely written with JAX allowing  hardware acceleration
* Uses pre-processed bounding boxes for traffic agents and road graph for the environment (does not include sensor data).
* Driving behavior evaluation includes:
```
1. Log Divergence: L2 distance between simulated and logged poses.
2. Collision: An indicator metric determining whether the vehicle collides with another object.
3. Offroad: An indicator metric on whether the vehicle drives off the road.
4. Wrong-Way: An indicator metric on whether the vehicle is driving on the wrong side of the road.
5. Kinematics Infeasibility: An indicator metric on whether the vehicle’s action results in a kinematically infeasible transition.
```

* Apparently user does not have to locally store the dataset to run the simulator, I'm going to check myself in the future how well cloud-based simulation holds up (assuming it's either free cloud access or is affordable enough..)

Advantages:
* Traffic agents can be simulate by log playback or with an IDM-based route-following model. The latter will react based on other vehicles to some extent.
* Simulation supports commonly used interfaces for RL
* Written all in JAX, probably also natively supports pytorch

Disadvantages:
* Framework & data are both relatively new, and thus not many prior work on this exact benchmark testing exists

### Running Waymax
I followed instructions from the Waymax Github repository: <br>
https://github.com/waymo-research/waymax



<!-- ## Other data-driven methods

### VISTA (2.0)
* Builds data-driven playback to train end-to-end driving RL agent  -->