# AI Cars

> Enable cars to drive autonomously in a simulation using Deep Reinforcement Learning.

## Table of Contents

* [Introduction](#introduction)
* [Training](#training)
* [Evaluation](#evaluation)
* [Screenshots](#screenshots)
* [Dependencies](#dependencies)
* [Acknowledgements](#acknowledgements)

## Introduction

- Carla offers the simulation environment with API
- Multiple agents (cars) are learning simultaneously
- An attached camera serves as input
- The agents can execute three actions (steer left/right, accelerate)
- They receive a positive reward for remaining speed and avoiding collisions

## Training

- Training was parallelized on multiple nodes
- Self-coded management software for synchronisation between nodes
- Central mainframe distributes scripts and collects results

## Evaluation

- Model 1 (Xception)
    - 30 million trainable parameters
    - way too big neural network
    - massive over-fitting
- Model 2 (Sequential with convolution & pooling layers)
    - 3 million trainable parameters
    - usable results
    - actually learning

## Screenshots

![](./screenshots/cybertruck.jpg)
<br/>
*Every agent controls a Cybertruck*

![](./screenshots/cam.jpg)
<br/>
*View of the agent through the attached camera*

![](./screenshots/fail.jpg)
<br/>
*An interesting situation during training*

![](./screenshots/map_2.jpg)
<br/>
*Small carla map (for testing only)*

![](./screenshots/map_1.jpg)
<br/>
*Bigger carla map (used for training)*

![](./screenshots/model_1.png)
<br/>
*Results of model 1*

![](./screenshots/model_2.png)
<br/>
*Results of model 2*

![](./screenshots/cluster.jpg)
<br/>
*Self-build computing cluster*

## Dependencies

- [Carla](https://carla.org)
- [TensorFlow](https://www.tensorflow.org)
- cv2
- numpy

## Acknowledgements

This project was based on a tutorial
by [Sentdex](https://pythonprogramming.net/introduction-self-driving-autonomous-cars-carla-python/).

*Original idea in September 2021*
