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

- Carla is used as the simulation environment
- There are multiple cars driving and learning simultaneously
- An attached camera serves as input
- The agent can execute three actions (steer left/right, accelerate)
- It receives a positive reward for remaining speed and avoiding collisions

## Training

- Training was parallelized on multiple nodes
- Self-coded management software for synchronisation between them
- Central mainframe node distributes scripts and collects logs

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

*Every agent controls a Cybertruck*
![Cybertruck](./Screenshots/Cybertruck.jpg)

*View of the agent through the attached camera*
![Cam](./Screenshots/Cam.jpg)

*An interesting situation during training*
![Fail](./Screenshots/Fail.jpg)

*Small carla map (for testing only)*
![Map_1](./Screenshots/Map_2.jpg)

*Bigger carla map (used for training)*
![Map_2](./Screenshots/Map_1.jpg)

*Results of model 1*
![Model_1](./Screenshots/Model_1.png)

*Results of model 2*
![Model_2](./Screenshots/Model_2.png)

*Self-build computing cluster*
![Cluster](./Screenshots/Cluster.jpg)

## Dependencies

- [Carla](https://carla.org)
- [TensorFlow](https://www.tensorflow.org)
- cv2
- numpy

## Acknowledgements

This project was based on a tutorial
by [Sentdex](https://pythonprogramming.net/introduction-self-driving-autonomous-cars-carla-python/).
