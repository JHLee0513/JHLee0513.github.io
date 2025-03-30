# Control Theory

## PID - Proportional, Integral, Derivate Control

The simplest and the most straightforward control algorithm. The controller is *feedback based i.e. closed-loop control*, where the error term calculating the offset desired state and current state is fed as input to the controller.

As the named suggets, PID has three modules:
1. Proportional: output control w.r.t current error term
2. Integral: output control w.r.t past error history
3. Derivative: output control w.r.t estimated future error term

The three modules combined (with weights) produce the final output control.


Another alternaitve is to use a known model of the system. One way to find such a model is *system identification*. For instance, this involves. This is how we calibrated the actuator control targets for our Clearpath Warthog, for instance, and this is relatively common to do so for various robotic platforms.


## MPC - Model Predictive Control

The theory of model predictive control is relatively well understood, mostly for linear systems than nonlinear systems. 



## MPPI

## LQR, iLQR