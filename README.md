Wave Simulation using Taichi

This project is a simulation of waves using Taichi, a high-performance programming language that enables easy parallelization and optimization of code.
Wave Equation

The wave equation is a second-order partial differential equation that describes the behavior of waves in a physical system. It is expressed in terms of the displacement function u(x,t) which represents the wave amplitude at position x and time t. The wave equation in one dimension is given by:
```
d^2u/dt^2 = c^2 d^2u/dx^2
```
where c is the wave speed. This equation describes how the wave amplitude changes over time and space. It can be used to model various types of waves, such as sound waves, electromagnetic waves, and water waves.

To numerically solve the wave equation, we discretize the time and space domains. We divide the space into a grid of points and the time into discrete time steps. The wave equation can then be approximated using finite difference methods, such as the central difference method or the upwind scheme.

For example, the central difference method approximates the second derivative with the following formula:
```
d^2u/dx^2 = (u(x+dx) - 2u(x) + u(x-dx)) / dx^2
```
Substituting this approximation into the wave equation and solving for u(x,t+dt), we get:
```
u(x,t+dt) = 2u(x,t) - u(x,t-dt) + c^2(dt/dx^2)(u(x+dx,t) + u(x-dx,t) - 2u(x,t))
```
This equation can be used to update the wave amplitude at each point on the grid, given the values of the amplitude at the previous time step. The boundary conditions of the wave equation also need to be considered to ensure accurate simulation results.
#Dependencies

Taichi: Make sure Taichi is installed. You can install it via pip install taichi.

Running the simulation:

  To run the simulation, simply run the script.

Understanding the code:

  The code begins by importing the required libraries and initializing the Taichi environment.
  The parameters required for the simulation are defined, such as the size of the grid, the time step, and the velocity field.
  The init() function initializes the surface and velocity fields for the simulation.
  The cast2sphere() function casts the simulation onto a sphere to create a 3D visualization.
  The update() function updates the simulation at each time step, based on the wave equation and the boundary conditions.
  Finally, the code sets up a Taichi window to visualize the simulation, and allows for user interaction to change parameters such as frequency, amplitude, and damping.

Modifying the simulation

  The simulation can be modified by changing the parameters defined at the beginning of the code, such as the grid size and the simulation time step.
  The visualization can be modified by changing the camera position and orientation, and by adjusting the particle size and color.
