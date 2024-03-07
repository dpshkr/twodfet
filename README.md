# twodfet
Python Simulation of 2D material based Field Effect Transistors.

![Double gated two dimensional MOSFET](docs/figs/titlefig.png)

Device research community has proposed 2D material based transistors to be a potential replacement for Silicon, 
especially for sub 10 nm channel lengths. This is because, 2D materials can be scaled down to a monolayer limit 
without introducing surface states / dangling bonds.

This tool is a 2D simulator, which simulates a monolayer channel FET with a double-gate structure.
Salient features include -

* Proper 2D material based-physics, like 2D density of states
* Free and open-source. Only depends on NumPy, SciPy and Matplotlib.
* Multiple transport models - drift-diffusion and semiclassical ballistic transport.

SEM image from title figure courtsey of *O’Brien, Kevin P., et al. "Process integration and future outlook of 2D transistors." Nature Communications 14.1 (2023): 6400.*

