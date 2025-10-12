# Dynamical System Optimization 

This is the barebone implementation of dynamical system optimization (dsopt). For a brief introduction, one can refer to the README in this [repo](https://github.com/nbfigueroa/ds-opt). This package should be used as a dependency of the linear parameter varying dynamical system ([LPV-DS](https://github.com/sunan-sun/lpvds)) framework.

The code contains the fast/convex learning of the parametrized Quadratic Lyapunov Function [1] as well as the original method proposed in [2], which has been renamed as `_optimize_P_legacy`.


## References

> [1] Li, T., Sun, S., Aditya, S. S., & Figueroa, N. (2025). Elastic Motion Policy: An Adaptive Dynamical System for Robust and Efficient One-Shot Imitation Learning. arXiv preprint arXiv:2503.08029.

> [2] Khansari-Zadeh, S. M., & Billard, A. (2014). Learning control Lyapunov function to ensure stability of dynamical system-based robot reaching motions. Robotics and Autonomous Systems, 62(6), 752-765.


## Contact

Contact: [Sunan Sun](https://sunan-sun.github.io/) (sunan@seas.upenn.edu)
