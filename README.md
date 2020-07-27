# SOCP Trajectory Optimization in Julia 
(For now the repository is known as Julia-Learning. Subject to change.)

## Overview
This is a repository where I am storing my files as I learn Julia. Work here is in collaboration with the REx Lab at Stanford (June 2020 - August 2020). If you have questions, please contact me at [dneamati@caltech.edu](dneamati@caltech.edu). Also, find me at [https://sites.google.com/view/danielneamati/](https://sites.google.com/view/danielneamati/). Thanks!

## Funding
Funding for this research is graciously provided by the Caltech Summer Undergraduate Research Fellowship (SURF) program. Specifically, I am very thankful to be the 2020 __Homer J. Stewart SURF Fellow__.

## Repository Roadmap
The following is a rough roadmap of the repository:
1. __*TrajectoryOptimizationWithSOCPs*__: This is the most recent folder and is currently in development. This folder contains trajectory optimizers that can handle quadratic cost functions, quadratic constraints, and second-order cone constraints. This folder largely builds off of the *SOCP_and_QP_SolversWithProjections* folder.
2. __*SOCP_and_QP_SolversWithProjections*__: This is the second most recent folder. The SOCP solver in this folder contains documentation using Documenter.jl. The folder contains both SOCP and QP solvers that use projections in the constraints to more accurately calculate constraint violation.
3. __*QP_SolverComparison*__: This folder compares Primal and Primal-Dual solvers located in the *Primal-Dual_AL_Newton_QP* (augmented lagrangian solver) and *Primal-Dual_IP_Newton_QP* (interior point solver) folders.
4. __*Primal-Dual_AL_Newton_QP*__ and __*Primal-Dual_IP_Newton_QP*__: These folders contain augmented lagrangian and interior point solvers, respectively.
5. __*LearningOptimization*__: This folder contains code from when I first started learning optimization. This includes gradient descent methods and simple log-barrier methods.
6. __*Basic Julia*__: Folder from when I first started learning Julia.
