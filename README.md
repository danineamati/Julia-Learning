# Julia-Learning

This is a repository where I am storing my files as I learn Julia. Work here is in collaboration with the REx Lab at Stanford (June 2020 - August 2020). Funding for this research is graciously provided by the Caltech Summer Undergraduate Research Fellowship (SURF) program. Specifically, I am very thankful to be the 2020 Homer J. Stewart SURF Fellow.

The following is a rough roadmap of the repository:
1. *TrajectoryOptimizationWithSOCPs*: This is the most recent folder and is currently in development. This folder contains trajectory optimizers that can handle quadratic cost functions, quadratic constraints, and second-order cone constraints. This folder largely builds off of the *SOCP_and_QP_SolversWithProjections* folder.
2. *SOCP_and_QP_SolversWithProjections*: This is the second most recent folder. The SOCP solver in this folder contains documentation using Documenter.jl. The folder contains both SOCP and QP solvers that use projections in the constraints to more accurately calculate constraint violation.
3. *QP_SolverComparison*: This folder compares Primal and Primal-Dual solvers located in the *Primal-Dual_AL_Newton_QP* (augmented lagrangian solver) and *Primal-Dual_IP_Newton_QP* (interior point solver) folders.
4. *Primal-Dual_AL_Newton_QP* and *Primal-Dual_IP_Newton_QP*: These folders contain augmented lagrangian and interior point solvers, respectively.
5. *LearningOptimization*: This folder contains code from when I first started learning optimization. This includes gradient descent methods and simple log-barrier methods.
6. *Basic Julia*: Folder from when I first started learning Julia.
