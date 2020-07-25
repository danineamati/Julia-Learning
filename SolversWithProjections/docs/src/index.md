# SOCP Solver Documentation

The key functionality is the primal SOCP solver.

```@meta
CurrentModule = SolversWithProjections
```

The SOCP problem is formulated as follows.

```math
\begin{aligned}
\underset{x, s, t}{\text{minimize}} \quad& \frac{1}{2} x^\top Q x + p^\top x \\
\text{subject to} \quad& ||s||_2 ≤ t \\
& Ax - b = s \\
& c^\top x - d = t
\end{aligned}
```

Below, the documentation is presented in the following order:
1. Setup
2. Solver

*Section 1: Setup* covers
1. SOCP Primal Variable Struct and Methods
2. Quadratic Objective Struct and Methods
3. Lagrangian Struct and Methods
4. Solver Parameters

*Section 2: Solver* covers
1. The high level solver functionality
2. The trust region on the interior

# Section 1: Setup

## SOCP Primal Variable Struct and Methods
```@docs
SOCP_primals
primalVec
primalStruct
getXVals
```

## Quadratic Objective Struct and Methods
```@docs
objectiveQP
fObjQP
dfdxQP
```

## Lagrangian Struct and Methods
```@docs
augLagQP_2Cone
evalAL
evalGradAL
evalHessAl
calcNormGradResiduals
getNormRes
calcALArr
getViolation
```

## Solver Parameters
```@docs
solverParams
solParamPrint
```

# Section 2: Solver

## High Level Solver Functionality
```@docs
ALPrimalNewtonSOCPmain
newtonTRLS_ALPSOCP
newtonStepALPSOCP
```

## Trust Region
```@docs
findDamping
dampingInitialization
```
