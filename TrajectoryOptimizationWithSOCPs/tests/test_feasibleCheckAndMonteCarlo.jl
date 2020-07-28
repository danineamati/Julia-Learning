# Test Feasible Check Code and Monte Carlo Code (which are related)

include("..\\src\\other_utils\\feasibleCheck.jl")
include("..\\src\\other_utils\\monteCarlo.jl")

runTestsFeasible = true
runTestsMonteCarlo = true

if runTestsFeasible
    println("\n")
    print("Choosing a point in the feasible region (true): ")
    println(isFeasiblePolyHedron([4 5; -4 5; 0 -1], [20; 30; 1], [-1; 0.5]))

    print("Choosing a point outside the feasible region (false): ")
    println(isFeasiblePolyHedron([4 5; -4 5; 0 -1], [20; 30; 1], [-10; 0.5]))

    print("Choosing another point outside the feasible region (false): ")
    println(isFeasiblePolyHedron([4 5; -4 5; 0 -1], [20; 30; 1], [-1; 10]))

    print("Choosing another point outside the feasible region (false): ")
    println(isFeasiblePolyHedron([4 5; -4 5; 0 -1], [20; 30; 1], [-1; -10]))
end

if runTestsMonteCarlo
    println("\n")
    print("Generating a 3D point: ")
    pt1 = getPoint([3; 2; -5], [10; 100; 0.5])
    println(pt1)
    print("Checking that the point is in the bounds: ")
    print("$(10 > pt1[1] > 3) & $(100 > pt1[2] > 2)")
    println(" & $(0.5 > pt1[3] > -5)")

    println("Generating 3 Random 4D Points")
    display(montecarlo([3; 2; -5; -0.01], [10; 100; 0.5; 0.01], 3))

    println("Generating 2 Random Points in a smaller [0, 1] box")
    display(montecarlo([-20; -20], [20; 20], 2, inBox))
end
