# Test the Constraint Manager

include("..\\src\\constraints\\constraintManager.jl")


# Start Simple
println("########################")
println("#   Simple Test Case   #")
println("########################")

c1 = AL_AffineEquality([5], [10])
c2 = AL_AffineEquality([5], [10])
c3 = AL_AffineEquality([5], [10])
lambdaVec = [[2], [3], [4]]

cM1 = constraintManager([c1, c2, c3], lambdaVec)
x0Test = [5]
penaltyTest = 10.0
println("Evaluating the Constraints: ")
println(evalConstraints(x0Test, cM1, penaltyTest))

println("Evaluate Gradient of the Constraint Terms: ")
println(evalGradConstraints(x0Test, cM1, penaltyTest))

println("Evaluate Hessian of the Constraint Terms: ")
println(evalHessConstraints(x0Test, cM1, penaltyTest))

println("########################")
println("#    2D Test Case 1    #")
println("########################")
c1 = AL_AffineEquality([5 0; 0 6], [10; 12])
c2 = AL_AffineEquality([5 0; 0 6], [10; 12])
c3 = AL_AffineEquality([5 0; 0 6], [10; 12])
lambdaVec = [[2; 2], [3; 4], [4; 2]]

cM1 = constraintManager([c1, c2, c3], lambdaVec)
x0Test = [5; -3]
penaltyTest = 10.0
println("Evaluating the Constraints: ")
println(evalConstraints(x0Test, cM1, penaltyTest))

println("Evaluate Gradient of the Constraint Terms: ")
println(evalGradConstraints(x0Test, cM1, penaltyTest))

println("Evaluate Hessian of the Constraint Terms: ")
println(evalHessConstraints(x0Test, cM1, penaltyTest))


println("########################")
println("#  2D Ineq. Test Case  #")
println("########################")
c1 = AL_AffineInequality([5 0; 0 6], [10; 12])
c2 = AL_AffineInequality([5 0; 0 6], [10; 12])
c3 = AL_AffineInequality([5 0; 0 6], [10; 12])
lambdaVec = [[2; 2], [3; 4], [4; 2]]

cM1 = constraintManager([c1, c2, c3], lambdaVec)
x0Test = [5; -3]
penaltyTest = 10.0
println("Evaluating the Constraints: ")
println(evalConstraints(x0Test, cM1, penaltyTest))

println("Evaluate Gradient of the Constraint Terms: ")
println(evalGradConstraints(x0Test, cM1, penaltyTest))

println("Evaluate Hessian of the Constraint Terms: ")
println(evalHessConstraints(x0Test, cM1, penaltyTest))

println("###############################")
println("#  2D Affine Mixed Test Case  #")
println("###############################")
c1 = AL_AffineInequality([5 0; 0 6], [10; 12])
c2 = AL_AffineEquality([5 0; 0 6], [10; 12])
c3 = AL_AffineInequality([5 0; 0 6], [10; 12])
lambdaVec = [[2; 2], [3; 4], [4; 2]]

cM1 = constraintManager([c1, c2, c3], lambdaVec)
x0Test = [5; -3]
penaltyTest = 10.0
println("Evaluating the Constraints: ")
println(evalConstraints(x0Test, cM1, penaltyTest))

println("Evaluate Gradient of the Constraint Terms: ")
println(evalGradConstraints(x0Test, cM1, penaltyTest))

println("Evaluate Hessian of the Constraint Terms: ")
println(evalHessConstraints(x0Test, cM1, penaltyTest))
