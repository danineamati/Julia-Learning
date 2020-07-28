# Constraint Manager holds a list of constaints and can run the necessary
# calculus

include("constraints.jl")

using SparseArrays

"""
    constraintManager

Contains a list of constraints and the dual variables (lambda) that correspond
with each constraint.
"""
struct constraintManager
    cList::Array{constraint, 1}
    lambdaList
end


#############################################################
###            Evaluate a List of Constraints             ###
#############################################################

function evalConstraints(yCurr, cM::constraintManager, al::augLag)

    total_eval = 0

    for (i, c) in enumerate(cM.cList)
        lambda = cM.lambdaList[i]
        cVal = getNormToProjVals(c, yCurr)
        total_eval += al.rho * (norm(cVal)^2) + al.lambda * cVal
    end

    return total_eval
end

#############################################################
###     Evaluate the Gradient of a List of Constraints    ###
#############################################################

function evalGradConstraints(yCurr, cM::constraintManager, al::augLag)

    total_grad = zeros(size(yCurr))

    for (i, c) in enumerate(cM.cList)
        lambda = cM.lambdaList[i]
        cVal = getNormToProjVals(c, yCurr)
        cJacob = getGradC(c, yCurr)
        total_grad += cJacob' * (al.rho * cVal + lambda)
    end

    return total_grad

end


#############################################################
###     Evaluate the Hessian of a List of Constraints     ###
#############################################################

function evalHessConstraints(yCurr, cM::constraintManager, al::augLag)

    total_hess = spzeros(size(yCurr, 1), size(yCurr, 1))

    for c in cM.cList
        total_hess += getHessC(c)
    end

    return total_grad

end
