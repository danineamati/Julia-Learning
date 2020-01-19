include("apm.jl")

sol = apm_solve("2nd_order",4)

print(" Time:\n")
print(sol["time"])
print("\n y:\n")
print(sol["y"])
