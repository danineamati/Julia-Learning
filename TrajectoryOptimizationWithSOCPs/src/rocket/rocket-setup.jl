# Contains the rocket set up and structs

"""
    rocket_simple

Barebones "spherical" rocket. This is to say, this struct only holds the mass
and the isp (specific impulse) of the rocket.
"""
struct rocket_simple
    mass
    isp
end
