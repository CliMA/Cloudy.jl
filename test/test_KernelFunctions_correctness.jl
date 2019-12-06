"Testing correctness of KernelFunctions module."

using Cloudy.KernelFunctions

import Cloudy.KernelTensors: check_symmetry

# testing attributes of kernel functions
# symmetry
@test check_symmetry(linear) == nothing
