using NLPModels
using NLPModelsIpopt
using BlackBoxOptim
using Cloudy.BasisFunctions

# SETUP: number of basis functions, locations, initial basis functions, initial guess
Nb = 4
rbf_loc = exp.(range(log(1),stop=log(100), length=Nb))
rbf_sigma1 = 1.0

# optimization function
function quadratic_moment_jumps(rbf_loc::Array{FT, 1}, rbf_sigma1::FT, sigma::Array{FT,1}) where {FT<:Real}
    rbf_sigma = [rbf_sigma1]
    append!(rbf_sigma, sigma)
    basis = Array{PrimitiveUnivariateBasisFunc}(undef, Nb)
    for i = 1:Nb
        basis[i] = GaussianBasisFunction(rbf_loc[i], rbf_sigma[i])
    end
    M1_i = get_moment(basis, 1.0)
    M2_i = get_moment(basis, 2.0)

    mom_jumps = M2_i[1:end-1] - M1_i[1:end-1]./M1_i[2:end].*M2_i[2:end]

    obj = sum(mom_jumps.^2)
    return obj
end

# let's go:
f(x) = quadratic_moment_jumps(rbf_loc, rbf_sigma1, x)
SR = Array{Tuple{Float64, Float64},1}(undef, Nb-1)
for i=2:Nb
   SR[i-1] = (rbf_loc[i], 1000.0)
end
res = bboptimize(f; SearchRange = SR)
best_candidate(res)