using Cloudy.BasisFunctions
using Cloudy.Collocation

# ATTEMPT 2: match a bunch of data at particular bin locations and times

# set up of the read-in data
bin_lims = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
bin_centers = (bin_lims[2:end]+bin_lims[1:end-1])/2

# set up of the collocation stuff
coll_mu = [1.5, 4.5, 9.5]

function loss_gaussbasis(Nb::FT, coll_mu::Array{FT}, coll_sigma::Array{FT}, bin_centers::Array{FT}, bin_data::Array{FT, 2}) where {FT <: Real}
    # set up the basis with the current guess at hyperparameters
    basis = Array{PrimitiveUnivariateBasisFunc}(undef, Nb)
    for i = 1:Nb
        basis[i] = GaussianBasisFunction(coll_mu[i], coll_sigma[i])
    end

    A = get_rbf_inner_products(basis)

    N_time = length(bin_data, 2)
    n_exact = bin_data[find_closest_bin_points(coll_mu, bin_centers)][:]

    L = 0
    
    for t=1:N_time
        c_t = A\n_exact[:][t]
        n_exact_full = bin_data[:][t]
        n_approx_full = evaluate_rbf(basis, c_t, bin_centers)

        L = L + (n_exact_full - n_approx_full).^2
    end
end

function find_closest_bin_points(coll_mu::Array{FT}, bin_centers::Array{FT}) where {FT <: Real}
    findnearest(A::AbstractArray,t) = findmin(abs.(A.-t))[2]

    N = length(coll_mu)
    bin_indices = Array{FT}(undef, N)

    for i=1:N
        bin_indices[i] = findnearest(bin_centers, coll_mu[i])
    end
end
