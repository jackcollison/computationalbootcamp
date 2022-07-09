# Load required packages
using CSV, DataFrames, GLM, Distributions, Plots, Optim, NLSolversBase, Random, LinearAlgebra, Distributed

# Problem 1
# Load data and rename
data = CSV.read("/Users/jackcollison/Desktop/Wisconsin/Computational Bootcamp/computationalbootcamp/lwage.csv", DataFrame, header=false)
rename!(data, ["Wage", "College", "Experience"])

# Generate new variables
data.Experience2 = data.Experience.^2

# Fit least squares
ols = lm(@formula(Wage ~ College + Experience + Experience2), data)
println("OLS yields: \n $ols")

# Functionality for log-likelihood minimization
function ll_min(seed::Int64, ssize::Int64, data::DataFrame)
    # Set up environment
    Random.seed!(seed)
    nvar = 4
    Y = data.Wage
    n = length(Y)
    X = hcat(ones(n), data.College, data.Experience, data.Experience2)

    # Random sample without replacement
    indices = shuffle(1:n)[1:ssize]
    X = X[indices, :]
    Y = Y[indices, :]
    n = length(Y)

    # Log-likelihood functionality
    function Log_Likelihood(X, Y, β, log_σ)
        σ = exp(log_σ)
        llike = -n / 2 * log(2π) - n / 2 * log(σ^2) - (sum((Y - X * β).^2) / (2σ^2))
        llike = -llike
    end

    # Initial guess
    x₀ = [2.175, 0.515, 0.040, 0.000, 0.585]
    func = TwiceDifferentiable(vars -> Log_Likelihood(X, Y, vars[1:nvar], vars[nvar + 1]), x₀; autodiff=:forward);
    opt = optimize(func, x₀; g_tol = 1e-5)
    parameters = Optim.minimizer(opt)
    parameters[nvar+1] = exp(parameters[nvar+1])

    # Return
    parameters
end

# Fit on whole sample
ll_min(0, length(data.Wage), data)

ll_min(0, Int(floor(length(data.Wage) / 2)), data)

# Package for presentation

## TODO: Fix this up and add bootstrapping for standard errors

# # Problem 2
# # Monte-Carlo simulations
# function matching(n::Int64, iterations::Int64)
#     # Function to shuffle and find matches
#     function find_matches(n::Int64)
#         x, y = collect(1:1:n), shuffle!(collect(1:1:n))
#         sum(x[i] == y[i] for i in 1:n)
#     end

#     # Get matches
#     matches = [find_matches(n) for i in 1:iterations]
#     histogram(matches, bins=10, title="Matches for n = $n")
# end

# # Simulate matches
# p1 = matching(10, 10000)
# p2 = matching(20, 10000)
# Plots.plot(p1, p2, layout = (2,1), legend=:none)

# # Problem 3
# # Simulate savings
# function sim_savings(E::Float64, S::Float64, P::Float64, years::Int64, uncertain::Bool, iterations::Int64)
#     # Simulate savings and earnings
#     p = zeros(iterations)
#     for i in 1:iterations
#         # Instantiate
#         Sᵢ, Eᵢ = S, E

#         # Loop over years
#         for t in 1:years
#             # Calculate savings and earnings
#             Sᵢ *= (1 + ifelse(uncertain, rand(Normal(0.06, 0.06)), 0.06))
#             Eᵢ *= (1 + ifelse(uncertain, rand(Uniform(0, 0.06)), 0.03))
#             Sᵢ += P * Eᵢ
#         end

#         # Get proportion of earnings
#         p[i] = Sᵢ / Eᵢ
#     end

#     # Return
#     p
# end

# # Run simulations for certain and uncertain outcomes
# p = sim_savings(100.0, 100.0, 0.1125, 37, false, 1)
# println("The returns are p = ", p[1])
# p = sim_savings(100.0, 100.0, 0.1125, 37, true, 100000)
# println("The proportion of returns at least 10x earnings is estimated as p = ", sum(p .< 10) / length(p))
# p = sim_savings(100.0, 100.0, 0.15, 37, true, 100000)
# println("The proportion of returns at least 10x earnings is estimated as p = ", sum(p .< 10) / length(p))