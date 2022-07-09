# Load required packages
using Distributed
@everywhere using Optim, NLSolversBase, Random, SharedArrays, DataFrames, CSV

# Add cores
addprocs(3)

# Functionality for log-likelihood minimization
@everywhere function ll_min(seed::Int64, ssize::Int64)
    # Load data and rename
    data = CSV.read("/Users/jackcollison/Desktop/Wisconsin/Computational Bootcamp/computationalbootcamp/lwage.csv", DataFrame, header=false)
    rename!(data, ["Wage", "College", "Experience"])

    # Generate new variables
    data.Experience2 = data.Experience.^2

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

# Shared array for parallel processing
boot = SharedArray{Float64}(100, 5)

# Parallelized loop
@sync @distributed for seed = 1:100
    boot[seed,:] = ll_min(seed, Int(floor(length(data.Wage) / 2)))
end

# Bootstrap standard errors
se = zeros(5)
for i = 1:5
    se[i] = std(boot[:,i])/10
end

# Print standard errors
se