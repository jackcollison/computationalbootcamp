# Load required packages
using Distributed

# Add cores
addprocs(3)

# Include packages everywhere
@everywhere using Optim, NLSolversBase, Random, SharedArrays, DataFrames, CSV

# Functionality for log-likelihood minimization
@everywhere function ll_min(seed::Int64)
    # Load data and rename
    println("Reading data...")
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
    println("Shuffling...")
    indices = shuffle(1:n)[1:Int(floor(n / 2))]
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
    println("Fitting data...")
    x₀ = [2.175, 0.515, 0.040, 0.000, 0.585]
    func = TwiceDifferentiable(vars -> Log_Likelihood(X, Y, vars[1:nvar], vars[nvar + 1]), x₀; autodiff=:forward);
    opt = optimize(func, x₀; g_tol = 1e-5)
    parameters = Optim.minimizer(opt)
    parameters[nvar+1] = exp(parameters[nvar+1])

    # Return
    println("Returning parameters...")
    parameters
end

# Shared array
iterations = 100
parameters = SharedArray{Float64}(iterations, 5)

# Distributed computing
@elapsed @sync @distributed for i = 1:iterations
    parameters[i,:] = ll_min(i)
end

# Bootstrapped standard errors
println("Standard errors are given by: \n", [std(parameters[:, i]) / sqrt(iterations) for i in 1:5])