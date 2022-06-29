# Load libraries
using Plots

# Question 1
function factorial2(n::Int64)
    fact = 1
    for i in 1:n
        fact *= i
    end
    return fact
end

# Question 2
function p(x::Float64, coeff::Vector{Float64})
    val = 0
    for (i, a) in enumerate(coeff)
        val += a * x^(i-1)
    end
    return val
end

# Question 3
function sim_pi(n::Int64)
    B = 0
    for i in 1:n
        a, b = rand(2)
        d = sqrt((0.5 - a)^2 + (0.5 - b)^2)
        B += (d <= 0.5)
    end
    return 4 * B / n
end

# Question 4
function sim_data(a::Float64, b::Float64, c::Float64, d::Float64, σ::Float64, n::Int64)
    # Initialize
    x₁, x₂ = randn(50), randn(50)
    coefs = []

    # Simulations
    for i in 1:n
        # Generate w, y
        w = randn(50)
        y = a .* x₁ .+ b .* x₁.^2 .+ c .* x₂ .+ d .+ σ .* w

        # Regression
        X = hcat(x₁, x₁.^2, x₂, ones(50))
        coef = inv(X' * X) * X' * y
        push!(coefs, coef)
    end
    p1 = histogram(coefs[1,:], bins=10, title="a")
    p2 = histogram(coefs[2,:], bins=10, title="b")
    p3 = histogram(coefs[3,:], bins=10, title="c")
    p4 = histogram(coefs[4,:], bins=10, title="d")
    Plots.plot(p1, p2, p3, p4, layout = (2,2), legend=:none)
end
sim_data(0.1, 0.2, 0.5, 1.0, 0.1, 200)

# Question 5
function random_walk(n::Int64, tmax::Int64, α::Float64, σ::Float64, a::Float64)
    # Initializations
    x₀ = 1
    T = []

    # Simulations
    for i in 1:n
        # Generate data
        xₚ = x₀
        for i in 1:tmax
            # Next data point and check
            xₜ = α * xₚ + σ * randn()
            if xₜ <= a || i == tmax
                push!(T, i)
                break
            end
            xₚ = xₜ
        end
    end
    histogram(T, bins=10, title="Stopping Times", legend=:none)
end
random_walk(100, 200, 0.8, 0.2, 0.0)
random_walk(100, 200, 1.0, 0.2, 0.0)
random_walk(100, 200, 1.2, 0.2, 0.0)

# Question 6
function newton(f, fₚ, x₀::Float64, tol::Float64, maxiter::Int64)
    # Initial guess
    xₚ = x₀
    xₜ = xₚ - f(xₚ) / fₚ(xₚ)

    # Loop until tolerance or maximum iterations
    for i in 1:maxiter
        # Update guess
        xₚ = xₜ
        xₜ = xₚ - f(xₚ) / fₚ(xₚ)

        # Check stopping condition
        if abs(xₜ - xₚ) <= tol
            println("Found root at $xₜ in $i iterations.")
            return xₜ
        end
    end

    # Return at maximum iterations
    println("Found root at $xₜ in $maxiter iterations.")
    return xₜ
end
f(x) = (x - 1)^3
fₚ(x) = 3 * (x - 1)^2
newton(f, fₚ, 0.0, 1e-3, 100)

# Question 7
# Import packages
using Parameters, Plots, LinearAlgebra

# Create primatives
@with_kw struct Primitives
    # Define global constants
    β::Float64 = 0.99 
    θ::Float64 = 0.36
    δ::Float64 = 0.025
    M::Array{Float64, 2} = [0.977 0.023; 0.074 0.926]
    k_grid::Array{Float64,1} = collect(range(1.0, length = 50, stop = 45.0))
    z_grid::Array{Float64,1} = [1.25, 0.2]
    nk::Int64 = length(k_grid)
    nz::Int64 = length(z_grid)
end

# Structure for results
mutable struct Results
    # Value and policy functions
    value_func::Array{Float64, 2}
    policy_func::Array{Float64, 2}
end

# Functionality to solve model
function Solve_model()
    # Initialize primitives
    prim = Primitives()
    value_func, policy_func = zeros(prim.nk, prim.nz), zeros(prim.nk, prim.nz)
    res = Results(value_func, policy_func)

    # Instantiate error and iteration; loop until convergence
    error, n = 100, 0
    while error > eps()
        # Increment counter
        n += 1

        # Call Bellman operator and update
        v_next = Bellman(prim, res)
        error = maximum(abs.(v_next .- res.value_func))
        res.value_func = v_next

        # Print error every so often
        if mod(n, 5000) == 0 || error < eps()
            println(" ")
            println("*************************************************")
            println("AT ITERATION = ", n)
            println("MAX DIFFERENCE = ", error)
            println("*************************************************")
        end
    end

    # Return values
    prim, res
end

# Bellman operator
function Bellman(prim::Primitives, res::Results)
    # Unpack primitive structure, instantiate value function
    @unpack β, θ, δ, M, k_grid, z_grid, nk, nz = prim
    v_next = zeros(nk, nz)

    # Iterate over state space and productivity
    for (i_k, k) in enumerate(k_grid)
        for (i_z, z) in enumerate(z_grid)
            # Candidate maximum value, budget constraint
            max_util = -1e10
            budget = k^θ + (1 - δ) * k

            # Iterate over next period capital choice
            for (i_kp, kp) in enumerate(k_grid)
                # Find consumption
                c = budget - kp

                # Check positivity
                if c > 0
                    # Compute value
                    val = log(c) + β * (res.value_func[i_kp, :] ⋅ M[i_z, :])

                    # Check maximum
                    if val > max_util
                        # Update values
                        max_util = val
                        res.policy_func[i_kp, i_z] = kp
                    end
                end
            end

            # Update next iteration
            v_next[i_k, i_z] = max_util
        end
    end

    # Return values
    v_next
end

# Check Functionality
@elapsed prim, res = Solve_model()
p1 = plot(prim.k_grid, res.value_func)
p2 = plot(prim.k_grid, res.policy_func)

# Plots
Plots.plot(p1, p2, layout = (2,1), legend=:none)