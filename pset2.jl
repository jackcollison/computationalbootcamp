# Load required libraries
using Optim, Interpolations, Plots

# Problem 1
# Himmelblau functionality
function Himmelblau(p::Vector{Float64})
    (p[1]^2 + p[2] - 11)^2 + (p[1] + p[2]^2 - 7)^2
end

# Instantiation of grids
x, y = collect(-4:0.01:4), collect(-4:0.01:4)
nx, ny = length(x), length(y)
z = zeros(nx, ny)

# Fill in function values
for i = 1:nx, j = 1:ny
    z[i,j] = Himmelblau([x[i], y[j]])
end

# Surface plot
Plots.surface(x, y, z, seriescolor=:viridis, camera = (50,50))

# Gradient function
function g!(G, p::Vector{Float64})
    G[1] = 4 * p[1] * (p[1]^2 + p[2] - 11) + 2 * (p[1] + p[2]^2 - 7)
    G[2] = 2 * (p[1]^2 + p[2] - 11) + 4 * p[2] * (p[1] + p[2]^2 - 7)
end

# Hessian function
function h!(H, p::Vector{Float64})
    H[1] = 4 * (p[1]^2 + p[2] - 11) + 8 * p[1]^2 + 2
    H[2] = 4 * p[1] + 4 * p[2]
    H[3] = 4 * p[1] + 4 * p[2]
    H[4] = 2 + 4 * (p[1] + p[2]^2 - 7) + 8 * p[2]^2
end

# Initial guess
x₀ = [0.0, 0.0]

# Newton's method optimization
@elapsed opt = optimize(Himmelblau, g!, h!, x₀)
println("Newton's method on the Himmelblau function yields: \n", opt)

## TODO: Use different initial guesses

# Nelder-Mead optimization
@elapsed opt = optimize(Himmelblau, x₀)
println("Nelder-Mead on the Himmelblau function yields: \n", opt)

## TODO: Use different initial guesses, compare iterations to Newton's method

# Problem 2
# Ackley functionality
function Ackley(p::Vector{Float64})
    -20 * exp(-0.2 * sqrt(0.5 * (p[1]^2 + p[2]^2))) - exp(0.5 * (cos(2 * π * p[1]) + cos(2 * π * p[2]))) + ℯ + 20
end

# Instantiation of grids
x, y = collect(-4:0.01:4), collect(-4:0.01:4)
nx, ny = length(x), length(y)
z = zeros(nx, ny)

# Fill in function values
for i = 1:nx, j = 1:ny
    z[i,j] = Ackley([x[i], y[j]])
end

# Surface and contour plots
p1 = Plots.surface(x, y, z, seriescolor=:viridis, camera = (50,50))
p2 = Plots.contourf(x, y, z, seriescolor=:inferno)
Plots.plot(p1, p2, layout = (2,1), legend=:none)

# Initial guess
x₀ = [1.0, 1.0]

# LBFGS optimization
opt = optimize(Ackley, x₀, LBFGS())
println("LBFGS on the Ackley function yields: \n", opt)

## TODO: Use different initial guesses

# Nelder-Mead optimization
opt = optimize(Ackley, x₀)
println("Nelder-Mead on the Ackley function yields: \n", opt)

## TODO: Use different initial guesses

# Problem 3
# Rastrigin functionality
function Rastrigin(p::Vector{Float64})
    10 * length(p) + sum([pᵢ^2 - 10 * cos(pᵢ) for pᵢ in p])
end

# Instantiation of grids
x = collect(-5.12:0.01:5.12)
nx = length(x)
y = zeros(nx)

# Fill in function values
for i = 1:nx
    y[i] = Rastrigin([x[i]])
end

# Plot function
Plots.plot(x, y, legend=:none)

# Instantiation of grids
x, y = collect(-5.12:0.01:5.12), collect(-5.12:0.01:5.12)
nx, ny = length(x), length(y)
z = zeros(nx, ny)

# Fill in function values
for i = 1:nx, j = 1:ny
    z[i,j] = Rastrigin([x[i], y[j]])
end

# Surface and contour plots
p1 = Plots.surface(x, y, z, seriescolor=:viridis, camera = (50,50))
p2 = Plots.contourf(x, y, z, seriescolor=:inferno)
Plots.plot(p1, p2, layout = (2,1), legend=:none)

# Initial guess
x₀ = [0.0, 0.0]

# LBFGS optimization
opt = optimize(Rastrigin, x₀, LBFGS())
println("LBFGS on the Rastrigin function yields: \n", opt)

## TODO: Use different initial guesses

# Nelder-Mead optimization
opt = optimize(Rastrigin, x₀)
println("Nelder-Mead on the Rastrigin function yields: \n", opt)

## TODO: Use different initial guesses

# Problem 4
# Linearization functionality
function linearize(f, a::Float64, b::Float64, n::Int64, x::Float64)
    # Generate grid
    X = collect(range(a, b, length=n))
    v = zeros(n)

    # Fill in values
    for (i, xᵢ) in enumerate(X)
        v[i] = f(xᵢ)
    end

    # First point bigger than x and point before
    i₊ = findfirst(z -> z > x, X)
    i₋ = i₊ - 1

    # Linear interpolation
    v[i₋] + (x - X[i₋]) * (v[i₊] - v[i₋]) / (X[i₊] - X[i₋])
end

# Problem 5
