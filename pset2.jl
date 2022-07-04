# Load required libraries
using Optim, Interpolations, Plots

# # Problem 1
# # Himmelblau functionality
# function Himmelblau(p::Vector{Float64})
#     (p[1]^2 + p[2] - 11)^2 + (p[1] + p[2]^2 - 7)^2
# end

# # Instantiation of grids
# x, y = collect(-4:0.01:4), collect(-4:0.01:4)
# nx, ny = length(x), length(y)
# z = zeros(nx, ny)

# # Fill in function values
# for i = 1:nx, j = 1:ny
#     z[i,j] = Himmelblau([x[i], y[j]])
# end

# # Surface plot
# Plots.surface(x, y, z, seriescolor=:viridis, camera = (50,50))

# # Gradient function
# function g!(G, p::Vector{Float64})
#     G[1] = 4 * p[1] * (p[1]^2 + p[2] - 11) + 2 * (p[1] + p[2]^2 - 7)
#     G[2] = 2 * (p[1]^2 + p[2] - 11) + 4 * p[2] * (p[1] + p[2]^2 - 7)
# end

# # Hessian function
# function h!(H, p::Vector{Float64})
#     H[1] = 4 * (p[1]^2 + p[2] - 11) + 8 * p[1]^2 + 2
#     H[2] = 4 * p[1] + 4 * p[2]
#     H[3] = 4 * p[1] + 4 * p[2]
#     H[4] = 2 + 4 * (p[1] + p[2]^2 - 7) + 8 * p[2]^2
# end

# # Initial guesses
# X₀ = [[0.0, 0.0], [1.0, 1.0], [100.0, 0.0], [0.0, 100.0], [-1.0, -1.0], [-100.0, 0.0], [0.0, -100.0], [-1.0, 1.0], [1.0, -1.0]]

# # Different guesses
# for x₀ in X₀
#     # Newton's method optimization
#     @elapsed opt = optimize(Himmelblau, g!, h!, x₀)
#     println("Newton's method on the Himmelblau function with starting point $x₀ yields: \n", opt)

#     # Nelder-Mead optimization
#     @elapsed opt = optimize(Himmelblau, x₀)
#     println("Nelder-Mead on the Himmelblau function with starting point $x₀ yields: \n", opt)
# end

# # Problem 2
# # Ackley functionality
# function Ackley(p::Vector{Float64})
#     -20 * exp(-0.2 * sqrt(0.5 * (p[1]^2 + p[2]^2))) - exp(0.5 * (cos(2 * π * p[1]) + cos(2 * π * p[2]))) + ℯ + 20
# end

# # Instantiation of grids
# x, y = collect(-4:0.01:4), collect(-4:0.01:4)
# nx, ny = length(x), length(y)
# z = zeros(nx, ny)

# # Fill in function values
# for i = 1:nx, j = 1:ny
#     z[i,j] = Ackley([x[i], y[j]])
# end

# # Surface and contour plots
# p1 = Plots.surface(x, y, z, seriescolor=:viridis, camera = (50,50))
# p2 = Plots.contourf(x, y, z, seriescolor=:inferno)
# Plots.plot(p1, p2, layout = (2,1), legend=:none)

# # Different guesses
# for x₀ in X₀
#     # LBFGS optimization
#     opt = optimize(Ackley, x₀, LBFGS())
#     println("LBFGS on the Ackley function with starting point $x₀ yields: \n", opt)

#     # Nelder-Mead optimization
#     opt = optimize(Ackley, x₀)
#     println("Nelder-Mead on the Ackley function with starting point $x₀ yields: \n", opt)
# end

# # Problem 3
# # Rastrigin functionality
# function Rastrigin(p::Vector{Float64})
#     10 * length(p) + sum([pᵢ^2 - 10 * cos(pᵢ) for pᵢ in p])
# end

# # Instantiation of grids
# x = collect(-5.12:0.01:5.12)
# nx = length(x)
# y = zeros(nx)

# # Fill in function values
# for i = 1:nx
#     y[i] = Rastrigin([x[i]])
# end

# # Plot function
# Plots.plot(x, y, legend=:none)

# # Instantiation of grids
# x, y = collect(-5.12:0.01:5.12), collect(-5.12:0.01:5.12)
# nx, ny = length(x), length(y)
# z = zeros(nx, ny)

# # Fill in function values
# for i = 1:nx, j = 1:ny
#     z[i,j] = Rastrigin([x[i], y[j]])
# end

# # Surface and contour plots
# p1 = Plots.surface(x, y, z, seriescolor=:viridis, camera = (50,50))
# p2 = Plots.contourf(x, y, z, seriescolor=:inferno)
# Plots.plot(p1, p2, layout = (2,1), legend=:none)

# # Initial guess
# for x₀ in X₀
#     # LBFGS optimization
#     opt = optimize(Rastrigin, x₀, LBFGS())
#     println("LBFGS on the Rastrigin function with starting point $x₀ yields: \n", opt)

#     # Nelder-Mead optimization
#     opt = optimize(Rastrigin, x₀)
#     println("Nelder-Mead on the Rastrigin with starting point $x₀ function yields: \n", opt)
# end

# # Problem 4
# # Linearization functionality
# function linearize(f, a::Float64, b::Float64, n::Int64, x::Float64)
#     # Generate grid
#     X = collect(range(a, b, length=n))
#     v = zeros(n)

#     # Fill in values
#     for (i, xᵢ) in enumerate(X)
#         v[i] = f(xᵢ)
#     end

#     # First point bigger than x and point before
#     i₊ = findfirst(z -> z > x, X)
#     i₋ = i₊ - 1

#     # Linear interpolation
#     v[i₋] + (x - X[i₋]) * (v[i₊] - v[i₋]) / (X[i₊] - X[i₋])
# end

# Problem 5
# Approximation of log(1 + x)
function approximate(grid::Vector{Float64})
    # Generate grids, interpolations
    X̂ = interpolate(grid, BSpline(Linear()))
    f̂n = interpolate([log(1 + x) for x in grid], BSpline(Linear()))

    # Define grid and collect indices
    X = collect(0.0:0.1:100.0)
    indices = [optimize(index -> abs(X̂(index) - x), 1.0, length(grid)).minimizer for (i, x) in enumerate(X)]

    # Define objects
    f̂ = [f̂n(i) for i in indices]
    f = [log(1 + x) for x in X]
    errors = abs.(f - f̂)

    # Return values
    errors, f̂, f
end

# Approximate for coarse grid
X = collect(0.0:0.1:100.0)
X̂ = collect(0.0:10.0:100.0)
errors, f̂, f = approximate(X̂)

# Report sum of squared errors and plot
println("The sum of errors is: ", sum(errors))
p1 = plot(X, errors)
p2 = plot(X, [f̂, f])
Plots.plot(p1, p2, layout = (2,1), legend=:none)

# Objective for optimal points
function objective(grid::Vector{Float64})
    # Infinite value if out of range
    if sum([xᵢ > 100 || xᵢ < 0 for xᵢ in grid]) > 0
        return Inf
    else
        # Concatenate grid and approximate
        grid = vcat(0.0, grid, 100.0)
        errors, f̂, f = approximate(grid)
        return sum(errors)
    end
end

# Optimization
X̂ = collect(10.0:10.0:90.0)
opt = optimize(x -> objective(x), X̂; g_tol = 1e-3, x_tol = 1e-2)

# Report and plot
grid = vcat(0.0, opt.minimizer, 100.0)
errors, f̂, f = approximate(grid)
p1 = plot(X, errors)
p2 = plot(X, [f̂, f])
Plots.plot(p1, p2, layout = (2,1), legend=:none)
