# Problem 1
# Make change functionality
function make_change(n::Int64, S::Vector{Int64})
    # Initialize values
    m = length(S)
    values = zeros(n + 1, m)

    # Base case: one way to make change for zero
    values[1,:] .= 1

    # Iterate over remaining values
    for i in 2:(n + 1)
        for j in 1:m
            # Initialize solution count
            count = 0

            # Check remaining balance
            if i - S[j] > 0
                # Increment counter if balance remains
                count += values[i - S[j], j]
            end

            # Check if j is the last coin
            if j > 1
                count += values[i, j - 1]
            end

            # Set values
            values[i, j] = count
        end
    end

    # Return values
    values
end

# Test function
n = 10
S = [2,5,3,6]
make_change(n, S)

# Problem 2
# Cut rod functionality
function cut_rod(P::Vector{Int64})
    # Initialize values
    n = length(P)
    values = zeros(n + 1)

    # Iterate over remaining values
    for i in 2:(n + 1)
        # Candidate maximum value
        value = -Inf

        # Loop over cuts
        for j in 1:(i - 1)
            # Get maximum
            value = max(value, P[j] + values[i - j])
        end

        # Set value
        values[i] = value
    end

    # Return final value
    values[n + 1]
end

# Test function
P = [1, 5, 8, 9, 10, 17, 17, 20]
cut_rod(P)

# Problem 3
# Knapsack functionality
function knapsack(V::Vector{Int64}, W::Vector{Int64}, C::Int64)
    # Initialize values
    n = length(V)
    values = zeros(C + 1, n + 1)

    # Iterate over remaining values
    for i in 2:(C + 1)
        for j in 2:(n + 1)
            # Base is not using item j
            values[i, j] = values[i, j - 1]

            # Check if there is enough capacity
            if W[j - 1] <= (i - 1)
                values[i, j] = max(values[i, j - 1], V[j - 1] + values[i - W[j - 1], j - 1])
            end
        end
    end

    # Return final value
    values[C + 1, n + 1]
end

# Test function
W = [10, 20, 30]
V = [60, 100, 120]
C = 50
knapsack(V, W, C)