
from typing import List


def zero_one_knapsack_with_recursion(capacity: int, weights: List[int], values: List[int]):
    """

    Args:
        capacity:
        weights:
        values:

    Returns:

    """
    n = len(weights)
    memo = {}

    def dfs(i, cap):
        if i == n or cap == 0:
            return 0

        if (i, cap) in memo:
            return memo[(i, cap)]

        # condition-0: do not select current shopping
        res = dfs(i+1, cap)

        # condition-1: select current shopping
        if weights[i] <= cap:
            res = max(res, dfs(i+1, cap - weights[i]) + values[i])

        memo[(i, cap)] = res

        return res

    return dfs(0, capacity)


def zero_one_knapsack_with_2d(capacity: int, weights: List[int], values: List[int]):
    n = len(weights)

    # dp[i][c] -> dp[i][c] denotes the maximum value for the first i items (0..i-1) under capacity c

    # initial dp
    dp = [[0 for _ in range(capacity+1)] for _ in range(n+1)]

    for i in range(n):
        w = weights[i]
        v = values[i]
        for c in range(1, capacity+1):
            # dp equation
            if c < w:
                dp[i+1][c] = dp[i][c]
            else:
                dp[i+1][c] = max(dp[i][c], dp[i][c-w] + v)

    return dp[n][capacity]


def zero_one_knapsack_with_1d(capacity: int, weights: List[int], values: List[int]):
    n = len(weights)

    # dp[c] -> dp[i][c] denotes the maximum value under capacity c

    # initial dp
    dp = [0 for _ in range(capacity + 1)]

    for i in range(n):
        w = weights[i]
        v = values[i]
        for c in range(capacity, w-1, -1):
            # dp equation
            # if c < w:
            #     p[i + 1][c] = dp[i][c]
            #     # remove i -> d[c] = dp[c]
            dp[c] = max(dp[c], dp[c-w]+v)

    return dp[capacity]


if __name__ == "__main__":
    capacity = 10
    weights = [2, 3, 4, 5]
    values = [4, 6, 8, 8]

    res_recursion = zero_one_knapsack_with_recursion(capacity, weights, values)
    res_2d = zero_one_knapsack_with_2d(capacity, weights, values)
    res_1d = zero_one_knapsack_with_1d(capacity, weights, values)

    print(res_recursion)
    print(res_2d)
    print(res_1d)

