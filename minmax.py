def minimax(depth, node, maximizing, values, alpha, beta):

    if depth == 0:
        return values[node]

    if maximizing:

        best = float('-inf')

        for child in values[node]:

            value = minimax(
                depth - 1,
                child,
                False,
                values,
                alpha,
                beta
            )

            best = max(best, value)
            alpha = max(alpha, best)

            if beta <= alpha:
                break

        return best

    else:

        best = float('inf')

        for child in values[node]:

            value = minimax(
                depth - 1,
                child,
                True,
                values,
                alpha,
                beta
            )

            best = min(best, value)
            beta = min(beta, best)

            if beta <= alpha:
                break

        return best


values = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': 3,
    'E': 5,
    'F': 2,
    'G': 9
}

result = minimax(
    2,
    'A',
    True,
    values,
    float('-inf'),
    float('inf')
)

print("Best value:", result)
