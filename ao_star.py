def ao_star(graph, h, node, solved):

    if node in solved:
        return h[node]

    if node not in graph or not graph[node]:
        solved.add(node)
        return 0

    best_cost = float('inf')
    best_choice = None

    for option in graph[node]:

        cost = 0

        for child, edge in option:
            cost += edge + h[child]

        if cost < best_cost:
            best_cost = cost
            best_choice = option

    h[node] = best_cost

    for child, edge in best_choice:
        ao_star(graph, h, child, solved)

    solved.add(node)

    return best_cost


graph = {
    'A': [
        [('B', 1)],
        [('C', 2), ('D', 2)]
    ],

    'B': [
        [('E', 2)]
    ],

    'C': [],
    'D': [],
    'E': []
}

h = {
    'A': 0,
    'B': 2,
    'C': 1,
    'D': 1,
    'E': 0
}

cost = ao_star(graph, h, 'A', set())

print("Minimum Cost:", cost)
