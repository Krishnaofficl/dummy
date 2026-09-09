import heapq

def a_star(graph, h, start, goal):

    queue = [(h[start], 0, start, [start])]
    visited = set()

    while queue:

        f, g, node, path = heapq.heappop(queue)

        if node in visited:
            continue

        visited.add(node)

        if node == goal:
            return g, path

        for neighbor, cost in graph[node]:

            if neighbor not in visited:

                new_g = g + cost
                new_f = new_g + h[neighbor]

                heapq.heappush(
                    queue,
                    (new_f,
                     new_g,
                     neighbor,
                     path + [neighbor])
                )

    return float('inf'), []


graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('F', 3)],
    'D': [('G', 1)],
    'E': [('G', 2)],
    'F': [('G', 2)],
    'G': []
}

h = {
    'A': 4,
    'B': 3,
    'C': 4,
    'D': 1,
    'E': 2,
    'F': 2,
    'G': 0
}

cost, path = a_star(graph, h, 'A', 'G')

print("Minimum Cost:", cost)
print("Path:", " -> ".join(path))
