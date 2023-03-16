import numpy as np
import random


class Rrtstar:
    def __init__(self, *, start_point=None, end_point=None,
                 bin_map=None, search_radius=10, goal_radius=10,
                 neighborhood_radius=20, total_nodes=500):
        self.start_node = self.Node(*start_point)
        self.goal_node = self.Node(*end_point)
        self.bool_map = bin_map
        self.search_radius = search_radius
        self.goal_radius = goal_radius
        self.goal_neighbors = dict()
        self.neighborhood_radius = neighborhood_radius
        self.total_nodes = total_nodes
        self.graph = {self.node_index(self.start_node): self.start_node}
        self.node_counter = 1
        self.dist_reached = False
        self.path = []

        self.nodes = np.array([start_point]).astype(np.uint32)

    class Node:
        def __init__(self, x, y, cost=0, parent=None):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent = parent

    def node_index(self, node):
        return node.y * self.bool_map.shape[1] + self.bool_map.shape[0]

    def step(self):
        rand_x = random.randrange(self.bool_map.shape[1]//2, self.bool_map.shape[1])
        rand_y = random.randrange(self.bool_map.shape[0])
        rand_node = self.Node(rand_x, rand_y)
        i_nearest, nearest_node = self.find_nearest_node(rand_node)
        interpolated_node = self.calc_interpolated_node(nearest_node, rand_node)
        i_interpolated = self.node_index(interpolated_node)

        if not interpolated_node:
            return
        self.nodes = np.append(self.nodes, [[interpolated_node.x, interpolated_node.y]], axis=0)

        interpolated_node.parent = i_nearest                # temporary parent
        neighbors = self.find_neighbors(interpolated_node)
        i_parent, parent = self.choose_parent(neighbors, interpolated_node)

        interpolated_node.parent = i_parent
        interpolated_node.cost = self.distance(parent, interpolated_node) + parent.cost
        self.graph[i_interpolated] = interpolated_node
        # display connection between parent and interpolated_node

        self.rewire(neighbors, interpolated_node)

        if self.distance(interpolated_node, self.goal_node) < self.goal_radius and \
                not self.any_collisions_between(interpolated_node, self.goal_node):
            self.goal_neighbors[i_interpolated] = interpolated_node

        self.node_counter += 1
        if self.node_counter == self.total_nodes:
            self.dist_reached = True

    def get_path(self):
        if len(self.goal_neighbors) == 0:
            print('Path not found')
            return None

        i_parent, goal_parent = self.best_goal_neighbor()
        self.goal_node.parent = i_parent
        self.graph[self.node_index(self.goal_node)] = self.goal_node

        iter_node = self.goal_node
        self.path.append([iter_node.x, iter_node.y])
        while iter_node.parent:
            iter_node = self.graph[iter_node.parent]
            self.path.append([iter_node.x, iter_node.y])
        return self.path

    def find_nearest_node(self, target_node):
        i, node = min(self.graph.items(), key=lambda l: self.distance(l[1], target_node))
        return i, node

    @staticmethod
    def distance(node1, node2):
        return np.linalg.norm([node1.x - node2.x, node1.y - node2.y])

    def calc_interpolated_node(self, real, imagine):
        d = self.distance(real, imagine)
        r = self.any_collisions_between(real, imagine)
        if d < self.search_radius and not r:
            return imagine
        elif r > self.search_radius:
            r = self.search_radius

        x = int(real.x + (imagine.x - real.x) * r / d)
        y = int(real.y + (imagine.y - real.y) * r / d)
        if self.bool_map[y][x] == 1 and r == 1:
            return None
        return self.Node(x, y)

    def any_collisions_between(self, node1, node2):
        # checking all cells for an obstacle at a distance less than search_radius
        # (imagine.x - real.x)/d - unit direction vector
        d = self.distance(node1, node2)
        for r in range(int(d)):
            x = int(node1.x + (node2.x - node1.x) * r / d)
            y = int(node1.y + (node2.y - node1.y) * r / d)
            if self.bool_map[y][x] == 1:
                # last distance without obstacle on the path
                return r-1
        return False

    def find_neighbors(self, node):
        # neighbors = {i: node for i, node in self.graph.items()
        #              if self.distance(node, target) < self.neighborhood_radius}
        neighbors = dict(filter(lambda neighbor: self.distance(neighbor[1], node) < self.neighborhood_radius,
                                self.graph.items()))
        return neighbors

    def choose_parent(self, neighbors_dict, new_node):
        neighbors = dict(filter(lambda neighbor: not self.any_collisions_between(neighbor[1], new_node),
                                neighbors_dict.items()))
        i, best_parent = min(neighbors.items(),
                             key=lambda neighbor: neighbor[1].cost + self.distance(neighbor[1], new_node))
        return i, best_parent

    def rewire(self, neighbors_dict, new_node):
        del neighbors_dict[new_node.parent]
        necessary_neighbors = {i: node for i, node in neighbors_dict.items()
                               if new_node.cost + self.distance(new_node, node) < node.cost and
                               not self.any_collisions_between(new_node, node)}
        for i in necessary_neighbors:
            node = necessary_neighbors[i]
            # old_parent = self.graph[node.parent]
            node.parent = self.node_index(new_node)
            node.cost = new_node.cost + self.distance(node, new_node)
            # display connection between node and new_node
            # and delete connection between old_parent and node

    def best_goal_neighbor(self):
        i, best_neighbor = min(self.goal_neighbors.items(),
                               key=lambda neighbor: neighbor[1].cost + self.distance(neighbor[1], self.goal_node))
        return i, best_neighbor


if __name__ == "__main__":
    space = np.array([[0] * 120] * 150)
    print(space.shape[0], space.shape[1])
    alg = Rrtstar(start_point=(160, 159), end_point=(125, 190), bin_map=space)
    print(alg.graph)
