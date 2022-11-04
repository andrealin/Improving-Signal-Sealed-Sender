import numpy as np
from collections import Counter

rng = np.random.default_rng()


# todo: rank does not work. or argsort does not work.
class Graph:
    def __init__(self, total_users, connectivity):
        num_edges = int(connectivity * total_users)
        self.edges = []
        self.adjacency_list = []
        for i in range(total_users):
            self.adjacency_list.append(set())
        for i in range(num_edges):
            u = rng.integers(0, total_users)
            v = u
            while v == u:
                v = rng.integers(0, total_users)

            self.edges.append((u, v)) # undirected edge
            self.adjacency_list[u].add(v)
            self.adjacency_list[v].add(u)
    def random_edge(self):
        random_index = rng.integers(0, len(self.edges))
        return self.edges[random_index]

def generate(users_per_epoch, graph):
    '''
    parameter: number of users per epoch
    parameter: number of total users

    paper:
    800 messages per epoch
    25% repeat
    25% response
    50% random
    fully connected graph
    :return: counts, 1 if user active this epoch, 0 otherwise.
    '''
    active_users = set()
    messages = []
    while len(active_users) < users_per_epoch:
        (u, v) = graph.random_edge()
        active_users.add(u)
        active_users.add(v)
        messages.append((u, v))
    return active_users, messages

def simulate_attack(
        users_per_epoch=800,
        total_users=1000000,
        connectivity=1):
    '''

    :param users_per_epoch:
    :param total_users:
    :param connectivity: higher connectivity means people have more contacts.
    :return:
    '''

    graph = Graph(total_users, connectivity)

    # find a suitable bob
    bob = None
    associates = set()

    while len(associates) == 0:
        bob = rng.integers(0, total_users)
        associates = graph.adjacency_list[bob]

    count = Counter()

    epochs = 0
    max_associate_place = total_users
    max_associate_places = [max_associate_place]

    while max_associate_place > len(associates):

        epochs += 1

        if epochs > 500:
            print(f"{epochs} epochs ran.")
            return max_associate_places

        # target epoch
        is_target_epoch = False
        while not is_target_epoch:
            active_users, _ = generate(users_per_epoch, graph)
            if bob in active_users:
                is_target_epoch = True
                count.update(active_users)

        # random epoch
        active_users, _ = generate(users_per_epoch, graph)
        count.subtract(active_users)

        # bob cannot be talking to themself
        count.subtract({bob})
        count.subtract({bob})
        count.subtract({bob})
        count.subtract({bob})

        print(f"epoch {epochs} counts {count}")

        # calculate associate places
        most_common = count.most_common()
        ppl = [x[0] for x in most_common]

        max_associate_place = 0
        for alice in associates:
            alice_loc = ppl.index(alice) if alice in ppl else -1
            if alice_loc == -1:
                # todo: what to do if edge never selected. does associate still count?
                alice_place = total_users
            else:
                alice_class = [x[0] for x in most_common if x[1] == most_common[alice_loc][1]]
                alice_place = len(alice_class) + ppl.index(alice_class[0])
            # print(f"epoch {epochs} associate {alice} place {alice_place}")

            if alice_place > max_associate_place:
                max_associate_place = alice_place

        max_associate_places.append(max_associate_place)


    print(f"summary ------- ")
    print(f"users per epoch {users_per_epoch}")
    print(f"total users {total_users}")
    print(f"connectivity {connectivity}")
    print(f"bob {bob} has {len(associates)} associate(s): {associates}")
    print(f"max associate places {max_associate_places}")

    return epochs

def num_epochs_vs_num_users_per_epoch_test():
    trials = 10
    users_per_epochs = [10, 100, 500, 1000]
    total_users = 10000
    connectivity = 1

    results = {}

    for users_per_epoch in users_per_epochs:
        num_epochss = []
        for i in range(trials):
            epochs = simulate_attack(
                users_per_epoch=users_per_epoch,
                total_users=total_users,
                connectivity=connectivity)
            num_epochss.append(epochs)
        results[users_per_epoch] = sum(num_epochss)/len(num_epochss)

    return results

def experiment():

    # print(f"epochs required: {simulate_attack(8, 10000, 3)}")
    # print(f"epochs required: {simulate_attack(8, 10000, 1)}")

    print(num_epochs_vs_num_users_per_epoch_test())

experiment()