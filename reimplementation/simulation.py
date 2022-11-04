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
            self.edges.append((u, v))
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

    bob = rng.integers(0, total_users)
    associates = graph.adjacency_list[bob]
    if len(associates) == 0:
        print("bob does not talk to anyone")
        return

    max_associate_place = total_users

    count = Counter()

    epochs = 0
    alice_ranks = [max_associate_place]

    while max_associate_place > len(associates):

        epochs += 1

        if epochs > 500:
            print(f"{epochs} epochs ran.")
            return alice_ranks

        # target epoch
        is_target_epoch = False
        while not is_target_epoch:
            active_users, messages = generate(users_per_epoch, graph)
            if bob in active_users:
                is_target_epoch = True
        count.update(active_users)

        # random epoch
        active_users, messages = generate(users_per_epoch, graph)
        count.subtract(active_users)

        # bob cannot be talking to themself
        count.subtract({bob})
        count.subtract({bob})
        count.subtract({bob})
        count.subtract({bob})

        print(f"epoch {epochs} counts {count}")

        # calculate alice place
        most_common = count.most_common()

        max_associate_place = 0
        for alice in associates:

            ppl = [x[0] for x in most_common]
            alice_loc = ppl.index(alice) if alice in ppl else -1
            if alice_loc == -1:
                continue
            alice_class = [x[0] for x in most_common if x[1] == most_common[alice_loc][1]]
            alice_place = len(alice_class) + ppl.index(alice_class[0])

            if alice_place > max_associate_place:
                max_associate_place = alice_place

        alice_ranks.append(max_associate_place)


    print(f"summary ------- ")
    print(f"users per epoch {users_per_epoch}")
    print(f"total users {total_users}")
    print(f"connectivity {connectivity}")
    print(f"bob {bob} has {len(associates)} associate(s): {associates}")
    print(alice_ranks)

    return epochs


def experiment():

    print(f"epochs required: {simulate_attack(8, 10000)}")

experiment()