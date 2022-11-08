import numpy as np
from collections import Counter
import logging
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

rng = np.random.default_rng()

logging.basicConfig(filename='simulation.log', filemode='w', level=logging.DEBUG)



# todo: rank does not work. or argsort does not work.
class Graph:
    def __init__(self, total_users, connectivity):
        num_edges = int(connectivity * total_users)
        self.edges = []
        self.adjacency_list = []
        self.total_users = total_users
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

# wikipedia user pages
class RealGraph:
    def __init__(self):
        self.total_users = 2394385

        self.edges = []
        self.adjacency_list = []
        for i in range(self.total_users):
            self.adjacency_list.append(set())

        with open('../wiki-Talk.txt') as f:
            count = 0
            for line in f:
                if count % 1000000 == 0:
                    print(f"edges read {count}")
                count += 1
                if line[0] is "#":
                    continue
                line = line.strip('\n').split('\t')
                u = int(line[0])
                v = int(line[1])

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
        graph=None,
        max_bob_num_associates=6,
        total_users=1000000,
        connectivity=1):
    '''

    :param users_per_epoch:
    :param total_users:
    :param connectivity: higher connectivity means people have more contacts.
    :return:
    '''

    if graph is None:
        graph = Graph(total_users, connectivity)

    total_users = graph.total_users

    # find a suitable bob
    bob = None
    associates = set()
    associates_too_popular = False

    while len(associates) == 0 or len(associates) > max_bob_num_associates or associates_too_popular:
        bob = rng.integers(0, total_users)
        associates = graph.adjacency_list[bob]

        associates_too_popular = False # if associate too popular, take long time to deanonymize
        for associate in associates:
            if len(graph.adjacency_list[associate]) > 100:
                associates_too_popular = True

    alice = list(associates)[0]
    print(f"bob {bob} has {len(associates)} associate(s): {associates}")
    print(f"btw, alice {alice} has {len(graph.adjacency_list[alice])} associates")

    count = Counter()

    epochs = 0
    max_associate_place = total_users
    max_associate_places = [max_associate_place]

    while max_associate_place > len(associates):

        epochs += 1

        if epochs > 500:
            print(f"{epochs} epochs ran.")
            return max_associate_places

        # old way of generating target epoch:
        # # this search is the current bottleneck
        # # should be faster for large num users per epoch?
        # print(f"{datetime.now()} start look for target epoch")
        #
        # # target epoch
        # is_target_epoch = False
        # while not is_target_epoch:
        #     active_users, _ = generate(users_per_epoch, graph)
        #     if bob in active_users:
        #         is_target_epoch = True
        #         count.update(active_users)
        # print(f"{datetime.now()} end look for target epoch")


        # current way of generating target epoch

        # for large num users per epoch, we don't need to search
        # for epoch with bob. we can make an epoch with bob.
        # and generate users_per_epoch - 2 other users.
        active_users, _ = generate(users_per_epoch-2, graph)
        active_users.add(bob)
        random_associate_index = rng.integers(0, len(associates))
        random_associate = list(associates)[random_associate_index]
        print(f"random associate added {random_associate}")
        active_users.add(random_associate)
        count.update(active_users)


        # random epoch
        active_users, _ = generate(users_per_epoch, graph)
        count.subtract(active_users)

        # bob cannot be talking to themself
        count.subtract({bob})
        count.subtract({bob})
        count.subtract({bob})
        count.subtract({bob})

        print(f"epoch {epochs} counts {count.most_common(10)}")

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
    print(f"epochs required {epochs}")
    print(f"total users {total_users}")
    print(f"connectivity {connectivity}")
    print(f"bob {bob} has {len(associates)} associate(s): {associates}")
    print(f"btw, alice {alice} has {len(graph.adjacency_list[alice])} associates")
    print(f"max associate places {max_associate_places}")

    return epochs

def num_epochs_vs_num_users_per_epoch_test(
        graph=None,
        users_per_epoch_range=range(20, 300, 40),
        trials=10,
        max_bob_num_associates=6):

    total_users = 10000
    connectivity = 1

    results = []

    for users_per_epoch in users_per_epoch_range:
        num_epochss = []
        for i in range(trials):
            print(f"{users_per_epoch} users per epoch: trial {i}")
            epochs = simulate_attack(
                users_per_epoch=users_per_epoch,
                graph=graph,
                max_bob_num_associates=max_bob_num_associates,
                total_users=total_users,
                connectivity=connectivity)
            num_epochss.append(epochs)
        results.append([users_per_epoch, sum(num_epochss)/len(num_epochss)])

    logging.info(f"trials {trials}")
    logging.info(f"total users {total_users}")
    logging.info(f"connectivity {connectivity}")
    logging.info(results)

    df = pd.DataFrame(results)
    df.plot.line(0, 1)

    plt.show()
    return results

def experiment():

    # print(f"epochs required: {simulate_attack(8, 10000, 3)}")
    # print(f"epochs required: {simulate_attack(8, 10000, 1)}")

    graph = RealGraph()

    # num_epochs_vs_num_users_per_epoch_test(graph=graph)
    # num_epochs_vs_num_users_per_epoch_test(
    #     graph=graph,
    #     users_per_epoch_range=range(1000, 15000, 2000),
    #     trials=20,
    #     max_bob_num_associates=1,  # too many associates, takes too long to deanonymize
    # )
    num_epochs_vs_num_users_per_epoch_test(
        graph=graph,
        users_per_epoch_range=[1000, 2000, 4000, 8000, 16000, 32000],
        trials=20,
        max_bob_num_associates=1,  # too many associates, takes too long to deanonymize
    )

    # num_epochs_vs_num_users_per_epoch_test(
    #     graph=graph,
    #     users_per_epoch_range=[1000, 50000, 100000],
    #     trials=20,
    #     max_bob_num_associates=1,  # too many associates, takes too long to deanonymize
    # )

    #todo
    '''
    parameters:
    connectivity
    num users per epoch
    num users: 1 million. or 100K
    num users bob talks to
    num users alice talks to
    whether there are popular users or not
    social graph
    bursty behavior
    day night
    '''

experiment()