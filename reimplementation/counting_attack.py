import enron
from parameters import Parameters
from dateutil import parser
import random



def attack():
    users, messages = enron.parse_data()
    # Bob = "arnold" # arnold just talks to themself
    # Bob = "mckay"
    # Bob = "wolfe"
    # Bob = "kaminski" # also talks to self
    Bob = "whalley"

    # users, messages = simulation.generate

    parameters = Parameters()

    # currently this start time doesn't matter
    parameters.start_time = parser.parse("2001-01-01 00:00:00-07:00")

    counts = {}
    for user in users:
        counts[user] = 0

    for message in messages[:1000]:
        message.parsed_time = parser.parse(message.timestamp)

    # remember to start including more messages
    messages_in_order = sorted(messages[:1000], key=lambda message: message.parsed_time)

    for message in messages_in_order[:10]:
        print(message)

    Bob_message_index = 0

    actual_senders = []


    for i in range(parameters.num_epochs):

        # samples num_epochs target epochs and num_epochs random epochs

        # target

        while messages_in_order[Bob_message_index].receiver != Bob:
            # print(f"is not Bob, is: {messages_in_order[Bob_message_index]}")
            Bob_message_index += 1
        # print("exited while")

        print(f"message to Bob: {messages_in_order[Bob_message_index]}")

        Bob_message_time = messages_in_order[Bob_message_index].parsed_time
        actual_senders.append(messages_in_order[Bob_message_index].sender)

        Bob_message_index += 1
        last_message_in_target_epoch = Bob_message_index + 1

        # assume week long epochs
        while (messages_in_order[last_message_in_target_epoch].parsed_time - Bob_message_time).days < 7:
            # print(f"increase count: {messages_in_order[last_message_in_target_epoch].receiver}")
            counts[messages_in_order[last_message_in_target_epoch].receiver] += 1

            # todo: currently, count increases w number of appearances in the epoch
            # should count just increase by one for being in epoch, regardless of number of times?
            last_message_in_target_epoch += 1

        # random

        random_index = random.randrange(len(messages_in_order))
        random_time = messages_in_order[random_index].parsed_time
        # todo out of range index. prolly if random time is chosen near the end.

        last_message_in_random_epoch = random_index + 1

        while (messages_in_order[last_message_in_random_epoch].parsed_time - random_time).days < 7:
            # print(f"decrease count: {messages_in_order[last_message_in_random_epoch].receiver}")
            counts[messages_in_order[last_message_in_random_epoch].receiver] -= 1
            last_message_in_random_epoch += 1

        d_view = [(v, k) for k, v in counts.items()]
        d_view.sort(reverse=True)  # natively sort tuples by first element
        print(f"num epochs {i} counts {d_view}")

    print("actual associates of Bob", actual_senders)


    # random epoch: any random epoch.

attack()