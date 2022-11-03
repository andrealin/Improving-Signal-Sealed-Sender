from dateutil import parser
from datetime import datetime
# import traceback
import numpy as np
from scipy.stats import gamma, poisson, binom, expon, pareto
import logging
import random
import ubuntu
import enron
import pandas as pd
import matplotlib.pyplot as plt

from parameters import Parameters

parse = parser.parse
logging.basicConfig(filename='ssd.log', filemode='w', level=logging.DEBUG)

more_info = True

# output
# p_zz = 0.99771457398589
# p_pz = 0.7129411764705882

num_particles = 100 # 250 # "250-550 performs well"
timestamp_format = "Date: %a, %d %b %Y %H:%M:%S %z (%Z)"
email_tag = "@enron.com"


class Message:

    def __init__(self, timestamp, sender, receiver, text=''):
        self.timestamp = timestamp
        self.sender = sender
        self.receiver = receiver
        self.text = text  # optional. just for curiosity.

    def __str__(self):
        return f'{self.timestamp} {self.sender} {self.receiver} {self.text}'



# ssdfilter

# lemme see if i can forget batches and just do epochs. why did they do batches anyway?

def lambda_ab_prior(): # ? idk
    return 1

def lambda_ob_prior(): # ?? idk
    return 5

def gamma_sample_weight(alpha, beta):
    sample = np.random.gamma(alpha, beta)
    assert beta == 1
    weight = gamma.pdf(sample, alpha)
    return sample, weight

# email parameters from ssd paper

def L_to_the_n(n, volumes, t, lambdas, p_ab, p_ob):
    summation = 0
    for k in range(min(volumes.A[t][n], volumes.B[t][n])+1):
        first_factor = binom.pmf(k, volumes.A[t][n], p_ab)
        second_factor = binom.pmf(volumes.B[t][n] - k, volumes.O[t][n], p_ob) # assume typo in paper
        summation += first_factor * second_factor
    # print(f"summation {summation}")
    return summation

def Likelihood(volumes, t, lambdas):
    p_ab = lambdas.AB / (lambdas.AB + lambdas.AO)
    p_ob = lambdas.OB / (lambdas.OB + lambdas.OO)

    product = 1
    for n in range(len(volumes.A[t])):
        product *= L_to_the_n(n, volumes, t, lambdas, p_ab, p_ob)

    # print(f"product {product}")

    L = poisson.pmf(sum(volumes.A[t]), lambdas.AB + lambdas.AO) * poisson.pmf(sum(volumes.O[t]), lambdas.OB + lambdas.OO) * product

    # if not L > 0:
        # print("not L > 0")
        # print("L to the 1: ", L_to_the_1)

    # L_to_the_1 = 0 # doing away with batches

    # for k in range(min(volumes.A[t], volumes.B[t])+1):
    #     L_to_the_1 += binom.pmf(k, volumes.A[t], p_ab) * binom.pmf(volumes.B[t] - k, volumes.O[t], p_ob)
    #     # print(f"k {k} A {volumes.A[t]} p_ab {p_ab} B {volumes.B[t]} O {volumes.O[t]} p_ob {p_ob}")
    # print("L to the 1: ", L_to_the_1)

    # print(f"L = {L}")
    return L


def Evolution(lambda_now, lambda_previous, parameters:Parameters):

    # TODO: are big delta cause one is zero. make separate histograms.
    delta = abs(lambda_now - lambda_previous)
    big_delta = parameters.p * expon.pdf(delta, scale=1/parameters.lambda_s) + (1 - parameters.p) * expon.pdf(delta, scale=1/parameters.lambda_b)
    if lambda_now == 0 and lambda_previous == 0:
        return parameters.p_zz
    elif lambda_now > 0 and lambda_previous == 0:
        # TODO: can improve? maybe remove big delta
        # return parameters.p_zp # i'm thinking, difference doesn't matter when starting out convo
        return parameters.p_zp * big_delta
    elif lambda_now == 0 and lambda_previous > 0:
        # TODO: can improve? maybe remove big delta
        return parameters.p_pz * big_delta
    else:
        return parameters.p_pp * big_delta


def Resample(lambda_prime_AB, lambda_prime_OB):
    pass

class Volumes:
    def __init__(self, num_epochs, multiple_batcher_per_epoch):
        batches_per_epoch = 7 if multiple_batcher_per_epoch else 1
        self.A = [[0] * batches_per_epoch] * num_epochs
        self.O = [[0] * batches_per_epoch] * num_epochs
        self.B = [[0] * 7] * num_epochs
        self.O_prime = [[0] * batches_per_epoch] * num_epochs

        # TODO: should be real number of messages. this is not a rate.
        self.real_rates = [0] * num_epochs # only used for debugging/evaluation

        self.guessed_AB = []
        self.guessed_OB = []
        self.num_particles = []

    def __str__(self, epoch_range=None):
        representation = ""
        epoch_range = epoch_range or range(len(self.A))
        for epoch in epoch_range:
            representation += f"\nepoch: {epoch}"
            representation += "\nAB: " + str(self.real_rates[epoch]) + "\t"
            if len(self.guessed_AB) > 0:
                representation += "guessed AB: " + str(self.guessed_AB[epoch]) + "\t"
            representation += "B: " + str(self.B[epoch]) + "\t"
            if len(self.guessed_OB) > 0:
                representation += "guessed OB: " + str(self.guessed_OB[epoch]) + "\t"
            representation += "A: " + str(self.A[epoch]) + "\t"
            representation += "O: " + str(self.O[epoch]) + "\t"
            representation += "O': " + str(self.O_prime[epoch]) + "\t"
            if len(self.num_particles) > 0:
                representation += "num: " + str(self.num_particles[epoch]) + "\t"
        return representation

    def latest(self):
        return self.__str__(epoch_range=range(len(self.guessed_AB) - 1, len(self.guessed_AB)))



class Lambdas:
    def __init__(self, AB, AO, OB, OO):
        self.AB = AB
        self.AO = AO
        self.OB = OB
        self.OO = OO


class Particle:

    def __init__(self, lambda_AB, lambda_OB):
        self.lambda_AB = lambda_AB
        self.lambda_OB = lambda_OB
        self.rejected = False

    def resample(self, lambda_prime_AB, lambda_prime_OB, w):
        accepted = np.random.random() <= w
        if accepted:
            self.lambda_AB = lambda_prime_AB
            self.lambda_OB = lambda_prime_OB
        else:
            self.rejected = True # i don't think resample is actually supposed to reject?

    def __str__(self):
        return "\n" + str(self.lambda_AB) + "\t" + str(self.lambda_OB)


def ssdfilter(volumes: Volumes, parameters: Parameters):
    # print(f"{datetime.now()} running ssdfilter attack...")

    total_silent = 0 # # epochs Alice and Bob were not talking
    total_false_positives = 0 # # epochs attack guessed talking but actually were not
    total_talking = 0 # # epochs Alice and Bob were talking
    total_false_negatives = 0 # # epochs attack guessed not talking but actually were

    particles = [Particle(lambda_ab_prior(), lambda_ob_prior()) for i in range(num_particles)] # does this make all the same?
    # print(f"num particles: {len(particles)}")
    for t in range(parameters.num_epochs):
        # print(f"epoch: {t}")
        total_weight = 0
        for particle in particles:
            if particle.rejected:
                continue
            lambda_A, w_A = gamma_sample_weight(sum(volumes.A[t]) + 1, 1)
            lambda_O, w_O = gamma_sample_weight(sum(volumes.O[t]) + 1, 1)
            lambda_B, w_B = gamma_sample_weight(sum(volumes.B[t]) + 1, 1)
            lambda_prime_AB, w_theta = parameters.Mixture()
            # print(lambda_A, w_A, lambda_O, w_O, lambda_B, w_B, lambda_prime_AB, w_theta)
            # print("lambda_prime_AB: ", lambda_prime_AB)
            if lambda_prime_AB > lambda_A or lambda_prime_AB > lambda_B:
                particle.rejected = True
                # print("rejected")
                continue
            lambda_prime_OB = lambda_B - lambda_prime_AB
            lambda_AO = lambda_A - lambda_prime_AB

            if lambda_prime_OB > lambda_O:
                particle.rejected = True
                # print("rejected due to low numbers of others messaging")
                continue

            lambda_OO = lambda_O - lambda_prime_OB
            # if lambda_OO < 0:
            #     print(f"lambda OO negative {lambda_OO} lambda_O {lambda_O} lambda_OB {lambda_prime_OB}")
            lambdas = Lambdas(lambda_prime_AB, lambda_AO, lambda_prime_OB, lambda_OO)
            likelihood = Likelihood(volumes, t, lambdas)
            evolution_probability = Evolution(lambda_prime_AB, particle.lambda_AB, parameters)

            w_AB = likelihood # remove evolution probability

            if parameters.dependent_epochs:
                w_AB = likelihood * evolution_probability

            # print(f"likelihood {likelihood}")
            # print(f"evolution {evolution_probability}")
            # print(f"w_ab {w_AB}")

            w = w_AB / (w_A * w_O * w_B * w_theta)
            particle.w = w
            particle.lambda_AB = lambda_prime_AB
            particle.lambda_OB = lambda_prime_OB
            total_weight += w

        # print(f"total weight {total_weight}")


        new_particles = []
        for particle in particles:
            if not particle.rejected:
                # print("lambda_AB: ", particle.lambda_AB)
                # print("normalized weight: ", particle.w / total_weight)
                for i in range(num_particles):
                    # print(f"particle.w = {particle.w}")
                    if np.random.random() <= particle.w/total_weight:
                        new_particles.append(Particle(particle.lambda_AB, particle.lambda_OB))
        particles = new_particles
        lambda_AB = sum([particle.lambda_AB for particle in particles]) / len(particles)
        lambda_OB = sum([particle.lambda_OB for particle in particles]) / len(particles)
        volumes.guessed_AB.append(lambda_AB)
        volumes.guessed_OB.append(lambda_OB)
        volumes.num_particles.append(len(particles))
        logging.info("latest:")

        logging.info(volumes.latest())

        if volumes.real_rates[t] == 0:
            total_silent += 1
            if lambda_AB >= parameters.threshold:
                total_false_positives += 1
        else:
            total_talking += 1
            if lambda_AB < parameters.threshold:
                total_false_negatives += 1




    # return 0 # return lambda_AB max i t i don't understand TODO
    lambda_AB = sum([particle.lambda_AB for particle in particles])/len(particles)
    lambda_OB = sum([particle.lambda_OB for particle in particles])/len(particles)

    logging.info(volumes.latest())

    logging.info("\n" + str(lambda_AB) + " " + str(lambda_OB))

    square_error = (lambda_AB - volumes.real_rates[parameters.num_epochs - 1]) ** 2

    logging.info("\n\nSummary -----------------------\n")

    logging.info(f"parameters {parameters}")

    logging.info(f"total silent epochs: {total_silent}")
    if total_silent > 0:
        logging.info(f"false positive rate: {total_false_positives/total_silent}")
    logging.info(f"total talking epochs: {total_talking}")
    if total_talking > 0:
        logging.info(f"false negative rate: {total_false_negatives/total_talking}")
    print(f"{datetime.now()} ran attack")

    return square_error


months_str_length = len("yyyy-mm")

alt_months_str_start = len("Thu, 10 ")
alt_alt_months_str_start = len("Thu, 1 ")
alt_months_str_length = len("May 2001")

def get_all_volumes(messages, parameters=Parameters()):
    volumes = {}
    volumes.senders = {}
    volumes.receivers = {}
    volumes.real_rates = {}
    for message in messages:

        # https://docs.python.org/3/library/datetime.html
        # Only days, seconds, and microseconds remain
        if parameters.allowed_months is not "all":
            if (message.timestamp[:months_str_length] in parameters.allowed_months) \
                    or message.timestamp[
                       alt_months_str_start:alt_months_str_start + alt_months_str_length] in parameters.allowed_months \
                    or message.timestamp[
                       alt_alt_months_str_start:alt_months_str_start + alt_months_str_length] in parameters.allowed_months:
                pass
            else:
                continue
        delta = parse(message.timestamp) - parameters.start_time
        if parameters.epoch_length == "week":
            epoch = int(delta.days / 7)  # an epoch is a week # TODO timedelta.weeks
        elif parameters.epoch_length == "hour":
            epoch = int(delta.days * 24 + delta.seconds / 3600)
        elif parameters.epoch_length == "quarter":
            epoch = int(delta.days * 24 + delta.seconds / 3600 * 4)
        elif parameters.epoch_length == "minute":
            epoch = int(delta.seconds / 60)  # an epoch is a minute

        if epoch < 0 or epoch >= parameters.num_epochs:
            continue

        if parameters.multiple_batches_per_epoch:
            batch = int(delta.days)
            if parameters.epoch_length == "week":
                batch = batch % 7
            else:
                print(f"batches for epoch lengths {parameters.epoch_length} not supported")
        else:
            batch = 0

        # sort into buckets/rates/num messages sent and received

        A = message.sender
        B = message.receiver

        if A not in volumes.senders:
            volumes.senders[A][epoch][batch] = 1
        else:
            volumes.senders[A][epoch][batch] = volumes.senders[A][epoch][batch] + 1


        if B not in volumes.receivers:
            volumes.receivers[B][epoch][batch] = 1
        else:
            volumes.receivers[B][epoch][batch] = volumes.receivers[B][epoch][batch] + 1


        if (A, B) not in volumes.real_rates:
            volumes.real_rates[(A, B)][epoch][batch] = 1
        else:
            volumes.real_rates[(A, B)][epoch][batch] = volumes.real_rates[(A, B)][epoch][batch] + 1

    return volumes

def get_volumes(A, B, messages, parameters=Parameters()):
    volumes = Volumes(parameters.num_epochs, parameters.multiple_batches_per_epoch)

    # print(f"{datetime.now()} sorting messages...")
    for message in messages:

        # https://docs.python.org/3/library/datetime.html
        # Only days, seconds, and microseconds remain
        if parameters.allowed_months is not "all":
            if (message.timestamp[:months_str_length] in parameters.allowed_months)\
                    or message.timestamp[alt_months_str_start:alt_months_str_start + alt_months_str_length] in parameters.allowed_months\
                    or message.timestamp[alt_alt_months_str_start:alt_months_str_start + alt_months_str_length] in parameters.allowed_months:
                pass
            else:
                continue
        delta = parse(message.timestamp) - parameters.start_time
        if parameters.epoch_length == "week":
            epoch = int(delta.days / 7)  # an epoch is a week # TODO timedelta.weeks
        elif parameters.epoch_length == "hour":
            epoch = int(delta.days * 24 + delta.seconds / 3600)
        elif parameters.epoch_length == "quarter":
            epoch = int(delta.days * 24 + delta.seconds / 3600 * 4)
        elif parameters.epoch_length == "minute":
            epoch = int(delta.seconds / 60)  # an epoch is a minute

        if epoch < 0 or epoch >= parameters.num_epochs:
            continue

        if parameters.multiple_batches_per_epoch:
            batch = int(delta.days)
            if parameters.epoch_length == "week":
                batch = batch % 7
            else:
                print(f"batches for epoch lengths {parameters.epoch_length} not supported")
        else:
            batch = 0

        # print(f"epoch {epoch} sender {message.sender} receiver {message.receiver}")
        if message.sender == A:
            volumes.A[epoch][batch] = volumes.A[epoch][batch] + 1
            if message.receiver == B:
                logging.info(f"match: epoch {epoch} message {message}")
                volumes.real_rates[epoch] = volumes.real_rates[epoch] + 1
        else:
            volumes.O[epoch][batch] = volumes.O[epoch][batch] + 1
        if message.receiver == B:
            volumes.B[epoch][batch] = volumes.B[epoch][batch] + 1
        else:
            volumes.O_prime[epoch][batch] = volumes.O_prime[epoch][batch] + 1

    # print(f"{datetime.now()} sorted")

    return volumes


def guess_pair(
        A,
        B,
        messages,
        parameters=Parameters(),
        volumes=None
):

    """

    :param A:
    :param B:
    :param messages:
    :param parameters
    :return:
    """
    print(f"{datetime.now()} guess_pair {A} {B}")
    logging.info(f"{datetime.now()} guess_pair")
    logging.info(f"start: {parameters.start_time}")

    if volumes is None:
        volumes = get_volumes(A, B, messages, parameters)

    conversation_trace = True
    successive_silent_epochs = 0
    for epoch in range(parameters.num_epochs):
        if volumes.real_rates[epoch] == 0:
            successive_silent_epochs += 1
            if successive_silent_epochs > 3:
                conversation_trace = False
        else:
            successive_silent_epochs = 0

    logging.info("------TRIAL------ sender " + A + " receiver " + B)
    logging.info("conversation trace: " + str(conversation_trace))

    if parameters.only_conversation and not conversation_trace:
        return
    if parameters.only_silent and conversation_trace:
        return

    # logging.info(volumes)

    error = ssdfilter(volumes, parameters)
    logging.info("square error: " + str(error))
    return error


def run_enron(parameters=Parameters()):
    # estimate_parameters(*parse_data())

    users, messages = enron.parse_data()
    # print(users, messages)

    all_volumes = get_all_volumes(messages, parameters)

    print("one run arnold ward")
    guess_pair("arnold", "ward", messages, parameters=parameters)
    # guess_pair("allen", "arnold", messages, parameters=parameters)
    # guess_pair("neal", "nemec", messages, parameters=parameters)

    users_tuple = tuple(users)
    mses = []

    for i in range(parameters.num_trials):

        if i % 10 == 0:
            print("trial: ", i)

        error = None

        while error is None:

            A = "dummy_variable"
            B = "dummy_variable"

            while A == B:

                if parameters.only_conversation:
                    random_message = random.choice(messages)
                    A = random_message.sender
                    B = random_message.receiver
                else:
                    A = random.choice(users_tuple)
                    B = random.choice(users_tuple)

            error = guess_pair(
                A,
                B,
                messages,
                parameters=parameters
            )

        mses.append(error)

    print(mses)
    print(parameters.num_trials)
    mean_mse = sum(mses) / parameters.num_trials
    logging.info("mean mse: " + str(mean_mse))

    if parameters.multiple_datapoints:
        logging.info("mses", str(mses))
        return mses
    else:
        return mean_mse

def run_ubuntu(parameters=Parameters()):
    messages = ubuntu.parse_data()

    parameters.start_time = parse("2004-11-23 00:00:00+00:00")
    parameters.allowed_months = "2004-11 2004-12 2005-01"
    parameters.allowed_months = "2004-11"

    guess_pair(
        "stuNNed",
        "crimsun",
        messages,
        parameters=parameters,
    )
    # guess_pair(
    #     "PatrikJohansson",
    #     "frank",
    #     messages,
    #     start_time = parser.parse("2005-10-31 00:00:00+00:00"),
    #     epoch_length="hour",
    #     num_epochs=2000
    # )
    # guess_pair(
    #     "Flyzoola",
    #     "histo",
    #     messages,
    #     start_time=parser.parse("2008-10-23 00:00:00+00:00"),
    #     epoch_length="hour",
    #     num_epochs=2000
    # )
    # guess_pair(
    #     "ejofee",
    #     "Seveas",
    #     messages,
    #     start_time=parser.parse("2005-12-16 00:00:00+00:00"),
    #     epoch_length="hour",
    #     num_epochs=2000
    # )
    # guess_pair(
    #     "ejofee",
    #     "Seveas",
    #     messages,
    #     start_time=parser.parse("2009-01-01 00:00:00+00:00"),
    #     epoch_length="hour",
    #     num_epochs=2000
    # )

def run_attack(dataset="enron", parameters=Parameters()):
    """

    :param dataset: "ubuntu" or "email"
    :return:
    """


    if dataset is "ubuntu":

        parameters.epoch_length = "hour"
        parameters.num_epochs = 50

        run_ubuntu(parameters)
    elif dataset is "enron":
        return run_enron(parameters)
    else:
        print(f"incorrect dataset {dataset}")




def run_independent_rounds(dataset="ubuntu", parameters=Parameters()):
    parameters.dependent_epochs = False
    # print(parameters)
    run_attack(dataset=dataset, parameters=parameters)

def run_tweaked_parameters(dataset="ubuntu"):

    parameters = Parameters()

    parameters.p = 0.88 # doesn't matter anymore
    parameters.lambda_s = 1.0 # stays the same
    parameters.lambda_b = 1.0 # changed

    def uniform():
        here_is_the_mixture = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        here_are_the_probabilities = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]
        random_number = np.random.random()
        index = 0
        current_sum = 0
        for probability in here_are_the_probabilities:
            current_sum += probability
            if current_sum > random_number:
                break
            else:
                index = index + 1

        return here_is_the_mixture[index], here_are_the_probabilities[index]

    parameters.Mixture = uniform

    run_attack(dataset=dataset, parameters=parameters)


def main_redacted():
    def pdf(x, width):
        # linear prior lol
        return 2 / width - 2 * x / width / width

    def continuous_prior(p_z):
        if np.random.random() < p_z:
            return 0, p_z

        sample = expon.rvs(scale=2)
        probability = expon.pdf(sample, scale=2)
        # print(f"{sample} {probability}")

        return sample, probability * (1 - p_z)

    # run_original_attack(dataset="ubuntu")
    # run_tweaked_parameters()
    # run_independent_rounds()

    parameters = Parameters()
    parameters.num_trials = 1

    run_original_attack(parameters=parameters)
    #
    # run_independent_rounds(dataset="enron", parameters=parameters)

    # run_attack(dataset="ubuntu", parameters=parameters)

    # run_attack(dataset="enron", parameters=parameters)

def run_original_attack():
    return run_attack()


def original_and_simplified_test():
    original_mean_error = run_original_attack()

    parameters = Parameters()
    parameters.multiple_batches_per_epoch = False

    modified_mean_error = run_attack(parameters=parameters)

    logging.info(f"mean square error: original {original_mean_error} vs modified {modified_mean_error}")


def replicate_graph_test():

    # silent_parameters = Parameters()
    # # silent_parameters.num_trials = 100
    # silent_parameters.num_trials = 20
    # silent_parameters.multiple_datapoints = True
    # silent_parameters.only_silent = True
    # silent_mean_square_errors = run_attack(parameters=silent_parameters)
    #
    # df = pd.DataFrame(silent_mean_square_errors)
    # df.boxplot()

    comm_parameters = Parameters()
    # comm_parameters.num_trials = 100
    comm_parameters.num_trials = 5
    comm_parameters.multiple_datapoints = True
    comm_parameters.only_conversation = True
    comm_mean_square_errors = run_attack(parameters=comm_parameters)

    df = pd.DataFrame(comm_mean_square_errors)
    df.boxplot()

    plt.show()

def main():
    replicate_graph_test()
    # original_and_simplified_test()


if __name__ == "__main__":
    main()

