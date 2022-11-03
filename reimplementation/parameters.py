import numpy as np
from dateutil import parser

'''
parameters used in the evolution probability function:

p, lambda_s, lambda_b:
* constants for the best fit line to the histogram of rate difference from t to t-1.
* calculated using expectation maximization algorithm in the paper. copied over
* tweaked by hand in my implementation.

p_zz
* probability a zero rate in epoch t becomes zero in epoch t + 1

p_pz
* probability a positive rate in epoch t becomes zero in epoch t + 1

other parameters:

“Mixture”
* prior belief about a rate. a heavy tailed distribution.
* each new rate for epoch t + 1 is sampled from this prior, and then the rates are weighted according to likelihood the rate fits the observed volumes, as well as according to the evolution probability function
* i assume one way to find this mixture would be to plot all the rates and fit a line of best fit
* currently in my implementation i just came up with one by hand
* could we just use uniform?

p_z
* probability a rate is zero
* i don’t know exactly how they use this. perhaps in the “Mixture”?
* i don’t use this in my code
# TODO: prior for initial rate


epoch length:
* smaller means more consecutive epochs with positive rates, which gives the attack more chances due to evolution
* but bigger means more messages per epoch, which increases the rate itself.

TODO: histogram rate / day of week, rate / day and time, rate / time (24 hrs), rate / 7 days
TODO: pandas dataframe .histogram
'''


class Parameters:
    def __init__(self):

        # used in rate evolution
        self.p_pz = 0.04
        self.p_pp = 1 - self.p_pz
        self.p_zz = 0.995
        self.p_zp = 1 - self.p_zz

        # the following parameters fit a line to the histogram of differences
        # between rate in t-1 and t
        self.p = 0.88
        self.lambda_s = 1.0
        self.lambda_b = 4.0

        # above means talking, below means not talking in that epoch
        self.threshold = 1.0

        # i don't use this. i assume it should be used in prior? aka Mixture
        self.p_z = 0.958

        # how the attack is run
        self.start_time = parser.parse("2001-06-10 00:00:00-07:00")
        self.allowed_months = "all"
        self.epoch_length = "week"
        self.multiple_batches_per_epoch = True
        self.num_epochs = 16

        self.only_silent = False
        self.only_conversation = False
        self.num_trials = 50

        # data format. default returns just the one number, the mean of the datapoints.
        self.multiple_datapoints = False

        # considerations
        self.dependent_epochs = True


    def __str__(self):
        representation = ""
        for key, value in self.__dict__.items():
            representation += f"\n{key}: {value}"
        return representation

    def Mixture(self):  # TODO: room for improvement
        """
        prior belief about lambda_AB rate
        can override with a different prior
        :return:
        """
        # "some particles representing no communication, some
        # with typical communication volumes, and some with
        # outlandishly high rates (to ensure robustness)."

        # uniform_sample = random.random()
        # width = 20
        # acceptance_probability = Evolution(uniform_sample * width, 0)

        here_is_the_mixture = [0, .5, 1, 2, 3, 4, 5, 8, 10]
        here_are_the_probabilities = [.3, .3, .2, .1, .05, .02, .02, .01]
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
