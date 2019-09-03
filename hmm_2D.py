from itertools import product
import random
import numpy as np
import pandas as pd
from pomegranate import *

np.random.seed(1)
random.seed(1)

# load data from csv files
def load_data(filepath):
    return pd.read_csv(filepath)


# split data into train and test components. 90%-10% split
def split_data(data_set, proportion=0.9):
    n = int(proportion * len(data_set))
    train_set = data_set[:n, :]
    test_set = data_set[n:, :]
    return train_set, test_set

def create_model():
    # Create and return a fresh HMM model

    # Emission distributions for each hidden state
    # Emissions
    e0 = IndependentComponentsDistribution(
        [
            NormalDistribution(-5, 5),
            NormalDistribution(0, 1)
        ]
    )
    e1 = IndependentComponentsDistribution(
        [
            NormalDistribution(5, 5),
            NormalDistribution(0, 1)
        ]
    )
    e2 = IndependentComponentsDistribution(
        [
            NormalDistribution(0, 5),
            NormalDistribution(0, 1)
        ]
    )

    # Hidden states, with their emission distributions
    h0 = State(e0, name='decreasing')  # State 0
    h1 = State(e1, name='increasing')  # State 1
    h2 = State(e2, name='level')       # State 2
    # There's a 3rd state (start) automatically added
    # There's a 4th state (end) automatically added

    model = HiddenMarkovModel()
    model.add_states(h0, h1, h2)

    # You have to give it some initial values to start the training from
    model.add_transition(model.start, h0, 0.33)
    model.add_transition(model.start, h1, 0.33)
    model.add_transition(model.start, h2, 0.33)

    model.add_transition(h0, h0, 0.8)
    model.add_transition(h0, h1, 0.1)
    model.add_transition(h0, h2, 0.1)

    model.add_transition(h1, h0, 0.1)
    model.add_transition(h1, h1, 0.8)
    model.add_transition(h1, h2, 0.1)

    model.add_transition(h2, h0, 0.1)
    model.add_transition(h2, h1, 0.1)
    model.add_transition(h2, h2, 0.8)

    model.bake()

    return model


data_set = load_data('rfq_15min_count_and_notional_hmm.csv')

# Take differences
data_set_diffs = data_set.values[1:, :] - data_set.values[:-1, :]


data_set2 = load_data('rfq_15min_count_and_notional_hmm_c_n.csv')

x = data_set.values
y = data_set2.values[1:]

#print(x)
#print(y)
#print((x-y).max())
#print(x==y)

# Split into training and test
train_data, test_data = split_data(data_set_diffs)


# Train using the EM algorithm.  Pass in the transpose (.T) of the data

# There are hyperparameters here: distribution_inertia=0.25, max_iterations=10
# If you leave them at their default values the HMM overfits to so that the
# 'level' model is a normal with mean 0 and standard dev 0, to catch all the
# long sequences of 0s

# Do an equivalent of the GridSearchCV here, but being careful with the cross
# validation, because this data is one long sequence
real_train_data, val_data = split_data(train_data, proportion=0.7)

# Hyperparameters
distribution_inertia_vals = [0.01, 0.05, 0.1, 0.25, 0.5]
max_iterations_vals = [2, 5, 10]  # Doesn't seem to like this being too high -
                                  # the predictions at the end come out as nan.
                                  # you may need to inch this up one by one

# Init values
best_prob = -9999999999999  # So it will always be 'beaten' at first
best_distribution_inertia = None
best_max_iterations = None

for inertia, iterations in product(distribution_inertia_vals, max_iterations_vals):
    model = create_model()
    model.fit(
        [real_train_data],
        distribution_inertia=inertia,
        max_iterations=iterations
    )
    prob = model.log_probability(val_data)

    if np.isnan(prob):
        print('Prob found to be nan for:')
        print('Distribution inertia: %s' % inertia)
        print('Max iterations: %s' % iterations)
        continue  # Don't store these results

    # If this is the best prob (validation data is most likely) then keep these hyperparameter values
    if prob > best_prob:
        best_prob = prob
        best_distribution_inertia = inertia
        best_max_iterations = iterations

print('')
print('Best params found:')
print('Distribution inertia: %s' % best_distribution_inertia)
print('Max iterations: %s' % best_max_iterations)

# Refit with best params
model = create_model()
model.fit(
    [train_data],
    distribution_inertia=best_distribution_inertia,
    max_iterations=best_max_iterations
)

print("transition matrix")
print(model.dense_transition_matrix())

print('')
print('Model is:')
print(model)  # Gets the normal distributions of the emissions

# Try out model on test data
test_on = test_data[120:140, :]
print(test_on.T)
viterbi_predictions = model.predict(test_on, algorithm='viterbi')
print("viterbi pred: {}".format(''.join(map(str, viterbi_predictions))))


# Example prediction
def make_prediction(model, data, range_min=0, range_max=200):
    ## DON'T USE THIS - LEFT IN FOR INTEREST
    """Use HMM to make a prediction of the next value.

    Arguments:
        model -- A fitted HMM model
        data -- A list or 1D numpy array of data.  Unknown values are OK - just
            put them in as np.nan
        range_min {int} -- Value to predict is expected to be an integer in range
            range_min <= val <= range_max
        range_max {int} -- See range_min

    """
    best_prob_so_far = -9999999999999  # So it will always be 'beaten' at first
    prediction = None

    # The strategy here is to go through each candidate value of the
    # prediction, appending it to `data` and calculating how probable the
    # model thinks that overall sequence is. The candidate corresponding to
    # the most probable sequence is our prediction

    # NO! This approach doesn't work.  What you postulate drags the
    # distribution over towards itself.

    for pred in range(range_min, range_max+1):
        try_data = data + [pred]
        prob = model.log_probability(try_data)
        if prob > best_prob_so_far:
            best_prob_so_far = prob
            prediction = pred

    return prediction


# Example prediction
def make_prediction2(model, data):
    """Use HMM to make a prediction of the next value.

    Arguments:
        model -- A fitted HMM model
        data -- A list or 1D numpy array of data.  Unknown values are OK - just
            put them in as np.nan

    """
    # The strategy here is to take a sequence and get the probabilities of each
    # hidden state at the end of the sequence.  From this work out the probs
    # of the hidden states at the next timestep (using the transition matrix).
    # Then you can e.g. select the most likely hidden state and use the mean
    # of its emission distribution as the prediction.

    probs = model.predict_proba(data)  # This bit isn't working XXXXXXXXXXXXXXXXXXXXX
    hidden_state_probs = probs[None, -1, :]

    # Add zero probs for start and end states.  Transition matrix is 5x5
    hidden_state_probs = np.append(hidden_state_probs, [[0, 0]], axis=1)
    transmat = model.dense_transition_matrix()

    # Multiply hidden state probs by transition matrix to get probs of next
    # hidden states
    next_hidden_state_probs = np.matmul(hidden_state_probs, transmat)
    predicted_hidden_state = np.argmax(next_hidden_state_probs)

    # Oddly convoluted line to get the mean of the first emission (i.e.
    # the RFQ one) and teh nominals prediction
    rfq_prediction = model.states[predicted_hidden_state].distribution.parameters[0][0].parameters[0]
    nominals_prediction = model.states[predicted_hidden_state].distribution.parameters[0][0].parameters[1]

    return predicted_hidden_state, rfq_prediction, nominals_prediction


test_seq = test_data[120:140, :]
predicted_hidden_state, rfq_prediction, nominals_prediction = make_prediction2(model, test_seq)

print('Predicted hidden state is: ')
print(predicted_hidden_state)

print('RFQ prediction is: ')
print(rfq_prediction)

print('Nominals prediction is: ')
print(nominals_prediction)
