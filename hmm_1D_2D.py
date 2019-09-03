from itertools import product
import random
import numpy as np
import pandas as pd
from pomegranate import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

np.random.seed(1)
random.seed(1)

path = os.getcwd() + '/'
data_folder = 'datafiles/'

# load data from csv files
def load_data(filepath):
    return pd.read_csv(filepath)


# split data into train and test components. 90%-10% split
def split_list(data_set, proportion=0.9):
    n = int(proportion * len(data_set))
    train_set = data_set[:n]#, :]
    test_set = data_set[n:]#, :]
    return train_set, test_set

def split_data(data_set, proportion=0.9):
    n = int(proportion * len(data_set))
    train_set = data_set[:n, :]
    test_set = data_set[n:, :]
    return train_set, test_set


def create_model():
    # Create and return a fresh HMM model

    # Emission distributions for each hidden state, capturing dec., inc. and level
    e0 = NormalDistribution(-5, 5)
    e1 = NormalDistribution(5, 5)
    e2 = NormalDistribution(0, 5)

    # Hidden states, with their emission distributions
    h0 = State(e0, name='decreasing')  # State 0
    h1 = State(e1, name='increasing')  # State 1
    h2 = State(e2, name='level')       # State 2
    # There's a 3rd state (start) automatically added
    # There's a 4th state (end) automatically added

    model = HiddenMarkovModel()
    model.add_states(h0, h1, h2)

    # You have to give it some initial values to start the training from. Each possible value of hidden states has a diff. distribution...
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

def create_2Dmodel():
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


def get_best_model(distribution_inertia_vals,max_iterations_vals, train_data, test_data, model_maker=create_model):
    # Train using the EM algorithm.  Pass in the transpose (.T) of the data

    # There are hyperparameters here: distribution_inertia=0.25, max_iterations=10
    # If you leave them at their default values the HMM overfits to so that the
    # 'level' model is a normal with mean 0 and standard dev 0, to catch all the
    # long sequences of 0s

    # Do an equivalent of the GridSearchCV here, but being careful with the cross
    # validation, because this data is one long sequence


    # Init values
    best_prob = -9999999999999  # So it will always be 'beaten' at first
    best_distribution_inertia = None
    best_max_iterations = None

    # product gives you all combinations
    for inertia, iterations in product(distribution_inertia_vals, max_iterations_vals):
        model = model_maker()
        model.fit(
            [real_train_data],
            distribution_inertia=inertia,
            max_iterations=iterations
        )
        # probab. of val. data occuring...ie. given the model fitted how likely was this sequence of valid. data
        # likelihood of observing this val. data.
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
    model = model_maker()
    model.fit(
        [train_data],
        distribution_inertia=best_distribution_inertia,
        max_iterations=best_max_iterations
    )

    return model, best_distribution_inertia, best_max_iterations, test_data



def date_count_plot(dates,counts,formatter = '%d-%m',y_label = "Count" ,x_label="Bin Dates",title="", int_only=False):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    #ax.set_facecolor('xkcd:silver')
    plt.plot(dates, counts,color="blue")
    if formatter:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(formatter))

    if int_only:
        yint = range(min(counts), math.ceil(max(counts)) + 1)
        plt.yticks(yint)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.title(title)

    if TWO_D:
        extra_label = "2D"
    else:
        extra_label = "1D"
    plt.savefig(path + title + extra_label + ".pdf")
    plt.show()





def get_histories_and_labels(x,history_length):
    histories = x[0:history_length]
    for i in range(1,len(x) - history_length):
        h = x[i : i + history_length]
        histories = np.row_stack((histories,h))
    labels = x[history_length:]
    return histories, labels

def get_histories_and_labels2D(x,history_length):
    labels = [a[0] for a in x[history_length:]] # Only want the counts
    histories = [x[i: i+ history_length] for i in range(len(x)-history_length)]
    return histories, labels



# Example prediction
def make_prediction(model, data):
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

    probs = model.predict_proba(data)
    hidden_state_probs = probs[None, -1, :]

    # Add zero probs for start and end states.  Transition matrix is 5x5
    hidden_state_probs = np.append(hidden_state_probs, [[0, 0]], axis=1)
    transmat = model.dense_transition_matrix()

    # Multiply hidden state probs by transition matrix to get probs of next
    # hidden states
    next_hidden_state_probs = np.matmul(hidden_state_probs, transmat)
    predicted_hidden_state = np.argmax(next_hidden_state_probs)
    prediction = model.states[predicted_hidden_state].distribution.parameters[0]

    return prediction
#    return predicted_hidden_state, prediction



# Example prediction
def make_prediction2D(model, data):
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

    return rfq_prediction #, nominals_prediction


def save_dict_to_file(dict,file_name):
    f = open(path + file_name + ".txt", "w")
    f.write(str(dict))
    f.close()


for TWO_D in [True, False]:


    if TWO_D:
        data= load_data(path + data_folder + '15min_bins_simulated_c_n.csv')
        data_set = data.values[:,2:4]

        data_set_diffs = data_set[1:,:] - data_set[:-1,:]
        train_data, test_data = split_data(data_set_diffs)
    else:
        data = load_data(path + data_folder + '15min_bins_simulated.csv')
        counts = data['COUNTS'].values
        data_set = counts
        data_set_diffs = data_set[1:]- data_set[:-1]
        train_data, test_data = split_list(data_set_diffs)

    times = pd.to_datetime(data['BIN_STARTS'].values)
    counts = data['COUNTS'].values

    # Split into training and test
    train_counts, test_counts = split_list(counts)
    train_times, test_times = split_list(times)

    real_train_data, val_data = split_list(train_data, proportion=0.7)

    if not TWO_D:
        real_train_data = real_train_data.flatten()  # For 1 dimension HMM we need 1d arrays
        train_data = train_data.flatten()
        val_data = val_data.flatten()
        test_data = test_data.flatten()

    if TWO_D:

        distribution_inertia_vals = [0.01, 0.05, 0.1, 0.25, 0.5]
        max_iterations_vals = [2, 5, 10]  # Doesn't seem to like this being too high -
        # the predictions at the end come out as nan.
        # you may need to inch this up one by one

        model, best_distribution_inertia, best_max_iterations, test_data = get_best_model(distribution_inertia_vals,max_iterations_vals,train_data, test_data, create_2Dmodel)

        test_on = test_data[:300, :]
        time_on = test_times[:300]
        counts_on = test_counts[:300]


    else:
        # Hyperparameters
        distribution_inertia_vals = [0.01, 0.05, 0.1, 0.25, 0.5]
        max_iterations_vals = [2, 5, 10, 20]
        model, best_distribution_inertia, best_max_iterations, test_data = get_best_model(distribution_inertia_vals,max_iterations_vals,train_data, test_data, create_model)
        # Example on test data
        test_on = test_data[:300]
        time_on = test_times[:300]
        counts_on = test_counts[:300]


    print("transition matrix")
    print(model.dense_transition_matrix())

    print('')
    print('Model is:')
    print(model)  # Gets the normal distributions of the emissions



    print(test_on.T)
    viterbi_predictions = model.predict(test_on, algorithm='viterbi')
    print("viterbi pred: {}".format(''.join(map(str, viterbi_predictions))))



    date_count_plot(time_on,viterbi_predictions[1:],'%H:%M',y_label="Regime", int_only=True, title="Regime plot")

    date_count_plot(time_on,counts_on, '%H:%M', title="Count to match regimes")


    if TWO_D:
        prediction_method = make_prediction2D
        histories_method = get_histories_and_labels2D
        extra_label = "2D"
    else:
        prediction_method = make_prediction
        histories_method = get_histories_and_labels
        extra_label = "1D"


    pred_results = {}
    for history_length in [2,4,8,16]:
        diff_pred_test = train_data
        diff_histories, diff_actual = histories_method(diff_pred_test,history_length)
        diff_predictions = [prediction_method(model,sequence) for sequence in diff_histories]
        RMS = math.sqrt(mean_squared_error(diff_actual, diff_predictions))
        R2 = r2_score(diff_actual, diff_predictions)
        pred_results[history_length] = {'RMS':RMS, 'Score':R2}


    save_dict_to_file(pred_results,"HMM-predictions" + extra_label)
    print(pred_results)





