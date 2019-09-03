import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import scipy.stats as stats
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge


# load data from csv files
def load_data(filepath):
    return pd.read_csv(filepath)

# split data into train and test components. 90%-10% split
def split_data(data_set):
    n = int(0.9 * len(data_set))

    train_set = data_set[:n]
    train_array = np.array(train_set)
    X_train = train_array[:, :-1]
    y_train = train_array[:, -1]

    training_dataset = (X_train, y_train)

    test_set = data_set[n:]
    test_array = np.array(test_set)
    X_test = test_array[:, :-1]
    y_test = test_array[:, -1]
    test_dataset = (X_test, y_test)

    return training_dataset, test_dataset


def train(model, hyp, train_data):
    clf = GridSearchCV(model, hyp, cv=5)
    (X, y) = train_data
    clf.fit(X, y)  # loss in this thing...

    best_params = clf.best_params_
    # best model has best param and fitted to train data
    best_model = clf.best_estimator_

    #cv_score = clf.cv_results_['mean_test_score']
    debug = 0
    return best_model, best_params #, cv_score


# uses best model on the 10% test data set aside
def test_score(model, test_data):
    (X, y) = test_data
    return model.score(X, y)


# I use this to plot score of a single chosen parameter
def single_param_plot(model, hyp, train_data, param, data_file, bar_chart=False):
    model_name = type(model).__name__
    (X, y) = train_data
    param_range = hyp[param]

    train_scores, test_scores = validation_curve(model, X, y, param, param_range, cv=5,
                                                 scoring="neg_mean_squared_error")


    train_rms = np.sqrt(-train_scores)
    test_rms = np.sqrt(-test_scores)

    if bar_chart:
        validation_bar_plot(train_rms,test_rms,param,param_range,data_file,model_name)
    else:
        validation_plot(train_rms, test_rms, param, param_range, data_file, model_name)


def save_plot(plot_name, show_plot=True):
    plt.savefig(path + plots_folder + plot_name + ".pdf")
    if show_plot:
        plt.show()


def validation_bar_plot(train_scores, test_scores, param, param_range, data_file, model_name):
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    ylim_min = min(train_scores_mean.min(), test_scores_mean.min()) - 0.5
    ylim_max = max(train_scores_mean.max(), test_scores_mean.max()) + 0.2
    #ylim_max = max(train_scores.max(), test_scores.max())


    labels = [str(a) for a in param_range]
    num_points = range(len(labels))
    x = np.arange(len(num_points))
    width = 0.3

    fig, ax = plt.subplots()

    ax.bar(x - width / 2, train_scores_mean, width, label="Training error")
    ax.bar(x + width /2, test_scores_mean, width, label="Testing error")

    plt.xlabel(param)
    plt.ylim(ylim_min,ylim_max)
    ax.set_ylabel("Root Mean Squared Error")
    title = param + model_name + data_file
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.legend(loc="best") #loc="lower right", fancybox=True, framealpha=0.5)
    save_plot(title)



def validation_plot(train_scores, test_scores, param, param_range, data_file, model_name):
    ylim_min = min(train_scores.min(), test_scores.min())
    ylim_max = max(train_scores.max(), test_scores.max())
    ylim_max2 = ylim_max # min(ylim_max,10)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)


    #ax = plt.gca()
    #ax.set_xscale('log')
    #
    # bar_chart = False
    #
    # if bar_chart:
    #     num_points = range(len(param_range))
    #     x_labels = [str(a) for a in param_range]
    #     plt.xticks(num_points, x_labels)
    #     plt.plot(train_scores_mean, label="Training error")
    #     plt.plot(test_scores_mean, label="Testing error", dashes=(2, 1))
    # else:
    xrange = param_range
    #xrange = np.log(param_range)  # logarithmic plot
    plt.xscale('log')
    plt.plot(xrange, train_scores_mean, label="Training error")
    plt.plot(xrange, test_scores_mean, label="Testing error", dashes=(2, 1))


    #plt.title(f"Effect of parameter {param} on {model_name} on file {data_file}")
    plt.xlabel(param)
    plt.ylabel("Root Mean Squared Error")
    plt.ylim(ylim_min, ylim_max2)

    plt.legend(loc="best")
    plot_name = model_name + param + data_file
    save_plot(plot_name)


# model with triple: chosen model, parameters with values, the parameter to perform single plot on
models = [
    (MLPRegressor(random_state=42), #, solver='sgd', learning_rate="invscaling"), #,solver='sgd'),
    {
        'learning_rate_init': np.logspace(-7, 0, 7),# Todo: CHANGE LAST VALUE BACK TO 7 7), # LEARNING RATE WITH SGD BEST VALUE AROUND 2 * 10 -5  BAD FROM 10^-4 on. -8 to -3.5 gives an OK plot with sgd
        # LEARNING RATE WITHOUT SGD -7 to 0 GIVES A GOOD PLOT
        'hidden_layer_sizes': ((100),  (400), (100,100), (50,50,50)), # Todo: PUT THESE BACK IN
        #'alpha': np.logspace(-5, 0,5), # Alpha good up until 1 then gets bad. Default was 0.0001

        # HIDDEN LAYERS: One layer seems best in the end, as others overfit the data (better at training worse at testing)
        # More neurons in this one layer the better, but not by much
        # Soo (400) > (100) > more complex things...

        # (100,) 2.3, 2.8
        # (5,3,1) 3.5
        # (10,5,1) 3
        # (400)  2.3 2.7 (slightly better than 100)
        # (5) worse than 100
        # (1000) very slightly better than 400
        # (100,100) better than 100
        # (400, 100) better at train worse at test
        # Roughly more layers are better at training and worse at testing
    },
    ["hidden_layer_sizes", "learning_rate_init"])
   ,
    (Ridge(), {
        'alpha': np.logspace(-10,10,100)
    },
     ["alpha"]),

    (BayesianRidge(), {
         'alpha_1': np.logspace(-10, 10, 5),
         'alpha_2': np.logspace(-10, 10, 5),
         'lambda_1': np.logspace(-10, 10, 10),
         'lambda_2': np.logspace(-10, 10, 5)
     }
      , ["lambda_1"]),
]

path = os.getcwd() + '/'
data_folder = 'datafiles/'




count_data_files = [
    "30min",
    "1hr",
    "2hr",
    "4hr"
]

notional_data_files = [
     "notional_30min",
     "notional_1hr",
     "notional_2hr",
     "notional_4hr"
]

data_files = count_data_files + notional_data_files

# dictionary with each entry containing the model, the filename and the results based on gridsearch

def weight_plot(model, model_name, param_name, data_file, X_train, y_train):
    weights = []
    n_alphas = 200
    param_range = np.logspace(0, 10, n_alphas)

    for a in param_range:
        clf = model(alpha=a)
        clf.fit(X_train, y_train)
        weights.append(clf.coef_)
    ax = plt.gca()

    ax.plot(param_range, weights)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plot_name = model_name + ' coefficients as a function of ' + param_name + ' for file ' + data_file
    #plt.title(plot_name)
    plt.axis('tight')
    save_plot(plot_name)


def multi_weight_plot(files_to_plot):
    for data_file in files_to_plot:
        data_set = load_data(path + data_folder + data_file + ".csv")
        train_data, test_data = split_data(data_set)
        X_train, y_train = train_data
        weight_plot(Ridge, "Ridge", 'alpha', data_file, X_train, y_train)



def model_results():
    consolidated_results = {}
    for (model, hyp, params_to_plot) in models:
        model_name = type(model).__name__
        consolidated_results[model_name] = {}
        for data_file in data_files:
            print('Training model {} on data {}'.format(model_name, data_file))
            results = {}

            data_set = load_data(path + data_folder + data_file + ".csv")
            train_data, test_data = split_data(data_set)

            # GRID SEARCH CV STUFF
            best_model, best_params = train(model, hyp, train_data)
            results['test score'] = test_score(best_model, test_data)
            results['train score'] = test_score(best_model, train_data)


            results['best params'] = best_params
            #results['cv score'] = cv_score
            X_train, y_train = train_data
            X_test, y_test = test_data

            y_predict_train = best_model.predict(X_train)
            y_predict_test = best_model.predict(X_test)

            results['train error'] = math.sqrt(mean_squared_error(y_train, y_predict_train))
            results['test error'] = math.sqrt(mean_squared_error(y_test, y_predict_test))

            consolidated_results[model_name][data_file] = results

            for param_to_plot in params_to_plot:
                # Uses the param_to_plot levels but leaves the rest as default values. Below doesnt use gridsearch
                if param_to_plot == "hidden_layer_sizes":
                    bar_chart = True
                else:
                    bar_chart = False

                single_param_plot(model, hyp, train_data, param_to_plot, data_file, bar_chart)

    return consolidated_results

def save_dict_to_file(dict,file_name):
    f = open(path + table_folder + file_name + ".txt", "w")
    f.write(str(dict))
    f.close()


def save_results(results, file_name = "consolidated_results"):
    # Firstly, save the whole dictionary of results as one file
    save_dict_to_file(results,file_name)


    # Iterate over models
    model_names = results.keys()

    for model_name in model_names:
        model_results = results[model_name]


        files = model_results.keys()

        # Make a table of hyperparameters checked
        m_hyp = {}
        for file in files:
            m_hyp[file] = model_results[file]['best params']

        hyp_title = model_name + " Parameters"
        save_dict_to_file(m_hyp, hyp_title)

        # ( Should make a table of default values for all hyperparameters too)


    # Make a table of ERRORS and scores for each model and each file (when trained with best parameters for that file)
    main_table = [['Model name','File','Train RMS', 'Test RMS', 'Train Score', 'Test Score']]

    for model_name in model_names:
        for file_name in data_files:
            row_results = results[model_name][file_name]
            row_data = [row_results['train error'], row_results['test error'], row_results['train score'], row_results['test score']]
            row_data_strings = ['{0:.2f}'.format(x) for x in row_data]
            row = [model_name,file_name] + row_data_strings
            main_table.append(row)

    # Save main table to file
    with open(path + table_folder + "main_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(main_table)



def train_size_plot(data_files_to_plot, num_sizes=40):
    train_sizes = list(np.arange(0, 1, 1 / num_sizes))[1:]
    model_names = []

    for data_file in data_files_to_plot:

        test_results = {}
        for (model, hyp, param_to_plot) in models:
            model_name = type(model).__name__
            model_names.append(model_name)

            test_results[model_name] = []

            data_set = load_data(path + data_folder + data_file + ".csv")
            train_data, test_data = split_data(data_set)
            X_train, y_train = train_data
            X_test, y_test = test_data

            for train_size in train_sizes:
                num_samples = int(len(X_train) * train_size)
                reduced_train_data = X_train[0:num_samples]
                y_train_reduced = y_train[0:num_samples]
                model.fit(reduced_train_data, y_train_reduced)

                y_predict_test = model.predict(X_test)

                test_results[model_name].append(math.sqrt(mean_squared_error(y_test, y_predict_test)))

            plt.plot(train_sizes, test_results[model_name], label=model_name)
        plt.legend(model_names)
        plt.xlabel("Fraction of training data")
        plt.ylabel("RMS Error")
        plot_name = f"Test error against training data size in file {data_file}"
        #plt.title(plot_name)
        save_plot(plot_name)


def results_plot(c_results, data_files, keys_to_plot, title, y_label):
    num_points = range(len(data_files))
    plt.xticks(num_points, data_files)

    model_names = c_results.keys()

    line_styles = ["-", "-", ":", "--", ":", "-", "--", "--", "-", "--", "-", "--", "-", "--", "-", "--"]

    count = 0
    line_names = []

    for model_name in model_names:

        model_results = c_results[model_name]

        for k in keys_to_plot:
            x = [model_results[data_file][k] for data_file in data_files]
            plt.plot(x, linestyle=line_styles[count])
            line_names.append(model_name + " " + k)
            count = count + 1

    plt.legend(line_names)
    #plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Data file")
    save_plot(title)


def results_bar_plot(c_results, data_files, key_to_plot, title, y_label):
    labels = data_files
    num_points = range(len(data_files))
    model_names = c_results.keys()

    x = np.arange(len(num_points))
    width = 0.3

    fig, ax = plt.subplots()
    num_models = len(model_names)

    plt.xticks(num_points, data_files)

    ##line_styles = ["-", "-", ":", "--", ":", "-", "--", "--", "-", "--", "-", "--", "-", "--", "-", "--"]#

    #count = 0
    #line_names = []
    model_count = -1 #ASSUMES 3 MODELS ARE USED, will then be [-1, 0, 1]
    num_models = len(model_names)


    for model_name in model_names:

        model_results = c_results[model_name]

        results_to_plot = [model_results[data_file][key_to_plot] for data_file in data_files]
        rects = ax.bar(x + model_count * width, results_to_plot, width, label=model_name)

        model_count = model_count + 1

    plt.xlabel("Data file")
    ax.set_ylabel(y_label)
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    save_plot(title)

    # See matplotlib Grouped bar chart with labels for reference

#    plt.legend(line_names)
#    plt.title(title)
#    plt.ylabel(y_label)
#    plt.xlabel("Data file")
#    save_plot(title)


def gaussian_plot(mean, variance, title="Outcome probabilities from Bayesian Ridge"):
    sigma = math.sqrt(variance)
    x = np.linspace(mean - 3 * sigma, mean + 3 * sigma, 100)
    y = stats.norm.pdf(x,mean,sigma)
    print(y)
    plt.plot(x, y)
    plt.xlabel("Outcome")
    plt.ylabel("Probability Density")
    #plt.title(title)
    save_plot(title)
    print("Mean: ", mean)
    print("Variance: ", variance)


# Plots Prediction distribution for first test sample in given file, using a trained Bayesian Ridge.
def B_ridge_predict_plot(data_file):

    b_model = False
    # Get the Bayesian Ridge model.
    for (model, hyp, param_to_plot) in models:
        model_name = type(model).__name__
        if model_name == "BayesianRidge":
            b_model = model
            b_hyp = hyp

    if b_model:
        data_set = load_data(path + data_folder + data_file + ".csv")
        train_data, test_data = split_data(data_set)
        X_test, y_test = test_data

        best_model, best_params = train(b_model, b_hyp, train_data)

        mean = best_model.predict(X_test)[0]
        var = best_model.alpha_
        gaussian_plot(mean, var)



#================ PRODUCE THE PLOTS AND TABLES FOR ALL UNSUPERVISED MODELS ================
plots_folder = 'plots_final2/'
table_folder = 'tables_final2/'

# Make table and plots folders if they don't exist yet.
for folder in [path + plots_folder, path + table_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Gathers all data on best parameters and predictions for each model
# and prints graph of how varying the chosen parameter affects the errors for each model and file
results = model_results()

#def supervised_models():

# Save and print out results to files
save_results(results)

# Plot graphs from various aspects of the main results
err_test_title = "RMS Test errors for each model and count file"
err_train_title = "RMS Train errors for each model and count file"
err_test_n_title = "RMS Test errors for each model and notional file"
err_train_n_title = "RMS Train errors for each model and notional file"



#err_title = "RMS Errors for each model and file"
err_label = "RMS Error"

# Testing out the bar charts.
results_bar_plot(results,count_data_files,'test error', err_test_title, err_label)
results_bar_plot(results,count_data_files, 'train error', err_train_title, err_label)
results_bar_plot(results,notional_data_files,'test error', err_test_n_title,err_label)
results_bar_plot(results,notional_data_files,'train error', err_train_n_title,err_label)

#results_plot(results,data_files,['test error', 'train error'],err_title,err_label)
#results_plot(results,count_data_files,['test error'], err_title,err_label)

test_score_title = "Test scores for each model and count file"
train_score_title = "Train scores for each model and count file"
test_score_n_title = "Test scores for each model and notional file"
train_score_n_title = "Train scores for each model and notional file"


score_label = "score"

results_bar_plot(results,count_data_files,'test score', test_score_title,score_label)
results_bar_plot(results,count_data_files,'train score', train_score_title,score_label)
results_bar_plot(results,notional_data_files,'test score', test_score_n_title,score_label)
results_bar_plot(results,notional_data_files,'train score', train_score_n_title,score_label)


#results_plot(results,data_files,['test score', 'train score'],score_title,score_label)


# Plots how RMS Errors are affected by training sizes for each model, with the default parameters used for each
train_size_plot(data_files)

# Plots a graph of weights against coefficient sizes for the Ridge model for each file
# (Choose one of these to go in the report)
multi_weight_plot(["4hr"])

# Plots a simple Gaussian representing one distribution of predictions produced by Bayesian Ridge.
# Plots the prediction for the first test row from the given file, using the trained Bayesian Ridge model.
B_ridge_predict_plot("4hr")


