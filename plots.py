# Master Thesis (Théo Stassen, Université de Liège) :
# "Comparison of probabilistic forecasting deep learning models in the context of renewable energy production"
#
# - Different plot functions


import matplotlib.pyplot as plt
from gluonts.dataset.util import to_pandas
import json
import numpy as np
import os


def plot_agg_metric_dict(agg_metrics_loc, agg_metrics_title, metric, config, goal, big=False):
    """
    Plot the selected agg metric dictionnary as an histogram
    :param agg_metrics_loc: location of the agg metrics dictionnary
    :param agg_metrics_title: title designating the agg metrics dictionnary
    :param metric: name of the metric
    :param config: "A or "B", indicating the dataset configuration
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    :param big: if True the plot size must be of bigger size
    """
    with open("agg_metrics/" + config + "/" + agg_metrics_loc + "/" + metric + goal + '.txt') as json_file:
        current_dict = json.load(json_file)
    if big:
        plt.figure(figsize=(14.5, 7.2))
    else:
        plt.figure(figsize=(4.75, 3.6))
    plt.bar(list(current_dict.keys()), current_dict.values(), color='g', )
    # plt.title(agg_metrics_title + " " + metric + " comparison")
    plt.ylabel(metric)
    plt.xlabel(agg_metrics_title)
    os.makedirs(os.path.dirname("plots/hist/" + config + "/" + agg_metrics_loc + "/"), exist_ok=True)
    plt.savefig("plots/hist/" + config + "/" + agg_metrics_loc + "/" + metric + goal + ".png", bbox_inches='tight')
    plt.show()


def plot_agg_metric_scatter(agg_metrics_loc, agg_metrics_title, metric1, metric2, config, goal, big=False):
    """
    Plot the selected agg metrics dictionaries as a scatter diagram
    :param agg_metrics_loc: location of the agg metrics dictionnary
    :param agg_metrics_title: title designating the agg metrics dictionnary
    :param metric1: name of the first metric
    :param metric2: name of the second metric
    :param config: "A or "B", indicating the dataset configuration
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    :param big: if True the plot size must be of bigger size
    """
    with open("agg_metrics/" + config + "/" + agg_metrics_loc + "/" + metric1 + goal + '.txt') as json_file:
        current_dict1 = json.load(json_file)
    with open("agg_metrics/" + config + "/" + agg_metrics_loc + "/" + metric2 + goal + '.txt') as json_file:
        current_dict2 = json.load(json_file)

    y = list(current_dict1.values())
    z = list(current_dict2.values())
    n = list(current_dict1.keys())
    print(y, z, n)

    fig, ax = plt.subplots(figsize=(4.75, 3.6)) if big is False else plt.subplots(figsize=(14.5, 7.2))
    plt.ylabel(metric1)
    plt.xlabel(metric2)
    ax.scatter(z, y)

    for i, txt in enumerate(n):
        ax.annotate(txt, (z[i], y[i]))
    # plt.title(agg_metrics_title + " " + metric1 + " " + metric2 + " comparison")
    os.makedirs(os.path.dirname("plots/scatter/" + config + "/" + agg_metrics_loc + "/"), exist_ok=True)
    plt.savefig("plots/scatter/" + config + "/" + agg_metrics_loc + "/" + metric1 + "_" + metric2 + goal + ".png",
                bbox_inches='tight')
    plt.show()


def plot_bandwidth_dict(agg_metrics_loc, agg_metrics_title, config):
    """
    Plot the bandwith metric dictionnary as a an histogram -> Not used in current version
    :param agg_metrics_loc: location of the agg metrics dictionnary
    :param agg_metrics_title: title designating the agg metrics dictionnary
    :param config: "A or "B", indicating the dataset configuration
    """
    with open("agg_metrics/" + config + "/" + agg_metrics_loc + "/" + "bandwidth" + ".txt") as json_file:
        current_dict = json.load(json_file)
    # plt.figure(figsize=(7,4.8))
    plt.bar(list(current_dict.keys()), current_dict.values())

    plt.title(agg_metrics_title + " bandwidth" + " comparison")
    os.makedirs(os.path.dirname("plots/hist/" + config + "/" + agg_metrics_loc + "/"), exist_ok=True)
    plt.savefig("plots/hist/" + config + "/" + agg_metrics_loc + "/bandwidth.png")
    plt.show()


def plot_item_metrics(models, item_metrics_title, item_metrics_loc, config):
    """
    Plot the selected item metric vector into frequency histogram -> Not used in current version
    :param models: vector of selected models
    :param item_metrics_title: title designating the item metrics file
    :param item_metrics_loc: location of the item metrics file
    :param config: "A or "B", indicating the dataset configuration
    """
    for model in models:
        item_metric = np.load("item_metrics/" + config + "/" + item_metrics_loc + '/' + model + '.npy')
        plt.hist(item_metric * 1000, bins=range(0, 100, 5), rwidth=0.8, label=model,
                 alpha=0.5)
    plt.title(item_metrics_title)
    plt.legend(loc='upper right')
    plt.show()


def plot_forecast_entry(ts_entry, forecast_entry, plot_length,
                        prediction_interval, plot_name, plot_loc, is_show, config, goal):
    """
    Plot first entry of the dataset and the corresponding forecast
    :param ts_entry: entry of the complete time series
    :param forecast_entry: entry of the probabilistic forecast
    :param plot_length: lenght of the plot
    :param prediction_interval: quantile interval that must be plot
    :param plot_name: Name of the plot
    :param plot_loc: Location of the plot
    :param is_show: if True, the plot is showed to the user
    :param config: "A or "B", indicating the dataset configuration
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    legend = ["observations",
              "median prediction"] + [f"{k}% prediction interval" for k in prediction_interval][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.25))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_interval, color='g')

    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.ylabel("Production (kWh)")
    plt.xlabel("Time (h)")
    # plt.title("Forecast : " + plot_name)
    os.makedirs(os.path.dirname("plots/forecast/" + config + "/" + plot_loc + "/" + goal + "/"), exist_ok=True)
    plt.savefig("plots/forecast/" + config + "/" + plot_loc + "/" + goal + "/" + str(plot_length) + ".png",
                bbox_inches='tight')
    if is_show:
        plt.show()
    plt.close()


def plot_distr_params(models, alphas, distributions, config):
    """
    Plot the selected distribution parameters as frequency histogram -> Not used in current version
    :param models: vector of selected models
    :param alphas: vector of selected alphas
    :param distributions: vector of selected distributions
    :param config: "A or "B", indicating the dataset configuration
    """
    for distribution in distributions:
        if distribution == "Gaussian":
            params = ["mu", "sigma"]
        elif distribution == "Laplace":
            params = ["mu", "b"]
        elif distribution == "PiecewiseLinear":
            params = ["gamma", "b", "knot_positions"]  # b and knot_positions gives only the first value
        elif distribution == "Uniform":
            params = ["low", "high"]
        elif distribution == "Student":
            params = ["mu", "sigma", "nu"]
        else:
            params = []

        i = 0
        for param in params:
            for alpha in alphas:
                for model in models:
                    if model[0] == "c":
                        distr_params = np.load(
                            "distribution_output/" + model + "_" + distribution + "_" + str(alpha) + ".npy")
                        plt.hist(distr_params[i], bins=range(0, 10, 1), rwidth=0.8, label=model + "_" + str(alpha),
                                 alpha=0.5)
            plt.title(param + " of obtained distribution (frequency of values along the time axis) ")
            plt.legend(loc='upper right')
            plt.show()
            i += 1
