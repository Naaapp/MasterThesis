import matplotlib.pyplot as plt
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
import json
import numpy as np
import os


def plot_agg_metric_dict(agg_metrics_loc, agg_metrics_title, metric, config):
    with open("agg_metrics/" + config + "/" + agg_metrics_loc + "/" + metric + '.txt') as json_file:
        current_dict = json.load(json_file)
    # plt.figure(figsize=(7, 4.8))
    plt.bar(list(current_dict.keys()), current_dict.values(), color='g')
    plt.title(agg_metrics_title + " " + metric + " comparison")
    os.makedirs(os.path.dirname("plots/hist/" + config + "/" + agg_metrics_loc + "/"), exist_ok=True)
    plt.savefig("plots/hist/" + config + "/" + agg_metrics_loc + "/" + metric + ".png")
    plt.show()


def plot_agg_metric_scatter(agg_metrics_loc, agg_metrics_title, metric1, metric2, config):
    with open("agg_metrics/" + config + "/" + agg_metrics_loc + "/" + metric1 + '.txt') as json_file:
        current_dict1 = json.load(json_file)
    with open("agg_metrics/" + config + "/" + agg_metrics_loc + "/" + metric2 + '.txt') as json_file:
        current_dict2 = json.load(json_file)

    y = list(current_dict1.values())
    z = list(current_dict2.values())
    n = list(current_dict1.keys())

    fig, ax = plt.subplots()
    ax.scatter(z, y)

    for i, txt in enumerate(n):
        ax.annotate(txt, (z[i], y[i]))
    plt.title(agg_metrics_title + " " + metric1 + " " + metric2 + " comparison")
    os.makedirs(os.path.dirname("plots/scatter/" + config + "/" + agg_metrics_loc + "/"), exist_ok=True)
    plt.savefig("plots/scatter/" + config + "/" + agg_metrics_loc + "/" + metric1 + "_" + metric2 + ".png")
    plt.show()


def plot_bandwidth_dict(agg_metrics_loc, agg_metrics_title, config):
    with open("agg_metrics/" + config + "/" + agg_metrics_loc + "/" + "bandwidth" + ".txt") as json_file:
        current_dict = json.load(json_file)
    # plt.figure(figsize=(7,4.8))
    plt.bar(list(current_dict.keys()), current_dict.values())

    plt.title(agg_metrics_title + " bandwidth" + " comparison")
    os.makedirs(os.path.dirname("plots/hist/" + config + "/" + agg_metrics_loc + "/"), exist_ok=True)
    plt.savefig("plots/hist/" + config + "/" + agg_metrics_loc + "/bandwidth.png")
    plt.show()


def plot_item_metrics(models, item_metrics_title, item_metrics_loc, config):
    for model in models:
        item_metric = np.load("item_metrics/" + config + "/" + item_metrics_loc + '/' + model + '.npy')
        plt.hist(item_metric * 1000, bins=range(0, 100, 5), rwidth=0.8, label=model,
                 alpha=0.5)
    plt.title(item_metrics_title)
    plt.legend(loc='upper right')
    plt.show()


def plot_forecast_entry(ts_entry, forecast_entry, plot_length,
                        prediction_interval, plot_name, plot_loc, is_show, config):
    legend = ["observations",
              "median prediction"] + [f"{k}% prediction interval" for k in prediction_interval][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_interval, color='g')

    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.title("Forecast : " + plot_name)
    os.makedirs(os.path.dirname("plots/forecast/" + config + "/" + plot_loc + "/"), exist_ok=True)
    plt.savefig("plots/forecast/" + config + "/" + plot_loc + "/" + str(plot_length) + ".png")
    if is_show:
        plt.show()
    plt.close()


def plot_train_test_dataset(dataset, config):
    entry = next(iter(dataset.train_ds))
    train_series = to_pandas(entry)
    train_series.plot()
    plt.grid(which="both")
    plt.legend(["train series"], loc="upper left")
    plt.show()

    entry = next(iter(dataset.test_ds))
    test_series = to_pandas(entry)
    test_series.plot()
    plt.axvline(train_series.index[-1], color='r')  # end of train dataset
    plt.grid(which="both")
    plt.legend(["test series", "end of train series"], loc="upper left")
    plt.show()

    print(f"Length of forecasting window in test dataset: "
          f"{len(test_series) - len(train_series)}")
    print(f"Learning length: "
          f"{dataset.learning_length}")
    print(f"Recommended prediction horizon: "
          f"{dataset.prediction_length}")
    print(f"Frequency of the time series: {dataset.freq}")


def plot_distr_params(models, alphas, distributions, config):
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
