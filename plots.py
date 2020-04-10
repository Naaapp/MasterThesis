import matplotlib.pyplot as plt
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
import json
import numpy as np


def plot_prob_forecasts(ts_entry, forecast_entry, plot_length,
                        prediction_interval, model, plot_name, is_show):
    legend = ["observations",
              "median prediction"] + [f"{k}% prediction interval"
                                      for k in
                                      prediction_interval][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_interval, color='g')

    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.title("Forecast : " + model)
    if is_show:
        plt.show()
    plt.savefig("plots/forecast_" + plot_name + "_" + str(plot_length) + ".png")
    plt.close()


def plot_train_test_dataset_first(dataset):
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


def save_item_metrics(dataset, forecasts, tss, model, metric):
    evaluator = Evaluator(quantiles=[0.005, 0.1, 0.5, 0.9, 0.995], )
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts),
                                          num_series=len(dataset.test_ds))
    if metric == "Coverage":
        low_coverage = item_metrics[["Coverage[0.005]"]].to_numpy()
        high_coverage = item_metrics[["Coverage[0.995]"]].to_numpy()
        low_score = 0.005 - low_coverage
        high_score = high_coverage - 0.995
        item_metric = high_score + low_score
    else:
        item_metric = item_metrics[[metric]].to_numpy()

    np.save("item_metrics/" + metric + '_' + model + '.npy', item_metric)


def hist_plot_item_metrics(metric, models):
    for model in models:
        item_metric = np.load("item_metrics/" + metric + '_' + model + '.npy')
        plt.hist(item_metric * 1000, bins=range(-100, 100, 1), rwidth=0.8, label=model,
                 alpha=0.5)
    plt.title(metric)
    plt.legend(loc='upper right')
    plt.show()


def add_agg_metric_to_dict(dataset, forecasts, tss, metric, plot_name, params_name, params_val):
    try:
        with open("agg_metrics/" + plot_name + "_" + params_name + "_" + metric + '.txt') as json_file:
            current_dict = json.load(json_file)
    except FileNotFoundError:
        current_dict = dict()

    evaluator = Evaluator(quantiles=[0.005, 0.1, 0.5, 0.9, 0.995])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts),
                                          num_series=len(dataset.test_ds))
    if metric == "Coverage":
        low_coverage = agg_metrics["Coverage[0.005]"]
        high_coverage = agg_metrics["Coverage[0.995]"]
        low_score = 0.005 - low_coverage
        high_score = high_coverage - 0.995
        agg_metric = high_score + low_score
    else:
        agg_metric = agg_metrics[metric]

    current_dict[params_val] = agg_metric

    with open("agg_metrics/" + plot_name + "_" + params_name + "_" + metric + '.txt', 'w') as outfile:
        json.dump(current_dict, outfile)


def plot_agg_metric_dict(metric, plot_name, params_name):
    with open("agg_metrics/" + plot_name + "_" + params_name + "_" + metric + '.txt') as json_file:
        current_dict = json.load(json_file)
    plt.bar(list(current_dict.keys()), current_dict.values(), color='g')
    plt.title(metric + " " + plot_name + " " + params_name + " comparison")
    plt.show()


def add_bandwidth_to_dict(forecasts, plot_name, params_name, params_val):
    try:
        with open("agg_metrics/" + plot_name + "_" + params_name + "_bandwidth.txt") as json_file:
            current_dict = json.load(json_file)
    except FileNotFoundError:
        current_dict = dict()
    bandwidth = np.mean([forecast.quantile(0.995) for forecast in forecasts])
    bandwidth -= np.mean([forecast.quantile(0.005) for forecast in forecasts])
    current_dict[params_val] = float(bandwidth)
    with open("agg_metrics/" + plot_name + "_" + params_name + "_bandwidth.txt", 'w') as outfile:
        json.dump(current_dict, outfile)


def plot_bandwidth_dict(plot_name, params_name):
    with open("agg_metrics/" + plot_name + "_" + params_name + "_bandwidth.txt") as json_file:
        current_dict = json.load(json_file)
    plt.bar(list(current_dict.keys()), current_dict.values(), color='g')
    plt.title("bandwidth" + " " + plot_name + " " + params_name + " comparison")
    plt.show()


def plot_distr_params(models, alphas, distributions):
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
                        print(params[i], alpha, distr_params[i])
                        plt.hist(distr_params[i], bins=range(0, 10, 1), rwidth=0.8, label=model + "_" + str(alpha),
                                 alpha=0.5)
            plt.title(param + " of obtained distribution (frequency of values along the time axis) ")
            plt.legend(loc='upper right')
            plt.show()
            i += 1


def save_distr_quantiles(model, forecast_entry, quantiles):
    for quantile_value in quantiles:
        np.save("distribution_output/" + model + "_" + str(quantile_value).replace(".", "_"),
                forecast_entry.quantile(quantile_value))
