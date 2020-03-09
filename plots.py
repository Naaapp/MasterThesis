import matplotlib.pyplot as plt
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
import json
import numpy as np


def plot_prob_forecasts(ts_entry, forecast_entry, plot_length,
                        prediction_interval, model, epochs):
    legend = ["observations",
              "median prediction"] + [f"{k}% prediction interval"
                                      for k in
                                      prediction_interval][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_interval, color='g')

    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.savefig("plots/forecast_"+str(plot_length)+"_"+model+"_"+str(epochs)+".png")
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
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts),
                                          num_series=len(dataset.test_ds))
    print(agg_metrics)
    item_metric = item_metrics[[metric]].to_numpy()

    np.save("item_metrics/" + metric + '_' + model + '.npy', item_metric)


def hist_plot_item_metrics(metric, models):
    for model in models:
        item_metric = np.load("item_metrics/" + metric + '_' + model + '.npy')
        print(item_metric)
        plt.hist(item_metric, bins=range(0, 100, 2), rwidth=0.8, label=model,
                 alpha=0.5)
    plt.title(metric)
    plt.legend(loc='upper right')
    plt.show()


def add_agg_metric_to_dict(dataset, forecasts, tss, model,metric):
    try:
        with open("agg_metrics/" + metric + '.txt') as json_file:
            current_dict = json.load(json_file)
    except FileNotFoundError:
        current_dict = dict()

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts),
                                          num_series=len(dataset.test_ds))
    agg_metric = agg_metrics[metric]
    current_dict[model] = agg_metric

    with open("agg_metrics/" + metric + '.txt', 'w') as outfile:
        json.dump(current_dict, outfile)


def plot_agg_metric_dict(metric):
    with open("agg_metrics/" + metric + '.txt') as json_file:
        current_dict = json.load(json_file)
    plt.bar(list(current_dict.keys()), current_dict.values(), color='g')
    plt.title(metric + " comparison")
    plt.show()
