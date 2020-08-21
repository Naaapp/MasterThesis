# Master Thesis (Théo Stassen, Université de Liège) :
# "Comparison of probabilistic forecasting deep learning models in the context of renewable energy production"
#
# - Different save functions

import json
import numpy as np
from gluonts.evaluation import Evaluator
import os


def save_agg_metric_to_dict(dataset, forecasts, tss, metric, agg_metric_loc, agg_metric_id, config, number, goal):
    """
    Evaluate the forecasts, obtain a metric value and add it to corresponding metric dictionary
    :param dataset: Dataset object containing training and testing sets and metadata
    :param forecasts: vector of probabilistic forecasts for each time series
    :param tss: vector of each time series
    :param metric: metric type to save
    :param agg_metric_loc: location of the agg metric dictionary
    :param agg_metric_id: identification of the metric we will add to the dictionary
    :param config: "A or "B", indicating the dataset configuration
    :param number: indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    try:
        with open("agg_metrics/" + config + "/" + agg_metric_loc + "/" + metric + goal + '.txt') as json_file:
            current_dict = json.load(json_file)
    except FileNotFoundError:
        os.makedirs(os.path.dirname("agg_metrics/" + config + "/" + agg_metric_loc + "/" + metric + goal + '.txt'), exist_ok=True)
        current_dict = dict()

    evaluator = Evaluator(quantiles=[0.005, 0.1, 0.5, 0.9, 0.995])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts),
                                          num_series=len(dataset.test_ds))
    print(json.dumps(agg_metrics, indent=4))
    if metric == "Coverage":
        # low_coverage = agg_metrics["Coverage[0.005]"]
        high_coverage = agg_metrics["Coverage[0.995]"]
        # low_score = 0.005 - low_coverage
        high_score = high_coverage - 0.995
        agg_metric = high_score  # + low_score
    else:
        agg_metric = agg_metrics[metric]

    if number == 1:
        current_dict[agg_metric_id] = agg_metric
    else:
        current_dict[agg_metric_id] = current_dict[agg_metric_id]*(1-1/number) + agg_metric * (1/number)

    with open("agg_metrics/" + config + "/" + agg_metric_loc + "/" + metric + goal + '.txt', 'w') as outfile:
        json.dump(current_dict, outfile)


def save_bandwidth_to_dict(forecasts, metric, agg_metric_loc, agg_metric_id, config):
    """
    Evaluate the forecasts, obtain a bandwidth metric value and add it to corresponding metric dictionary
    :param forecasts: vector of probabilistic forecasts for each time series
    :param metric: metric type to save
    :param agg_metric_loc: location of the agg metric dictionary
    :param agg_metric_id: identification of the metric we will add to the dictionary
    :param config: "A or "B", indicating the dataset configuration
    """
    try:
        with open("agg_metrics/" + config + "/" + agg_metric_loc + "/bandwidth.txt") as json_file:
            current_dict = json.load(json_file)
    except FileNotFoundError:
        os.makedirs(os.path.dirname("agg_metrics/" + config + "/" + agg_metric_loc + "/bandwidth.txt"),
                    exist_ok=True)

        current_dict = dict()
    bandwidth = np.mean([forecast.quantile(0.995) for forecast in forecasts])
    bandwidth -= np.mean([forecast.quantile(0.005) for forecast in forecasts])
    current_dict[agg_metric_id] = float(bandwidth)
    with open("agg_metrics/" + config + "/" + agg_metric_loc + "/bandwidth.txt", 'w') as outfile:
        json.dump(current_dict, outfile)


def save_item_metrics(dataset, forecasts, tss, metric, item_metric_loc, item_metric_id, config):
    """
    Evaluate the forecasts, obtain a item metric vector of values and add it to corresponding item metric file
    :param dataset: Dataset object containing training and testing sets and metadata
    :param forecasts: vector of probabilistic forecasts for each time series
    :param tss: vector of each time series
    :param metric:  metric type to save
    :param item_metric_loc: location of the agg metric dictionary
    :param item_metric_id: identification of the metric we will add to the dictionary
    :param config: "A or "B", indicating the dataset configuration
    """
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
    os.makedirs(os.path.dirname("item_metrics/" + config + "/" + item_metric_loc + "/" + item_metric_id + "/"), exist_ok=True)
    np.save("item_metrics/" + config + "/" + item_metric_loc + "/" + item_metric_id + "/" + metric + '.npy', item_metric)


def save_distr_quantiles(model, forecast_entry, quantiles, config):
    """

    :param model: selected model
    :param forecast_entry: entry of the probabilistic forecast
    :param quantiles: vector of quantiles to save
    :param config: "A or "B", indicating the dataset configuration
    """
    for quantile_value in quantiles:
        os.makedirs(os.path.dirname("distribution_output/" + config + "/" + model + "/"), exist_ok=True)
        np.save("distribution_output/" + config + "/" + model + "/" + str(quantile_value).replace(".", "_"),
                forecast_entry.quantile(quantile_value))
