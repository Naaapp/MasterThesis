import json
import numpy as np
from gluonts.evaluation import Evaluator
import os


def save_agg_metric_to_dict(dataset, forecasts, tss, metric, agg_metric_loc, agg_metric_id, config):
    try:
        with open("agg_metrics/" + config + "/" + agg_metric_loc + "/" + metric + '.txt') as json_file:
            current_dict = json.load(json_file)
    except FileNotFoundError:
        os.makedirs(os.path.dirname("agg_metrics/" + config + "/" + agg_metric_loc + "/" + metric + '.txt'), exist_ok=True)
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

    current_dict[agg_metric_id] = agg_metric

    with open("agg_metrics/" + config + "/" + agg_metric_loc + "/" + metric + '.txt', 'w') as outfile:
        json.dump(current_dict, outfile)


def save_bandwidth_to_dict(forecasts, metric, agg_metric_loc, agg_metric_id, config):
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
    for quantile_value in quantiles:
        os.makedirs(os.path.dirname("distribution_output/" + config + "/" + model + "/"), exist_ok=True)
        np.save("distribution_output/" + config + "/" + model + "/" + str(quantile_value).replace(".", "_"),
                forecast_entry.quantile(quantile_value))
