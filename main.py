from gluonts.dataset.common import ListDataset

import dataset as dt
from plots import plot_train_test_dataset_first, plot_prob_forecasts, \
    plot_agg_metric_dict, hist_plot_item_metrics
from forecast import forecast_dataset
from plots import add_agg_metric_to_dict, save_item_metrics
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
from gluonts.dataset.repository.datasets import get_dataset
import mxnet as mx

from mxnet import nd, gpu, gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import time

# Import dataset
# df = pd.read_csv("datasets/6months-minutes.csv")
# imported_dataset = np.array([df['Active Power'].to_numpy()])
# prediction_length = 10
# context_length = 60 * 1  # One day
# freq = "1min"
# start = pd.Timestamp("01-04-2019", freq=freq)
# dataset = dt.Dataset(custom_dataset=imported_dataset, start=start, freq=freq,
#                      prediction_length=prediction_length,
#                      learning_length=context_length)

# Import predefined dataset
pre_dataset = get_dataset("m4_hourly", regenerate=False)
dataset = dt.Dataset(pre_dataset)

chosen_metric = "QuantileLoss[0.5]"


models = ["DeepAr"]
for model in models:
    forecasts, tss = forecast_dataset(dataset, model=model,
                                      epochs=5)
    add_agg_metric_to_dict(dataset, forecasts, tss, model, chosen_metric)
    plot_prob_forecasts(tss[0], forecasts[0], 60, [50, 90, 99])
    save_item_metrics(dataset, forecasts, tss, model, chosen_metric)

models_to_show = models
plot_agg_metric_dict(chosen_metric)
