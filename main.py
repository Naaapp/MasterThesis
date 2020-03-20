# OSError: libcudart.so.9.2: cannot open shared object file: No such file or directory :
# sudo ldconfig /usr/local/cuda/lib64

from gluonts.dataset.common import ListDataset
from gluonts.distribution import GaussianOutput

import dataset as dt
from plots import plot_train_test_dataset_first, plot_prob_forecasts, \
    plot_agg_metric_dict, hist_plot_item_metrics
from forecast import forecast_dataset
from plots import add_agg_metric_to_dict, save_item_metrics, add_bandwidth_to_dict, plot_bandwidth_dict, \
    plot_distr_params, save_distr_quantiles
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
df = pd.read_csv("datasets/6months-minutes.csv")
imported_dataset = np.array([df['Active Power'].to_numpy()])
n_sample = 183
imported_dataset = imported_dataset.reshape(n_sample, -1)  # Split by days
imported_dataset = imported_dataset / 1000

prediction_length = 10
context_length = 60 * 1  # One day
freq = "1min"
start = pd.Timestamp("01-04-2019", freq=freq)
dataset = dt.Dataset(custom_dataset=imported_dataset, start=start, freq=freq,
                     prediction_length=prediction_length,
                     learning_length=context_length, cardinality=list([1]))

chosen_metric = "Coverage"
quantiles = list([0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995])

# models = ["cSimple", "SimpleFeedForward", "CanonicalRNN", "DeepAr", "DeepFactor", "GaussianProcess",
#           "NPTS", "MQCNN", "MQRNN", "R", "SeasonalNaive"]
# distributions = ["Gaussian", "Laplace", "PiecewiseLinear", ]
# No quantiles in Student, Uniform has a problem with loss

distributions = ["Gaussian"]
models = ["cSimpleFeedForward"]
alphas = [0.7, 0.8, 0.9]

for distribution in distributions:
    for alpha in alphas:
        for model in models:
            epochs = 10
            forecasts, tss = forecast_dataset(dataset, model=model, distrib=distribution,epochs=epochs, alpha=alpha,
                                              quantiles=quantiles)
            add_agg_metric_to_dict(dataset, forecasts, tss, model, alpha, chosen_metric)
            add_bandwidth_to_dict(forecasts, model, alpha)
            plot_prob_forecasts(tss[0], forecasts[0], 60, [50, 90, 99], model, alpha, epochs, distribution)
            plot_prob_forecasts(tss[0], forecasts[0], 60 * 3, [50, 90, 99], model, alpha, epochs, distribution)
            save_item_metrics(dataset, forecasts, tss, model, chosen_metric)
            if model in ["MQCNN", "MQRNN"]:
                save_distr_quantiles(model, forecasts[0], quantiles)

plot_agg_metric_dict(chosen_metric)
plot_bandwidth_dict()

if not [model in ["MQCNN", "MQRNN"] for model in models]:
    plot_distr_params(models, alphas, distributions)

# hist_plot_item_metrics(chosen_metric, models)  # Not precise for Coverage (too much around)



