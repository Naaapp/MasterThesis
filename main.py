# OSError: libcudart.so.9.2: cannot open shared object file: No such file or directory :
# sudo ldconfig /usr/local/cuda/lib64
import dataset as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math

# Import dataset
from comparison import compare_all_models, compare_simple, compare_simple_feed_forward, compare_canonicalrnn

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
                     learning_length=context_length)

chosen_metric = "Coverage"
quantiles = list([0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995])

distributions = ["Gaussian"]
models = ["cSimple", "SimpleFeedForward", "CanonicalRNN", "DeepAr", "DeepFactor", "GaussianProcess", "NPTS", "MQCNN",
          "MQRNN", "R", "SeasonalNaive"]
alphas = [0.9]
epochs = 10
compare_all_models(dataset, distributions, alphas, models, chosen_metric, epochs, True)

distribution = "Gaussian"
alpha = 0.9
num_cells = [10, 50, 100, 200, 300]
# compare_simple(dataset, distribution, alpha, chosen_metric, 10, num_cells)
# num_hidden_dimensions = [[10], [40], [40, 40], [40, 40, 40]]
# compare_simple_feed_forward(dataset, distribution, alpha, chosen_metric, 1, num_hidden_dimensions)
# num_layers = [1, 2, 5, 10]
# compare_canonicalrnn(dataset, distribution, alpha, chosen_metric, 10,'n_layers', num_layers,)
# hist_plot_item_metrics(chosen_metric, models)  # Not precise for Coverage (too much around)
