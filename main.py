# OSError: libcudart.so.9.2: cannot open shared object file: No such file or directory :
# sudo ldconfig /usr/local/cuda/lib64

from comparison import compare_all_models, compare_mqrnn, compare_mqcnn, compare_simple, compare_simple_feed_forward, \
    compare_canonicalrnn, compare_deepfactor, compare_alpha, compare_distrib, compare_model, compare_dataset_size, \
    compare_use_static
from imports import import_dataset
from plots import plot_distr_params, plot_agg_metric_scatter, plot_train_test_dataset

dataset = import_dataset(["6months-minutes", "2eol_measurements", "p_gestamp"],
                         ["6months-minutes", "2eol_measurements", "p_gestamp"])
config = "a"
# dataset = import_dataset(["6months-minutes"],["2eol_measurements"])
# config = "b"

# Config 1 : 6months + 2eol + gestamp | 6months + 2eol + gestamp
# Config 2 : 6months | 2eol

plot_train_test_dataset(dataset, config)

chosen_metric = "Coverage"
quantiles = list([0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995])

# Custom models with custom loss : cSimple, cSimpleFeedForward, cCanonicalRNN, cDeepAr
distributions = ["Gaussian", "Laplace"]
models = ["cSimple", "cSimpleFeedForward", "cCanonicalRNN", "DeepAr", "DeepFactor", "GaussianProcess",
          "NPTS", "MQCNN", "MQRNN", "R", "SeasonalNaive"]

alphas = [0, 0.9]
alpha = 0
model = "cSimple"
epochs = 10
# compare_all_models(dataset, distributions, alphas, models, chosen_metric, epochs, True, quantiles)
# plot_agg_metric_scatter("Global ", "global", chosen_metric, "bandwidth")

distribution = "Gaussian"
alpha = 0.9
# plot_distr_params(["cSimple", "cSimpleFeedForward","cCanonicalRNN"], [0], distributions)

# mlp_final_dim = [5,10,20,30,40]
# compare_mqcnn(dataset, distribution, alpha, chosen_metric, epochs, mlp_final_dim, quantiles)


num_cells = [10, 50]
compare_simple(dataset, distribution, alpha, chosen_metric, epochs, num_cells, quantiles, config)

# num_hidden_dimensions = [[10], [40], [40, 40], [40, 40, 40]]
# compare_simple_feed_forward(dataset, distribution, alpha, chosen_metric, epochs, num_hidden_dimensions, quantiles)
#
# num_layers = [1, 2, 5, 10]
# compare_canonicalrnn(dataset, distribution, alpha, chosen_metric, epochs, 'n_layers', num_layers, quantiles)

# num_hidden_global = [20,50,100]
# num_layers_global = [1, 2, 5]
# num_factors = [5, 10, 20]
# compare_deepfactor(dataset,distribution,alpha,chosen_metric,epochs,"n_hidden_global",num_hidden_global, quantiles)
# compare_deepfactor(dataset,distribution,alpha,chosen_metric,epochs,"n_layers_global",num_layers_global, quantiles)
# compare_deepfactor(dataset,distribution,alpha,chosen_metric,epochs,"n_factors",num_factors, quantiles)

# compare_alpha(dataset, distribution, alphas, model, chosen_metric, epochs, True, quantiles)

# compare_distrib(dataset, distributions, alpha, model, chosen_metric, epochs, True, quantiles)

# compare_model(dataset, distribution, alpha, models, chosen_metric, epochs, True, quantiles)

# dataset_sizes = [360, 720, 1440]
# compare_dataset_size(dataset_sizes, distribution, alpha, model, chosen_metric, epochs, True, quantiles)

# compare_use_static(dataset, distribution, alpha, "DeepAr", chosen_metric, epochs, True, quantiles) #not working
