# Master Thesis (Théo Stassen, Université de Liège) :
# "Comparison of probabilistic forecasting deep learning models in the context of renewable energy production"
#
# - Contains all function that performs comparison of hyperparameters

from forecast import forecast_dataset
from plots import *
from save import *


def add_metrics_plots_and_save(dataset, forecasts, tss, metrics, plot_name, plot_loc, agg_metric_loc, agg_metric_id,
                               is_show, config, number, goal):
    """
    Generic function that save the different metrics and plot the forecasts.
    :param dataset: dataset object containing training and testing time series and metadata
    :param forecasts: forecasts results
    :param tss: testing time series
    :param metrics: vector of metrics selected
    :param plot_name: name that must be put in plots
    :param plot_loc: location where the plot must be saved
    :param agg_metric_loc: location where the metric must be saved
    :param agg_metric_id: identification of the metric
    :param is_show: if True, the plot is showed to the user
    :param config: "A or "B", indicating the dataset configuration
    :param number: indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    for metric in metrics:
        save_agg_metric_to_dict(dataset, forecasts, tss, metric, agg_metric_loc, agg_metric_id, config, number, goal)
        save_item_metrics(dataset, forecasts, tss, metric, agg_metric_loc, agg_metric_id, config)
    save_bandwidth_to_dict(forecasts, plot_name, agg_metric_loc, agg_metric_id, config)
    plot_forecast_entry(tss[2], forecasts[2], 60, [50, 90, 99], plot_name, plot_loc, is_show, config, goal)
    plot_forecast_entry(tss[2], forecasts[2], 60 * 3, [50, 90, 99], plot_name, plot_loc, is_show, config, goal)


def compare_alpha(datasize, distribution, alphas, model, chosen_metrics, epochs, is_show, quantiles, config, num, goal):
    """
    Comparison of different alpha values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution argument of the model
    :param alphas: vector of alpha values argument of the model
    :param model: chosen model to train and evaluate
    :param chosen_metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param is_show: if True, the plot is showed to the user
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    for alpha in alphas:
        for i in range(1, num+1):
            dataset, dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                              distrib=distribution, alpha=alpha, goal=goal, config=config)
            add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metrics,
                                       "Alpha " + model + " " + str(alpha).replace(".", "_"),
                                       "alpha/" + model + "/" + str(alpha).replace(".", "_"),
                                       "alpha/" + model,
                                       str(alpha).replace(".", "_"),
                                       is_show, config, i, goal)
    for metric in chosen_metrics:
        plot_agg_metric_dict("alpha/" + model, "Alpha " + model, metric, config, goal)
    plot_bandwidth_dict("alpha/" + model, "Alpha " + model, config)
    plot_agg_metric_scatter("alpha/" + model, "Alpha " + model, chosen_metrics[0], chosen_metrics[1], config, goal)


def compare_distrib(datasize, distributions, alpha, model, chosen_metrics, epochs, is_show, quantiles, config, num, goal):
    """
    Comparison of different output distributions
    :param datasize: designated size of the dataset time series
    :param distributions: vector of output distributions argument of the model
    :param alpha: alpha values argument of the model
    :param model: chosen model to train and evaluate
    :param chosen_metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param is_show: if True, the plot is showed to the user
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    for distribution in distributions:
        for i in range(1, num+1):
            dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                              distrib=distribution, alpha=alpha, goal=goal, config=config)
            add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metrics,
                                       "Distribution " + model + " " + distribution,
                                       "distribution/" + model,
                                       "distribution/" + model,
                                       distribution,
                                       is_show, config, i, goal)
    for metric in chosen_metrics:
        plot_agg_metric_dict("distribution/" + model + "/", "Distribution " + model
                             + " ", metric, config, goal)
    plot_bandwidth_dict("distribution/" + model + "/", "Distribution " + model
                        + " ", config)
    plot_agg_metric_scatter("distribution/" + model + "/", "Distribution " + model
                            + " ", chosen_metrics[0], chosen_metrics[1], config, goal)


def compare_model(datasize, distribution, alpha, models, chosen_metrics, epochs, is_show, quantiles, config, num, goal):
    """
    Comparison between different models
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param models chosen model to train and evaluate:
    :param chosen_metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param is_show: if True, the plot is showed to the user
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    for model in models:
        for i in range(1, num+1):
            dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                              distrib=distribution, alpha=alpha, goal=goal, config=config)
            add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metrics,
                                       "Model " + model,
                                       "model/" + model,
                                       "model/",
                                       model,
                                       is_show if not i else False, config, i, goal)

    for metric in chosen_metrics:
        plot_agg_metric_dict("model",
                             "Model ", metric, config, goal, big=True)
    plot_bandwidth_dict("model",
                        "Model ", config)
    plot_agg_metric_scatter("model",
                            "Model ",
                            chosen_metrics[0], chosen_metrics[1], config, goal, big=True)


def compare_lr(datasize, distribution, alpha, model, chosen_metrics, epochs, is_show, quantiles, lrs, config, num, goal):
    """
    Comparison of different learning rate values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param model: chosen model to train and evaluate
    :param chosen_metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param is_show: if True, the plot is showed to the user
    :param quantiles: vector of quantiles to evaluate
    :param lrs: vector of lrs argument of the model
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    for lr in lrs:
        for i in range(1, num+1):
            dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                              distrib=distribution, alpha=alpha, learning_rate=lr, goal=goal, config=config)
            add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metrics,
                                       "Learning Rate " + model + " " + str(lr),
                                       "lr/" + model + "/" + str(lr),
                                       "lr/" + model,
                                       str(lr),
                                       is_show, config, i, goal)
    for metric in chosen_metrics:
        plot_agg_metric_dict("lr/" + model, "Learning Rate " + model, metric, config, goal)
    plot_bandwidth_dict("lr/" + model, "Learning Rate " + model, config)
    plot_agg_metric_scatter("lr/" + model, "Learning Rate " + model, chosen_metrics[0], chosen_metrics[1], config, goal)


def compare_dataset_size(dataset_sizes, distribution, alpha, model, chosen_metrics, epochs, is_show, quantiles, config, num, goal):
    """
    Comparison of different size of dataset time series values
    :param dataset_sizes: vector of designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param model: chosen model to train and evaluate
    :param chosen_metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param is_show: if True, the plot is showed to the user
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    for dataset_size in dataset_sizes:
        for i in range(1, num+1):
            dataset, forecasts, tss = forecast_dataset(dataset_size, quantiles=quantiles, model=model, epochs=epochs,
                                              distrib=distribution, alpha=alpha, goal=goal, config=config)
            add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metrics,
                                       "Dataset Size " + model + " " + str(dataset_size),
                                       "datasize/" + model + "/" + str(dataset_size),
                                       "datasize/" + model,
                                       str(dataset_size),
                                       is_show, config, i, goal)
    for metric in chosen_metrics:
        plot_agg_metric_dict("datasize/" + model, "Dataset Size " + model, metric, config, goal)
    plot_bandwidth_dict("datasize/" + model, "Dataset Size " + model, config)
    plot_agg_metric_scatter("datasize/" + model, "Dataset Size " + model, chosen_metrics[0], chosen_metrics[1], config, goal)


def compare_use_static(datasize, distribution, alpha, model, chosen_metrics, epochs, is_show, quantiles, config, num, goal):
    """
    Comparison of with or without the use of static features
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param model: chosen model to train and evaluate
    :param chosen_metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param is_show: if True, the plot is showed to the user
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    for use_static in [True, False]:
        for i in range(1, num+1):
            dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                              distrib=distribution, alpha=alpha, use_static=use_static, goal=goal, config=config)
            add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metrics,
                                       "Use Static " + model + " " + str(use_static),
                                       "use_static/" + model + "/" + str(use_static),
                                       "use_static/" + model,
                                       str(use_static),
                                       is_show, config, i, goal)
    for metric in chosen_metrics:
        plot_agg_metric_dict("use_static/" + model, "Use Static " + model, metric, config, goal)
    plot_bandwidth_dict("use_static/" + model, "Use Static " + model, config)
    plot_agg_metric_scatter("use_static/" + model, "Use Static " + model, chosen_metrics[0], chosen_metrics[1], config, goal)


def compare_epochs(datasize, distribution, alpha, model, chosen_metrics, epochss, is_show, quantiles, config, num, goal):
    """
    Comparison of different epochs values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param model: chosen model to train and evaluate
    :param chosen_metrics: vector of metrics selected
    :param epochss epochs argument of the model:
    :param is_show: if True, the plot is showed to the user
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    for epochs in epochss:
        for i in range(1, num+1):
            dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                              distrib=distribution, alpha=alpha, goal=goal, config=config)
            add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metrics,
                                       "Epochs " + model + " " + str(epochs),
                                       "epochs/" + model + "/" + str(epochs),
                                       "epochs/" + model,
                                       str(epochs),
                                       is_show, config, i, goal)
    for metric in chosen_metrics:
        plot_agg_metric_dict("epochs/" + model, "Epochs " + model, metric, config, goal)
    plot_bandwidth_dict("epochs/" + model, "Epochs " + model, config)
    plot_agg_metric_scatter("epochs/" + model, "Epochs " + model, chosen_metrics[0], chosen_metrics[1], config, goal)


def compare_context_length(datasize, distribution, alpha, model, chosen_metrics, epochs, is_show, quantiles, config,
                           context_lengths, num, goal):
    """
    Comparison of different context length values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param model: chosen model to train and evaluate
    :param chosen_metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param is_show: if True, the plot is showed to the user
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param context_lengths: vector of context length argument of the model
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    for context_length in context_lengths:
        for i in range(1, num+1):
            dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                              distrib=distribution, alpha=alpha, context_length=context_length, goal=goal, config=config)
            add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metrics,
                                       "Context Length " + model + " " + str(context_length),
                                       "context/" + model + "/" + str(context_length),
                                       "context/" + model,
                                       str(context_length),
                                       is_show, config, i, goal)
    for metric in chosen_metrics:
        plot_agg_metric_dict("context/" + model, "Context Length " + model, metric, config, goal)
    plot_bandwidth_dict("context/" + model, "Context Length " + model, config)
    plot_agg_metric_scatter("context/" + model, "Context Length " + model, chosen_metrics[0], chosen_metrics[1], config, goal)


def compare_num_pieces(datasize, alpha, model, chosen_metrics, epochs, is_show, quantiles, config, num_piecess, num, goal):
    """
    Comparison of different number of pieces for PieceWiseLinear distibution values
    :param datasize: designated size of the dataset time series
    :param alpha: alpha values argument of the model
    :param model: chosen model to train and evaluate
    :param chosen_metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param is_show: if True, the plot is showed to the user
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num_piecess: vector of number of pieces argument of the model
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    for num_pieces in num_piecess:
        for i in range(1, num+1):
            dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                              distrib="PiecewiseLinear", alpha=alpha, num_pieces=num_pieces, goal=goal, config=config)
            add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metrics,
                                       "Num Pieces " + model + " " + str(num_pieces),
                                       "num_pieces/" + model + "/" + str(num_pieces),
                                       "num_pieces/" + model,
                                       str(num_pieces),
                                       is_show, config, i, goal)
    for metric in chosen_metrics:
        plot_agg_metric_dict("num_pieces/" + model, "Num Pieces " + model, metric, config, goal)
    plot_bandwidth_dict("num_pieces/" + model, "Num Pieces " + model, config)
    plot_agg_metric_scatter("num_pieces/" + model, "Num Pieces " + model, chosen_metrics[0], chosen_metrics[1], config, goal)


def plot_compare_model(model, params_name, metrics, config, goal):
    """
    Plot of metrics of models hyperparameter values comparison
    :param model: chosen model to train and evaluate
    :param params_name: name designating the parameter to tune
    :param metrics: vector of metrics selected
    :param config: "A or "B", indicating the dataset configuration
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    for metric in metrics:
        plot_agg_metric_dict(model + "/" + params_name, model + " " + params_name, metric, config, goal)
    plot_bandwidth_dict(model + "/" + params_name, model + " " + params_name, config)
    plot_agg_metric_scatter(model + "/" + params_name, model + " " + params_name, metrics[0], metrics[1], config, goal)


def compare_simple(datasize, distribution, alpha, metrics, epochs, num_cells, quantiles, config, num, goal):
    """
    Comparison of different Simple model hyperparameter values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param num_cells: vector of values of hyperparameter to tune
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    model = "cSimple"
    params_name = "n_cell"
    for num_cell in num_cells:
        for i in range(1, num+1):
            dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                              distrib=distribution, alpha=alpha, num_cells_simple=num_cell, goal=goal, config=config)
            params_val = str(num_cell)
            add_metrics_plots_and_save(dataset, forecasts, tss, metrics,
                                       model + " " + params_name + " " + params_val,
                                       model + "/" + params_name + "/" + params_val,
                                       model + "/" + params_name, params_val, False, config, i, goal)
    plot_compare_model(model, params_name, metrics, config, goal)


def compare_simple_feed_forward(datasize, distribution, alpha, metrics, epochs, num_hidden_dimensions, quantiles, config, num, goal):
    """
    Comparison of different SimpleFeedForward model hyperparameter values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param num_hidden_dimensions: vector of values of hyperparameter to tune
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    model = "cSimpleFeedForward"
    params_name = "n_hidden_dim"
    for num_hidden_dimension in num_hidden_dimensions:
        for i in range(1, num + 1):
            dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                              distrib=distribution, alpha=alpha, num_hidden_dimensions=num_hidden_dimension, goal=goal, config=config)
            params_val = str(num_hidden_dimension)
            add_metrics_plots_and_save(dataset, forecasts, tss, metrics,
                                       model + " " + params_name + " " + params_val,
                                       model + "/" + params_name + "/" + params_val,
                                       model + "/" + params_name, params_val, False, config, i, goal)
    plot_compare_model(model, params_name, metrics, config, goal)


def compare_canonicalrnn(datasize, distribution, alpha, metrics, epochs, params_name, values, quantiles, config, num, goal):
    """
    Comparison of different CanonicalRNN model hyperparameter values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param params_name: name designating the parameter to tune
    :param values: vector of values of hyperparameter to tune
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    model = "cCanonicalRNN"
    for value in values:
        for i in range(1, num + 1):
            if params_name == "n_layers":
                dataset, forecasts, tss = forecast_dataset(datasize, model=model, epochs=epochs, distrib=distribution,
                                                  alpha=alpha, quantiles=quantiles, num_layers_rnn=value, goal=goal, config=config)
            else:
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, num_cells_rnn=value, goal=goal, config=config)
            params_val = str(value)
            add_metrics_plots_and_save(dataset, forecasts, tss, metrics,
                                       model + " " + params_name + " " + params_val,
                                       model + "/" + params_name + "/" + params_val,
                                       model + "/" + params_name, params_val, False, config, i, goal)
    plot_compare_model(model, params_name, metrics, config, goal)


def compare_deepar(datasize, distribution, alpha, metrics, epochs, params_name, values, quantiles, config, num, goal):
    """
    Comparison of different DeepAr model hyperparameter values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param params_name: name designating the parameter to tune
    :param values: vector of values of hyperparameter to tune
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    model = "DeepAr"
    for value in values:
        for i in range(1, num + 1):
            if params_name == "n_layers":
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, num_layers_ar=value, goal=goal, config=config)
            else:
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, num_cells_ar=value, goal=goal, config=config)
            params_val = str(value)
            add_metrics_plots_and_save(dataset, forecasts, tss, metrics,
                                       model + " " + params_name + " " + params_val,
                                       model + "/" + params_name + "/" + params_val,
                                       model + "/" + params_name, params_val, False, config, i, goal)
    plot_compare_model(model, params_name, metrics, config, goal)


def compare_deepfactor(datasize, distribution, alpha, metrics, epochs, params_name, values, quantiles, config, num, goal):
    """
    Comparison of different DeepFactor model hyperparameter values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param params_name: name designating the parameter to tune
    :param values: vector of values of hyperparameter to tune
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    model = "DeepFactor"
    for value in values:
        for i in range(1, num + 1):
            if params_name == "n_hidden_global":
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, num_hidden_global=value, goal=goal, config=config)
            elif params_name == "n_layers_global":
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, num_layers_global=value, goal=goal, config=config)
            else:
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, num_factors=value, goal=goal, config=config)
            params_val = str(value)
            add_metrics_plots_and_save(dataset, forecasts, tss, metrics,
                                       model + " " + params_name + " " + params_val,
                                       model + "/" + params_name + "/" + params_val,
                                       model + "/" + params_name, params_val, False, config, i, goal)
    plot_compare_model(model, params_name, metrics, config, goal)


def compare_mqcnn(datasize, distribution, alpha, metrics, epochs, params_name, values, quantiles, config, num, goal):
    """
    Comparison of different MQCNN model hyperparameter values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param params_name: name designating the parameter to tune
    :param values: vector of values of hyperparameter to tune
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    model = "MQCNN"
    for value in values:
        for i in range(1, num + 1):
            if params_name == "mlp_final":
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, mlp_final_dim_c=value, goal=goal, config=config)
            else:
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, mlp_hidden_c=value, goal=goal, config=config)
            params_val = str(value)
            add_metrics_plots_and_save(dataset, forecasts, tss, metrics,
                                       model + " " + params_name + " " + params_val,
                                       model + "/" + params_name + "/" + params_val,
                                       model + "/" + params_name, params_val, False, config, i, goal)
    plot_compare_model(model, params_name, metrics, config, goal)


def compare_mqrnn(datasize, distribution, alpha, metrics, epochs, params_name, values, quantiles, config, num, goal):
    """
    Comparison of different MQRNN model hyperparameter values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param params_name: name designating the parameter to tune
    :param values: vector of values of hyperparameter to tune
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    model = "MQRNN"
    for value in values:
        for i in range(1, num + 1):
            if params_name == "mlp_final":
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, mlp_final_dim_c=value, goal=goal, config=config)
            else:
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, mlp_hidden_c=value, goal=goal, config=config)
            params_val = str(value)
            add_metrics_plots_and_save(dataset, forecasts, tss, metrics,
                                       model + " " + params_name + " " + params_val,
                                       model + "/" + params_name + "/" + params_val,
                                       model + "/" + params_name, params_val, False, config, i, goal)
    plot_compare_model(model, params_name, metrics, config, goal)


def compare_wavenet(datasize, distribution, alpha, metrics, epochs, params_name, values, quantiles, config, num, goal):
    """
    Comparison of different Wavenet model hyperparameter values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param params_name: name designating the parameter to tune
    :param values: vector of values of hyperparameter to tune
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    model = "Wavenet"
    for value in values:
        for i in range(1, num + 1):
            if params_name == "n_residue":
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, n_residue=value, goal=goal, config=config)
            elif params_name == "n_skip":
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, n_skip=value, goal=goal, config=config)
            else:
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, n_stacks=value, goal=goal, config=config)
            params_val = str(value)
            add_metrics_plots_and_save(dataset, forecasts, tss, metrics,
                                       model + " " + params_name + " " + params_val,
                                       model + "/" + params_name + "/" + params_val,
                                       model + "/" + params_name, params_val, False, config, i, goal)
    plot_compare_model(model, params_name, metrics, config, goal)


def compare_nbeats(datasize, distribution, alpha, metrics, epochs, params_name, values, quantiles, config, num, goal):
    """
    Comparison of different NBEATS model hyperparameter values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param params_name: name designating the parameter to tune
    :param values: vector of values of hyperparameter to tune
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num: indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    model = "Nbeats"
    for value in values:
        for i in range(1, num + 1):
            if params_name == "num_stacks":
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, num_stacks=value, goal=goal, config=config)
            else:
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, n_blocks=value, goal=goal, config=config)
            params_val = str(value)
            add_metrics_plots_and_save(dataset, forecasts, tss, metrics,
                                       model + " " + params_name + " " + params_val,
                                       model + "/" + params_name + "/" + params_val,
                                       model + "/" + params_name, params_val, False, config, i, goal)
    plot_compare_model(model, params_name, metrics, config, goal)


def compare_transformer(datasize, distribution, alpha, metrics, epochs, params_name, values, quantiles, config, num, goal):
    """
    Comparison of different Transformer model hyperparameter values
    :param datasize: designated size of the dataset time series
    :param distribution: output distribution of the chosen model
    :param alpha: alpha values argument of the model
    :param metrics: vector of metrics selected
    :param epochs: epochs argument of the model
    :param params_name: name designating the parameter to tune
    :param values: vector of values of hyperparameter to tune
    :param quantiles: vector of quantiles to evaluate
    :param config: "A or "B", indicating the dataset configuration
    :param num:  indicates the number of similar executions of this function that has been performed just before.
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    """
    model = "Transformer"
    for value in values:
        for i in range(1, num + 1):
            if params_name == "model_dim":
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, model_dim=value, goal=goal, config=config)
            else:
                dataset, forecasts, tss = forecast_dataset(datasize, quantiles=quantiles, model=model, epochs=epochs,
                                                  distrib=distribution, alpha=alpha, num_heads=value, goal=goal, config=config)
            params_val = str(value)
            add_metrics_plots_and_save(dataset, forecasts, tss, metrics,
                                       model + " " + params_name + " " + params_val,
                                       model + "/" + params_name + "/" + params_val,
                                       model + "/" + params_name, params_val, False, config, i, goal)
    plot_compare_model(model, params_name, metrics, config, goal)

