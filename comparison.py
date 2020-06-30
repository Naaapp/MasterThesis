from forecast import forecast_dataset
from imports import import_dataset
from plots import *
from save import *


def add_metrics_plots_and_save(dataset, forecasts, tss, metric, plot_name, plot_loc, agg_metric_loc, agg_metric_id,
                               is_show, config):
    save_agg_metric_to_dict(dataset, forecasts, tss, metric, agg_metric_loc, agg_metric_id, config)
    save_bandwidth_to_dict(forecasts, plot_name, agg_metric_loc, agg_metric_id, config)
    plot_forecast_entry(tss[0], forecasts[0], 60, [50, 90, 99], plot_name, plot_loc, is_show, config)
    plot_forecast_entry(tss[0], forecasts[0], 60 * 3, [50, 90, 99], plot_name, plot_loc, is_show, config)
    save_item_metrics(dataset, forecasts, tss, metric, agg_metric_loc, agg_metric_id, config)


def compare_all_models(dataset, distributions, alphas, models, chosen_metric, epochs, is_show, quantiles, config):
    for distribution in distributions:
        for alpha in alphas:
            for model in models:
                forecasts, tss = forecast_dataset(
                    dataset,
                    quantiles=quantiles,
                    model=model,
                    epochs=epochs,
                    distrib=distribution,
                    alpha=alpha
                )
                add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metric,
                                           "Global " + model + " " + str(alpha).replace(".", "_") + " " + distribution,
                                           "global/" + model + "/" + str(alpha).replace(".", "_") + "/" + distribution,
                                           "global",
                                           model + " " + str(alpha).replace(".", "_") + " " + distribution,
                                           is_show, config)

    plot_agg_metric_dict("global", "Global ", chosen_metric, config)
    plot_bandwidth_dict("global", "Global ", config)
    plot_agg_metric_scatter("global", "Global", chosen_metric, "bandwidth", config)


def compare_alpha(dataset, distribution, alphas, model, chosen_metric, epochs, is_show, quantiles, config):
    for alpha in alphas:
        forecasts, tss = forecast_dataset(
            dataset,
            quantiles=quantiles,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha
        )
        add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metric,
                                   "Alpha " + model + " " + distribution + " " + str(alpha).replace(".", "_"),
                                   "alpha/" + model + "/" + distribution + "/" + str(alpha).replace(".", "_"),
                                   "alpha/" + model + "/" + distribution,
                                   str(alpha).replace(".", "_"),
                                   is_show, config)

    plot_agg_metric_dict("alpha/" + model + "/" + distribution, "Alpha " + model + " " + distribution, chosen_metric, config)
    plot_bandwidth_dict("alpha/" + model + "/" + distribution, "Alpha " + model + " " + distribution, config)
    plot_agg_metric_scatter("alpha/" + model + "/" + distribution,
                            "Alpha " + model + " " + distribution, chosen_metric, "bandwidth", config)


def compare_distrib(dataset, distributions, alpha, model, chosen_metric, epochs, is_show, quantiles, config):
    for distribution in distributions:
        forecasts, tss = forecast_dataset(
            dataset,
            quantiles=quantiles,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha
        )
        add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metric,
                                   "Distribution " + model + " " + str(alpha).replace(".", "_") + " " + distribution,
                                   "distribution/" + model + "/" + str(alpha).replace(".", "_") + "/" + distribution,
                                   "distribution/" + model + "/" + str(alpha).replace(".", "_"),
                                   distribution,
                                   is_show, config)

    plot_agg_metric_dict("distribution/" + model + "/" + str(alpha).replace(".", "_"), "Distribution " + model
                         + " " + str(alpha).replace(".", "_"), chosen_metric, config)
    plot_bandwidth_dict("distribution/" + model + "/" + str(alpha).replace(".", "_"), "Distribution " + model
                        + " " + str(alpha).replace(".", "_"), config)
    plot_agg_metric_scatter("distribution/" + model + "/" + str(alpha).replace(".", "_"), "Distribution " + model
                            + " " + str(alpha).replace(".", "_"), chosen_metric, "bandwidth", config)


def compare_model(dataset, distribution, alpha, models, chosen_metric, epochs, is_show, quantiles, config):
    for model in models:
        forecasts, tss = forecast_dataset(
            dataset,
            quantiles=quantiles,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha
        )
        add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metric,
                                   "Model " + distribution + " " + str(alpha).replace(".", "_") + " " + model,
                                   "model/" + distribution + "/" + str(alpha).replace(".", "_") + "/" + model,
                                   "model/" + str(alpha).replace(".", "_") + "/" + distribution,
                                   model,
                                   is_show, config)

    plot_agg_metric_dict("model/" + str(alpha).replace(".", "_") + "/" + distribution,
                         "Model " + str(alpha).replace(".", "_") + "/" + distribution, chosen_metric, config)
    plot_bandwidth_dict("model/" + str(alpha).replace(".", "_") + "/" + distribution,
                        "Model " + str(alpha).replace(".", "_") + "/" + distribution, config)
    plot_agg_metric_scatter("model/" + str(alpha).replace(".", "_") + "/" + distribution,
                            "Model " + str(alpha).replace(".", "_") + "/" + distribution, chosen_metric, "bandwidth", config)


def compare_lr(dataset, distribution, alpha, model, chosen_metric, epochs, is_show, quantiles, lrs, config):
    for lr in lrs:
        forecasts, tss = forecast_dataset(
            dataset,
            quantiles=quantiles,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha,
            learning_rate=lr
        )
        add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metric,
                                   "Learning Rate " + model + " " + lr,
                                   "lr/" + model + "/" + lr,
                                   "lr/" + model,
                                   lr,
                                   is_show, config)

    plot_agg_metric_dict("lr/" + model, "Learning Rate " + model, chosen_metric, config)
    plot_bandwidth_dict("lr/" + model, "Learning Rate " + model, config)
    plot_agg_metric_scatter("lr/" + model, "Learning Rate " + model, chosen_metric, "bandwidth", config)


def compare_dataset_size(dataset_sizes, distribution, alpha, model, chosen_metric, epochs, is_show, quantiles, config):
    for dataset_size in dataset_sizes:
        dataset = import_dataset(["6months-minutes"], ["2eol_measurements"], dataset_size)
        forecasts, tss = forecast_dataset(
            dataset,
            quantiles=quantiles,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha,
        )
        add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metric,
                                   "Dataset Size " + model + " " + str(dataset_size),
                                   "datasize/" + model + "/" + str(dataset_size),
                                   "datasize/" + model,
                                   str(dataset_size),
                                   is_show, config)

    plot_agg_metric_dict("datasize/" + model, "Dataset Size " + model, chosen_metric, config)
    plot_bandwidth_dict("datasize/" + model, "Dataset Size " + model, config)
    plot_agg_metric_scatter("datasize/" + model, "Dataset Size " + model, chosen_metric, "bandwidth", config)


def compare_use_static(dataset, distribution, alpha, model, chosen_metric, epochs, is_show, quantiles, config):
    for use_static in [True, False]:
        forecasts, tss = forecast_dataset(
            dataset,
            quantiles=quantiles,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha,
            use_static=use_static,
            cardinality=dataset.cardinality_train
        )
        add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metric,
                                   "Use Static " + model + " " + str(use_static),
                                   "use_static/" + model + "/" + str(use_static),
                                   "use_static/" + model,
                                   str(use_static),
                                   is_show, config)

    plot_agg_metric_dict("use_static/" + model, "Use Static " + model, chosen_metric, config)
    plot_bandwidth_dict("use_static/" + model, "Use Static " + model, config)
    plot_agg_metric_scatter("use_static/" + model, "Use Static " + model, chosen_metric, "bandwidth", config)


def compare_simple(dataset, distribution, alpha, metric, epochs, num_cells, quantiles, config):
    model = "cSimple"
    params_name = "n_cell"
    for num_cell in num_cells:
        forecasts, tss = forecast_dataset(
            dataset,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha,
            num_cells_simple=num_cell,
            quantiles=quantiles
        )
        params_val = str(num_cell)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric,
                                   model + " " + params_name + " " + params_val,
                                   model + "/" + params_name + "/" + params_val,
                                   model + "/" + params_name,
                                   params_val,
                                   False,
                                   config)

    plot_agg_metric_dict(model + "/" + params_name, model + " " + params_name, metric, config)
    plot_bandwidth_dict(model + "/" + params_name, model + " " + params_name, config)
    plot_agg_metric_scatter(model + "/" + params_name, model + " " + params_name, metric, "bandwidth", config)


def compare_simple_feed_forward(dataset, distribution, alpha, metric, epochs, num_hidden_dimensions, quantiles, config):
    model = "cSimpleFeedForward"
    params_name = "n_hidden_dim"
    for num_hidden_dimension in num_hidden_dimensions:
        forecasts, tss = forecast_dataset(
            dataset,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha,
            num_hidden_dimensions=num_hidden_dimension,
            quantiles=quantiles
        )
        params_val = str(num_hidden_dimension)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric,
                                   model + " " + params_name + " " + params_val,
                                   model + "/" + params_name + "/" + params_val,
                                   model + "/" + params_name,
                                   params_val,
                                   False,
                                   config)

    plot_agg_metric_dict(model + "/" + params_name, model + " " + params_name, metric, config)
    plot_bandwidth_dict(model + "/" + params_name, model + " " + params_name, config)
    plot_agg_metric_scatter(model + "/" + params_name, model + " " + params_name, metric, "bandwidth", config)


def compare_canonicalrnn(dataset, distribution, alpha, metric, epochs, params_name, values, quantiles, config):
    model = "cCanonicalRNN"
    for value in values:
        if params_name == "n_layers":
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_layers_rnn=value,
                quantiles=quantiles
            )
        else:
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_cells_rnn=value,
                quantiles=quantiles
            )
        params_val = str(value)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric,
                                   model + " " + params_name + " " + params_val,
                                   model + "/" + params_name + "/" + params_val,
                                   model + "/" + params_name,
                                   params_val,
                                   False, config)

    plot_agg_metric_dict(model + "/" + params_name, model + " " + params_name, metric, config)
    plot_bandwidth_dict(model + "/" + params_name, model + " " + params_name, config)
    plot_agg_metric_scatter(model + "/" + params_name, model + " " + params_name, metric, "bandwidth", config)


def compare_deepar(dataset, distribution, alpha, metric, epochs, params_name, values, quantiles, config):
    model = "DeepAr"
    for value in values:
        if params_name == "n_cells":
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_layers_ar=value,
                quantiles=quantiles
            )
        else:
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_cells_ar=value,
                quantiles=quantiles
            )
        params_val = str(value)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric,
                                   model + " " + params_name + " " + params_val,
                                   model + "/" + params_name + "/" + params_val,
                                   model + "/" + params_name,
                                   params_val,
                                   False, config)

    plot_agg_metric_dict(model + "/" + params_name, model + " " + params_name, metric, config)
    plot_bandwidth_dict(model + "/" + params_name, model + " " + params_name, config)
    plot_agg_metric_scatter(model + "/" + params_name, model + " " + params_name, metric, "bandwidth", config)


def compare_deepfactor(dataset, distribution, alpha, metric, epochs, params_name, values, quantiles, config):
    model = "DeepFactor"
    for value in values:
        if params_name == "n_hidden_global":
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_hidden_global=value,
                quantiles=quantiles
            )
        elif params_name == "n_layers_global":
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_layers_global=value,
                quantiles=quantiles
            )
        else:
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_factors=value,
                quantiles=quantiles
            )
        params_val = str(value)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric,
                                   model + " " + params_name + " " + params_val,
                                   model + "/" + params_name + "/" + params_val,
                                   model + "/" + params_name,
                                   params_val,
                                   False, config)

    plot_agg_metric_dict(model + "/" + params_name, model + " " + params_name, metric, config)
    plot_bandwidth_dict(model + "/" + params_name, model + " " + params_name, config)
    plot_agg_metric_scatter(model + "/" + params_name, model + " " + params_name, metric, "bandwidth", config)


def compare_mqcnn(dataset, distribution, alpha, metric, epochs, mlp_final_dims, quantiles, config):
    model = "MQCNN"
    params_name = "mlp_final_dim"
    for mlp_final_dim in mlp_final_dims:
        forecasts, tss = forecast_dataset(
            dataset,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha,
            mlp_final_dim=mlp_final_dim,
            quantiles=quantiles
        )
        params_val = str(mlp_final_dim)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric,
                                   model + " " + params_name + " " + params_val,
                                   model + "/" + params_name + "/" + params_val,
                                   model + "/" + params_name,
                                   params_val,
                                   False, config)
        save_distr_quantiles(model, forecasts[0], quantiles, config)

    plot_agg_metric_dict(model + "/" + params_name, model + " " + params_name, metric, config)
    plot_bandwidth_dict(model + "/" + params_name, model + " " + params_name, config)
    plot_agg_metric_scatter(model + "/" + params_name, model + " " + params_name, metric, "bandwidth", config)


def compare_mqrnn(dataset, distribution, alpha, metric, epochs, mlp_final_dims, quantiles, config):
    model = "MQRNN"
    params_name = "mlp_final_dim"
    for mlp_final_dim in mlp_final_dims:
        forecasts, tss = forecast_dataset(
            dataset,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha,
            mlp_final_dim=mlp_final_dim,
            quantiles=quantiles
        )
        params_val = str(mlp_final_dim)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric,
                                   model + " " + params_name + " " + params_val,
                                   model + "/" + params_name + "/" + params_val,
                                   model + "/" + params_name,
                                   params_val,
                                   False, config)
        save_distr_quantiles(model, forecasts[0], quantiles, config)

    plot_agg_metric_dict(model + "/" + params_name, model + " " + params_name, metric, config)
    plot_bandwidth_dict(model + "/" + params_name, model + " " + params_name, config)
    plot_agg_metric_scatter(model + "/" + params_name, model + " " + params_name, metric, "bandwidth", config)
