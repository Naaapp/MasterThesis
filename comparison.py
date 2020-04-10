from forecast import forecast_dataset
from plots import add_agg_metric_to_dict, add_bandwidth_to_dict, plot_prob_forecasts, save_item_metrics, \
    plot_agg_metric_dict, plot_bandwidth_dict, plot_distr_params


def add_metrics_plots_and_save(dataset, forecasts, tss, metric, plot_name, params_name, params_val, is_show):
    complete_name = plot_name + "_" + params_name + "_" + params_val
    add_agg_metric_to_dict(dataset, forecasts, tss, metric, plot_name, params_name, params_val)
    add_bandwidth_to_dict(forecasts, plot_name, params_name, params_val)
    plot_prob_forecasts(tss[0], forecasts[0], 60, [50, 90, 99], plot_name, complete_name, is_show)
    plot_prob_forecasts(tss[0], forecasts[0], 60 * 3, [50, 90, 99], plot_name, complete_name, is_show)
    save_item_metrics(dataset, forecasts, tss, plot_name, metric)
    # if model in ["MQCNN", "MQRNN"]:
    #     save_distr_quantiles(model, forecasts[0], quantiles)


def compare_all_models(dataset, distributions, alphas, models, chosen_metric, epochs, is_show):
    for distribution in distributions:
        for alpha in alphas:
            for model in models:
                forecasts, tss = forecast_dataset(
                    dataset,
                    model=model,
                    epochs=epochs,
                    distrib=distribution,
                    alpha=alpha
                )
                params_val = model + "_" + str(alpha) + "_" + distribution
                add_metrics_plots_and_save(dataset, forecasts, tss, chosen_metric, model, "", params_val, is_show)

    plot_agg_metric_dict(chosen_metric, "", "")
    plot_bandwidth_dict("", "")

    # if not [model in ["MQCNN", "MQRNN"] for model in models]:
    #     plot_distr_params(models, alphas, distributions)


def compare_simple(dataset, distribution, alpha, metric, epochs, num_cells):
    model = "cSimple"
    params_name = "n_cell"
    for num_cell in num_cells:
        forecasts, tss = forecast_dataset(
            dataset,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha,
            num_cells_simple=num_cell
        )
        params_val = str(num_cell)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric, model, params_name, params_val)

    plot_agg_metric_dict(metric, model, params_name)
    plot_bandwidth_dict(model, params_name)


def compare_simple_feed_forward(dataset, distribution, alpha, metric, epochs, num_hidden_dimensions):
    model = "cSimpleFeedForward"
    params_name = "n_hidden_dim"
    for num_hidden_dimension in num_hidden_dimensions:
        forecasts, tss = forecast_dataset(
            dataset,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha,
            num_hidden_dimensions=num_hidden_dimension
        )
        params_val = str(num_hidden_dimension)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric, model, params_name, params_val)

    plot_agg_metric_dict(metric, model, params_name)
    plot_bandwidth_dict(model, params_name)


def compare_canonicalrnn(dataset, distribution, alpha, metric, epochs, params_name, values):
    model = "cCanonicalRNN"
    for value in values:
        if params_name == "n_layers":
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_layers_rnn=value
            )
        else:
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_cells_rnn=value
            )
        params_val = str(value)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric, model, params_name, params_val)

    plot_agg_metric_dict(metric, model, params_name)
    plot_bandwidth_dict(model, params_name)


def compare_deepar(dataset, distribution, alpha, metric, epochs, params_name, values):
    model = "DeepAr"
    for value in values:
        if params_name == "n_cells":
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_layers_ar=value
            )
        else:
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_cells_ar=value
            )
        params_val = str(value)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric, model, params_name, params_val)

    plot_agg_metric_dict(metric, model, params_name)
    plot_bandwidth_dict(model, params_name)


def compare_deepfactor(dataset, distribution, alpha, metric, epochs, params_name, values):
    model = "DeepFactor"
    for value in values:
        if params_name == "n_hidden_global":
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_hidden_global=value
            )
        elif params_name == "n_layers_global":
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_layers_global=value
            )
        else :
            forecasts, tss = forecast_dataset(
                dataset,
                model=model,
                epochs=epochs,
                distrib=distribution,
                alpha=alpha,
                num_factors=value
            )
        params_val = str(value)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric, model, params_name, params_val)

    plot_agg_metric_dict(metric, model, params_name)
    plot_bandwidth_dict(model, params_name)


def compare_mqcnn(dataset, distribution, alpha, metric, epochs, mlp_final_dims):
    model = "MQCNN"
    params_name = "mlp_final_dim"
    for mlp_final_dim in mlp_final_dims:
        forecasts, tss = forecast_dataset(
            dataset,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha,
            mlp_final_dim=mlp_final_dim
        )
        params_val = str(mlp_final_dim)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric, model, params_name, params_val)

    plot_agg_metric_dict(metric, model, params_name)
    plot_bandwidth_dict(model, params_name)


def compare_mqrnn(dataset, distribution, alpha, metric, epochs, mlp_final_dims):
    model = "MQRNN"
    params_name = "mlp_final_dim"
    for mlp_final_dim in mlp_final_dims:
        forecasts, tss = forecast_dataset(
            dataset,
            model=model,
            epochs=epochs,
            distrib=distribution,
            alpha=alpha,
            mlp_final_dim=mlp_final_dim
        )
        params_val = str(mlp_final_dim)
        add_metrics_plots_and_save(dataset, forecasts, tss, metric, model, params_name, params_val)

    plot_agg_metric_dict(metric, model, params_name)
    plot_bandwidth_dict(model, params_name)
