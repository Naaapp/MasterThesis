# Master Thesis (Théo Stassen, Université de Liège) :
# "Comparison of probabilistic forecasting deep learning models in the context of renewable energy production"
#
# - Different utils functions


from mxnet import nd
import numpy as np


def compute_custom_loss(loss, alpha, distr, future_target):
    """
    Compute the custom loss
    :param loss: value of the default loss
    :param alpha: alpha parameter value
    :param distr: output distribution object predicted
    :param future_target: value obsereved
    :return: value of the custom loss
    """
    alpha = alpha
    quantile_high = distr.quantile(nd.array([0.995]))[0]
    future_high = future_target - quantile_high
    loss1 = nd.exp(future_high) * alpha
    loss = loss1 + loss
    return loss


def save_distr_params(distr, count, distr_output_type, alpha, model):
    """
    Save the parameters of the distribution output object -> Not used in current version
    :param distr: output distribution object predicted
    :param count: value that count the number of saves
    :param distr_output_type: type of the distribution output object
    :param alpha: value of parameter alpha
    :param model: selected model
    """
    if not count:
        if distr_output_type == "Gaussian":
            distr_params = [distr.mu[0].asnumpy(), distr.sigma[0].asnumpy()]
        elif distr_output_type == "Laplace":
            distr_params = [distr.mu[0].asnumpy(), distr.b[0].asnumpy()]
        elif distr_output_type == "PiecewiseLinear":
            distr_params = [distr.gamma[0].asnumpy(),
                            nd.transpose(distr.b[0])[0].asnumpy(),
                            nd.transpose(distr.knot_positions[0])[0].asnumpy()]
        elif distr_output_type == "Uniform":
            distr_params = [distr.low[0].asnumpy(), distr.high[0].asnumpy()]
        elif distr_output_type == "Student":
            distr_params = [distr.mu[0].asnumpy(), distr.sigma[0].asnumpy(), distr.nu[0].asnumpy()]
        else:
            distr_params = []
        np.save("distribution_output/" + model + "_" + distr_output_type + "_" +
                str(alpha).replace(".", "_") + ".npy", distr_params)
        count += 1
