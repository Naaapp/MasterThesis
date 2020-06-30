from mxnet import nd
import numpy as np


def compute_custom_loss(loss, alpha, distr, future_target):
    alpha = alpha
    quantile_high = distr.quantile(nd.array([0.995]))[0]
    quantile_low = distr.quantile(nd.array([0.005]))[0]
    future_high = future_target - quantile_high
    future_low = quantile_low - future_target
    loss1 = nd.exp(future_high) * alpha
    loss2 = nd.exp(future_low) * alpha
    loss = loss1 + loss2 + loss
    return loss


def save_distr_params(distr, count, distr_output_type, alpha, model):
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
            distr_params = [distr.mu[0].asnumpy(), distr.sigma[0].asnumpy(), distr.mu[0].asnumpy()]
        else:
            distr_params = []
        np.save("distribution_output/" + model + "_" + distr_output_type + "_" +
                str(alpha).replace(".", "_") + ".npy", distr_params)
        count += 1
