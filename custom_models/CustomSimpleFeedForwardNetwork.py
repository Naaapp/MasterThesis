from mxnet import gluon
import mxnet as mx
from mxnet import nd
import numpy as np

class CustomSimpleFeedForwardNetwork(gluon.HybridBlock):
    def __init__(self,
                 prediction_length,
                 distr_output,
                 distr_output_type,
                 num_cells,
                 num_sample_paths=100,
                 alpha=0,
                 count=0,
                 **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.distr_output_type = distr_output_type
        self.num_cells = num_cells
        self.num_sample_paths = num_sample_paths
        self.alpha = alpha
        self.proj_distr_args = distr_output.get_args_proj()
        self.count = count

        with self.name_scope():
            # Set up a 2 layer neural network that its ouput will be projected to the distribution parameters
            self.nn = mx.gluon.nn.HybridSequential()
            self.nn.add(mx.gluon.nn.Dense(units=self.num_cells, activation='relu'))
            self.nn.add(mx.gluon.nn.Dense(units=self.prediction_length * self.num_cells, activation='relu'))


class CustomSimpleFeedForwardTrainNetwork(CustomSimpleFeedForwardNetwork):
    def hybrid_forward(self, F, past_target, future_target):
        # compute network output
        net_output = self.nn(past_target)

        # (batch, prediction_length * nn_features)  ->  (batch, prediction_length, nn_features)
        net_output = net_output.reshape(0, self.prediction_length, -1)

        # project network output to distribution parameters domain
        distr_args = self.proj_distr_args(net_output)

        # compute distribution
        distr = self.distr_output.distribution(distr_args)

        # negative log-likelihood
        loss = distr.loss(future_target)

        # custom quantile based loss
        if self.distr_output_type == "Gaussian":
            alpha = self.alpha
            quantile_high = distr.quantile(nd.array([0.995]))[0]
            quantile_low = distr.quantile(nd.array([0.005]))[0]
            future_high = future_target - quantile_high
            future_low = quantile_low - future_target
            loss1 = nd.exp(future_high)*alpha
            loss2 = nd.exp(future_low)*alpha
            loss = loss1 + loss2 + loss

        return loss


class CustomSimpleFeedForwardPredNetwork(CustomSimpleFeedForwardNetwork):
    # The prediction network only receives past_target and returns predictions
    def hybrid_forward(self, F, past_target):
        # repeat past target: from (batch_size, past_target_length) to
        # (batch_size * num_sample_paths, past_target_length)
        repeated_past_target = past_target.repeat(
            repeats=self.num_sample_paths, axis=0
        )

        # compute network output
        net_output = self.nn(repeated_past_target)

        # (batch * num_sample_paths, prediction_length * nn_features)  ->  (batch * num_sample_paths, prediction_length, nn_features)
        net_output = net_output.reshape(0, self.prediction_length, -1)

        # project network output to distribution parameters domain
        distr_args = self.proj_distr_args(net_output)

        # compute distribution
        distr = self.distr_output.distribution(distr_args)

        # Save the distribution parameters to file
        # For an unknown reason, predict is executed several times, only the first execution give the correct
        # prediction

        if not self.count:
            if self.distr_output_type == "Gaussian":
                distr_params = [distr.mu[0].asnumpy(), distr.sigma[0].asnumpy()]
            elif self.distr_output_type == "Laplace":
                distr_params = [distr.mu[0].asnumpy(), distr.b[0].asnumpy()]
            elif self.distr_output_type == "PiecewiseLinear":
                distr_params = [distr.gamma[0].asnumpy(), distr.slopes[0].asnumpy(), distr.knot_spacings[0].asnumpy()]
            elif self.distr_output_type == "Uniform":
                distr_params = [distr.low[0].asnumpy(), distr.high[0].asnumpy()]
            elif self.distr_output_type == "Student":
                distr_params = [distr.mu[0].asnumpy(), distr.sigma[0].asnumpy(), distr.nu[0].asnumpy()]
            else:
                distr_params = []
            print(self.distr_output_type)
            np.save("distribution_output/cSimpleFeedForward_" + self.distr_output_type + "_" + str(self.alpha) +
                    ".npy", distr_params)
            self.count += 1

        # get (batch_size * num_sample_paths, prediction_length) samples
        samples = distr.sample()

        # reshape from (batch_size * num_sample_paths, prediction_length) to
        # (batch_size, num_sample_paths, prediction_length)

        return samples.reshape(shape=(-1, self.num_sample_paths, self.prediction_length))