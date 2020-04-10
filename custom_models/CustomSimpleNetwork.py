from mxnet import gluon
import mxnet as mx
from mxnet import nd
import numpy as np
from utils import compute_custom_loss, save_distr_params


class CustomSimpleNetwork(gluon.HybridBlock):
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


class CustomSimpleTrainNetwork(CustomSimpleNetwork):
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
        loss = compute_custom_loss(loss, self.distr_output_type, self.alpha, distr, future_target)

        return loss


class CustomSimplePredNetwork(CustomSimpleNetwork):
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
        save_distr_params(distr, self.count, self.distr_output_type, self.alpha, "cSimple")

        # get (batch_size * num_sample_paths, prediction_length) samples
        samples = distr.sample()

        # reshape from (batch_size * num_sample_paths, prediction_length) to
        # (batch_size, num_sample_paths, prediction_length)

        return samples.reshape(shape=(-1, self.num_sample_paths, self.prediction_length))
