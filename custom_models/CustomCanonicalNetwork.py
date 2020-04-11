# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Third-party imports
from mxnet.gluon import HybridBlock

# First-party imports
from gluonts.block.feature import FeatureEmbedder
from gluonts.block.scaler import MeanScaler
from gluonts.core.component import validated
from gluonts.distribution import DistributionOutput
from gluonts.model.common import Tensor

from utils import compute_custom_loss, save_distr_params


class CustomCanonicalNetworkBase(HybridBlock):
    @validated()
    def __init__(
        self,
        model: HybridBlock,
        embedder: FeatureEmbedder,
        distr_output: DistributionOutput,
        distr_output_type: str,
        alpha: int,
        is_sequential: bool,
        count=0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.distr_output = distr_output
        self.is_sequential = is_sequential
        self.model = model
        self.embedder = embedder
        self.distr_output_type = distr_output_type
        self.alpha = alpha
        self.count = count

        with self.name_scope():
            self.proj_distr_args = self.distr_output.get_args_proj()
            self.scaler = MeanScaler(keepdims=True)

    def assemble_features(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        time_feat: Tensor,  # (batch_size, history_length, num_features)
    ) -> Tensor:
        embedded_cat = self.embedder(
            feat_static_cat
        )  # (batch_size, num_features * embedding_size)

        # a workaround when you wish to repeat without knowing the number
        # of repeats
        helper_ones = F.ones_like(
            F.slice_axis(time_feat, axis=2, begin=-1, end=None)
        )
        # (batch_size, history_length, num_features * embedding_size)
        repeated_cat = F.batch_dot(
            helper_ones, F.expand_dims(embedded_cat, axis=1)
        )

        # putting together all the features
        input_feat = F.concat(repeated_cat, time_feat, dim=2)
        return input_feat

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError


class CustomCanonicalTrainingNetwork(CustomCanonicalNetworkBase):
    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,
        # (batch_size, num_features, history_length)
        past_target: Tensor,  # (batch_size, history_length)
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            Function space
        feat_static_cat
            Shape: (batch_size, num_features)
        past_time_feat
            Shape: (batch_size, history_length, num_features)
        past_target
            Shape: (batch_size, history_length)

        Returns
        -------
        Tensor
            A batch of negative log likelihoods.
        """
        _, target_scale = self.scaler(
            past_target,
            F.ones_like(past_target),  # TODO: pass the actual observed here
        )

        input_feat = self.assemble_features(F, feat_static_cat, past_time_feat)
        outputs = self.model(input_feat)

        distr = self.distr_output.distribution(
            self.proj_distr_args(outputs), scale=target_scale
        )

        loss = distr.loss(past_target)
        loss = compute_custom_loss(loss, self.distr_output_type, self.alpha, distr, past_target)

        return loss


class CustomCanonicalPredictionNetwork(CustomCanonicalNetworkBase):
    @validated()
    def __init__(
        self, prediction_len: int, num_parallel_samples: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.prediction_len = prediction_len
        self.num_parallel_samples = num_parallel_samples

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        past_target: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        F
            Function space module.
        feat_static_cat
            Shape: (batch_size, num_features).
        past_time_feat
            Shape: (batch_size, history_length, num_features).
        future_time_feat
            Shape: (batch_size, history_length, num_features).
        past_target
            Shape: (batch_size, history_length).

        Returns
        -------
        Tensor
            a batch of prediction samples
            Shape: (batch_size, prediction_length, num_sample_paths)
        """

        _, target_scale = self.scaler(
            past_target,
            F.ones_like(past_target),  # TODO: pass the actual observed here
        )

        time_feat = (
            F.concat(past_time_feat, future_time_feat, dim=1)
            if self.is_sequential
            else future_time_feat
        )

        input_feat = self.assemble_features(F, feat_static_cat, time_feat)

        outputs = self.model(input_feat)

        if self.is_sequential:
            outputs = F.slice_axis(
                outputs, axis=1, begin=-self.prediction_len, end=None
            )

        distr = self.distr_output.distribution(
            self.proj_distr_args(outputs), scale=target_scale
        )
        save_distr_params(distr.base_distribution, self.count, self.distr_output_type, self.alpha, "cCanonicalRNN")
        print("test")

        samples = distr.sample(
            self.num_parallel_samples
        )  # (num_samples, batch_size, prediction_length, 1)

        return samples.swapaxes(0, 1)
