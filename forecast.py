# Master Thesis (Théo Stassen, Université de Liège) :
# "Comparison of probabilistic forecasting deep learning models in the context of renewable energy production"
#
# - Function taking all informations about the dataset and model characteristics, create an GluonTS estimator,
#   Train it if necessary, creates a GluonTS predictor, make predictions and send the forecast.

import mxnet as mx
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model import canonical, deepar, deep_factor, deepstate, gp_forecaster, npts, seq2seq, transformer, prophet, \
    r_forecast, seasonal_naive, wavenet, n_beats
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from pathlib import Path
from gluonts.distribution.gaussian import GaussianOutput
from gluonts.distribution.laplace import LaplaceOutput
from gluonts.distribution.student_t import StudentTOutput
from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.distribution.uniform import UniformOutput
from gluonts.model.predictor import Predictor
from gluonts.trainer import Trainer

from custom_models.CustomDeepFactorEstimator import CustomDeepFactorEstimator
from custom_models.CustomSimpleEstimator import CustomSimpleEstimator
from custom_models.CustomSimpleFeedFordwardEstimator import CustomSimpleFeedForwardEstimator
from custom_models.CustomCanonicalEstimator import CustomCanonicalRNNEstimator
from custom_models.CustomDeepArEstimator import CustomDeepAREstimator
from imports import import_dataset

mx.random.seed(0)


def forecast_dataset(datasize=None,
                     config="a",
                     goal=None,
                     epochs=None,
                     learning_rate=None,
                     num_samples=100,
                     model="SimpleFeedForward",
                     r_method="ets",
                     alpha=None,
                     distrib=None,
                     num_pieces=2,
                     quantiles=list([0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995]),
                     num_cells_simple=100,
                     num_hidden_dimensions=None,
                     num_cells_rnn=20,
                     num_cells_ar=None,
                     num_layers_rnn=10,
                     num_layers_ar=3,
                     embedding_dimension=10,
                     num_hidden_global=10,
                     num_layers_global=2,
                     num_factors=50,
                     mlp_final_dim_c=40,
                     mlp_final_dim_r=50,
                     n_residue=24,
                     n_skip=32,
                     num_stacks=40,
                     model_dim=16,
                     mlp_hidden_c=[30],
                     mlp_hidden_r=[20],
                     num_heads=8,
                     n_blocks=[1],
                     n_stacks=1,
                     use_static=False,
                     context_length=None
                     ):
    """
    Function taking all informations about the dataset and model characteristics, create an GluonTS estimator,
    Train it if necessary, creates a GluonTS predictor, make predictions and send the predicted forecast.
    :param datasize: size of the dataset time series
    :param config: "A or "B", indicating the dataset configuration
    :param goal: "1" or "2" indicating the goal that is pursued in terms of metric to optimize
    :param epochs: Number of epochs to train
    :param learning_rate: Learning rate of the training
    :param num_samples: number of samples of the predicted distribution
    :param model: model selected
    :param r_method: method used for R package
    :param alpha: alpha parameter value
    :param distrib: output distribution chosen
    :param num_pieces: number of pieces of the PiecewiseLinear distribution
    :param quantiles: vector of evaluated quantiles
    :param num_cells_simple: num cells parameter of model Simple
    :param num_hidden_dimensions: num hidden dimensions parameter of model SimpleFeedForward
    :param num_cells_rnn: num cells parameter of model CanonicalRNN
    :param num_cells_ar: num cells parameter of model DeepAr
    :param num_layers_rnn: num layers parameter of model CanonicalRNN
    :param num_layers_ar: num layers parameter of model DeepAr
    :param embedding_dimension: embedding dimensions parameter of model CanonicalRNN
    :param num_hidden_global: num hidden global parameter of model DeepFactor
    :param num_layers_global: num layer global parameter of model DeepFactor
    :param num_factors: num factors parameter of model DeepFactor
    :param mlp_final_dim_c: mlp final dim parameter of model MQCNN
    :param mlp_final_dim_r: mlp final dim parameter of model MQRNN
    :param n_residue: num residue parameter of model Wavenet
    :param n_skip: num skip parameter of model Wavenet
    :param num_stacks: num stacks parameter of model Nbeats
    :param model_dim: model dim parameter of model Transformer
    :param mlp_hidden_c: mlp hidden parameter of model MQCNN
    :param mlp_hidden_r: mlp hidden parameter of model MQRNN
    :param num_heads: num heads parameter of model Transformer
    :param n_blocks: num cells parameter of model Nbeats
    :param n_stacks: num cells parameter of model Wavenet
    :param use_static: whether use static features or not
    :param context_length: context lenght parameter value
    :return: Dataset object, the list of forecast and the list of complete time series
    """

    # Datasize parameter default value depends on the model

    if model == "Transformer":
        if datasize is None:
            datasize = 1440
    else:
        if datasize is None:
            datasize = 2880

    # Dataset shape depends on the configuration
    if config == "a":
        dataset = import_dataset(["6months-minutes", "2eol_measurements", "mesure_p_gestamp"],
                                 ["6months-minutes", "2eol_measurements", "mesure_p_gestamp"], datasize)
    elif config == "b":
        dataset = import_dataset(["6months-minutes", "mesure_p_gestamp"],
                                 ["2eol_measurements"], datasize, incr_test=True)
    elif config == "c":
        dataset = import_dataset(["6months-minutes", "mesure_p_gestamp"],
                                 ["6months-minutes", "mesure_p_gestamp"], datasize)
    else:
        dataset = None

    # num cells ar and num hidden dimensions parameter default value depends on the model
    if num_cells_ar is None:
        num_cells_ar = 100 if goal == "1" else 80
    if num_hidden_dimensions is None:
        num_hidden_dimensions = [100] if goal == "1" else [40,100]

    # learning rate parameter default value depends on the model
    if learning_rate is None:
        if model == "cSimpleFeedForward" \
                or model == "Nbeats" or model == "MQRNN" or model == "DeepAr" or model == "cDeepAr" or model == "Wavenet":
            learning_rate = 1e-3
        elif model == "DeepFactor":
            learning_rate = 1e-5
        elif model == "GaussianProcess"  :
            learning_rate = 1e-2
        elif model == "MQCNN" or model == "cCanonicalRNN" :
            learning_rate = 1e-4
        else:
            learning_rate = 1e-3

    # epochs parameter default value depends on the model and the goal pursued
    if goal == "1":
        if epochs is None:
            if model == "cSimpleFeedForward" or model == "DeepAr" or model == "cDeepAr" :
                epochs = 120
            elif model == "cCanonicalRNN":
                epochs = 20
            elif model == "DeepFactor" or model == "Transformer":
                epochs = 100
            elif model == "GaussianProcess" or model == "MQCNN" :
                epochs = 50
            elif model == "Nbeats" or model == "Wavenet" or model == "MQRNN":
                epochs = 120
            else:
                epochs = 100
    else:
        if epochs is None:
            if model == "cCanonicalRNN" :
                epochs = 20
            elif model == "DeepFactor" or model == "Transformer" or "GaussianProcess" \
                    or model == "Nbeats":
                epochs = 100
            elif model == "MQCNN":
                epochs = 50
            elif model == "MQRNN" or model == "Wavenet" or model == "cSimpleFeedForward" or model == "DeepAr" or model == "cDeepAr":
                epochs = 120
            else:
                epochs = 100

    # Output distribution parameter default value depends on the model and the goal pursued
    if goal == "1":
        if distrib is None:
            distrib = "PiecewiseLinear"
    else:
        if distrib is None:
            if model == "cSimpleFeedForward":
                distrib = "Gaussian"
            elif model == "DeepAr":
                distrib = "Laplace"
            else:
                distrib = "PiecewiseLinear"

    # Context length parameter default value depends on the model and the goal pursued
    if goal == "1":
        if context_length is None:
            if model == "DeepAr" or model == "cDeepAr" or model == "Nbeats":
                context_length = 120
            elif model == "MQCNN" or model == "cSimpleFeedForward":
                context_length = 30
            elif model == "MQRNN":
                context_length = 100
            else:
                context_length = 60
    else:
        if context_length is None:
            if context_length is None:
                if model == "DeepAr" or model == "cDeepAr" or model == "Nbeats":
                    context_length = 120
                elif model == "MQCNN" or model == "cSimpleFeedForward":
                    context_length = 30
                elif model == "MQRNN":
                    context_length = 100
                else:
                    context_length = 60

    # alpha parameter default value depends on the model  and the goal pursued
    if goal == "1":
        if alpha is None:
            alpha = 0
    else:
        if alpha is None:
            if model == "cSimpleFeedForward":
                alpha = 0
            else:
                alpha = 2

    # Declare distribution output GluonTS object
    if distrib == "Gaussian":
        distr_output = GaussianOutput()
    elif distrib == "Laplace":
        distr_output = LaplaceOutput()
    elif distrib == "PiecewiseLinear":
        distr_output = PiecewiseLinearOutput(num_pieces=num_pieces)
    elif distrib == "Uniform":
        distr_output = UniformOutput()
    elif distrib == "Student":
        distr_output = StudentTOutput()
    else:
        distr_output = None

    # Define the context of execution
    if model != "GaussianProcess":
        ctx = mx.Context("gpu")
    else:
        ctx = mx.Context("cpu")

    # Define the Trainer GluonTS object
    trainer = Trainer(epochs=epochs,
                      learning_rate=learning_rate,
                      num_batches_per_epoch=100,
                      ctx=ctx,
                      hybridize=True if model[0] != "c" else False
                      )

    # Define the Estimator GluonTS object (if the model need to be trained)
    if model == "cSimple":
        estimator = CustomSimpleEstimator(
            prediction_length=dataset.prediction_length,
            context_length=context_length,
            freq=dataset.freq,
            trainer=trainer,
            alpha=alpha,
            distr_output=distr_output,
            # distr_output_type=distrib,
            num_cells=num_cells_simple
        )
    elif model == "SimpleFeedForward":
        estimator = SimpleFeedForwardEstimator(
            prediction_length=dataset.prediction_length,
            context_length=context_length,
            freq=dataset.freq,
            trainer=trainer,
            distr_output=distr_output,
            num_hidden_dimensions=num_hidden_dimensions,
        )
    elif model == "cSimpleFeedForward":
        estimator = CustomSimpleFeedForwardEstimator(
            prediction_length=dataset.prediction_length,
            context_length=context_length,
            freq=dataset.freq,
            trainer=trainer,
            alpha=alpha,
            distr_output=distr_output,
            distr_output_type=distrib,
            num_hidden_dimensions=num_hidden_dimensions,
        )
    elif model == "CanonicalRNN":
        estimator = canonical.CanonicalRNNEstimator(
            freq=dataset.freq,
            context_length=context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            distr_output=distr_output
        )
    elif model == "cCanonicalRNN":
        estimator = CustomCanonicalRNNEstimator(
            freq=dataset.freq,
            context_length=context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            distr_output=distr_output,
            distr_output_type=distrib,
            alpha=alpha,
            num_layers=num_layers_rnn,
            num_cells=num_cells_rnn,
            embedding_dimension=embedding_dimension
        )
    elif model == "DeepAr":
        estimator = deepar.DeepAREstimator(
            freq=dataset.freq,
            context_length=context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            distr_output=distr_output,
            num_cells=num_cells_ar,
            num_layers=num_layers_ar,
            use_feat_static_cat=use_static,
            cardinality=dataset.cardinality_train if use_static else None,
        )
    elif model == "cDeepAr":
        estimator = CustomDeepAREstimator(
            freq=dataset.freq,
            context_length=context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            distr_output=distr_output,
            num_cells=num_cells_ar,
            num_layers=num_layers_ar,
            use_feat_static_cat=use_static,
            cardinality=dataset.cardinality_train,
        )
    elif model == "DeepFactor":
        estimator = deep_factor.DeepFactorEstimator(
            freq=dataset.freq,
            context_length=context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            distr_output=distr_output,
            num_hidden_global=num_hidden_global,
            num_layers_global=num_layers_global,
            num_factors=num_factors
        )
    elif model == "cDeepFactor":
        estimator = CustomDeepFactorEstimator(
            freq=dataset.freq,
            context_length=context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            distr_output=distr_output,
            num_hidden_global=num_hidden_global,
            num_layers_global=num_layers_global,
            num_factors=num_factors,
        )
    elif model == "DeepState":  # Not used because make the computer freeze
        estimator = deepstate.DeepStateEstimator(
            freq=dataset.freq,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            cardinality=dataset.cardinality_train,
            use_feat_static_cat=use_static
        )
    elif model == "GaussianProcess":
        estimator = gp_forecaster.GaussianProcessEstimator(
            freq=dataset.freq,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            cardinality=183,
        )
    elif model == "NPTS":
        estimator = npts.NPTSEstimator(
            freq=dataset.freq,
            prediction_length=dataset.prediction_length
        )
    elif model == "MQCNN":
        estimator = seq2seq.MQCNNEstimator(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=context_length,
            trainer=trainer,
            quantiles=quantiles,
            mlp_final_dim=mlp_final_dim_c,
            mlp_hidden_dimension_seq=mlp_hidden_c
        )
    elif model == "MQRNN":
        estimator = seq2seq.MQRNNEstimator(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=context_length,
            trainer=trainer,
            quantiles=quantiles,
            mlp_final_dim=mlp_final_dim_r,
            mlp_hidden_dimension_seq=mlp_hidden_r
        )
    elif model == "RNN2QR":  # Not used
        estimator = seq2seq.RNN2QRForecaster(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=context_length,
            trainer=trainer,
            cardinality=dataset.cardinality,
            embedding_dimension=1,
            encoder_rnn_layer=1,
            encoder_rnn_num_hidden=1,
            decoder_mlp_layer=[1],
            decoder_mlp_static_dim=1
        )
    elif model == "SeqToSeq":  # Not used
        estimator = seq2seq.Seq2SeqEstimator(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=context_length,
            trainer=trainer,
            cardinality=[1],
            embedding_dimension=1,
            decoder_mlp_layer=[1],
            decoder_mlp_static_dim=1,
            # encoder=Seq2SeqEncoder()
        )
    elif model == "Transformer":
        estimator = transformer.TransformerEstimator(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=context_length,
            trainer=trainer,
            distr_output=distr_output,
            model_dim=model_dim,
            num_heads=num_heads
        )
    elif model == "Wavenet":
        estimator = wavenet.WaveNetEstimator(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            trainer=trainer,
            n_residue=n_residue,
            n_skip=n_skip,
            n_stacks=n_stacks
        )
    elif model == "Nbeats":
        estimator = n_beats.NBEATSEstimator(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=context_length,
            trainer=trainer,
            num_stacks=num_stacks,
            num_blocks=n_blocks
        )

    else:
        estimator = None

    # Define the Predictor GluonTS object (by trainin the estimator or directly if not to train model)
    if model == "Prophet":
        predictor = prophet.ProphetPredictor(
            freq=dataset.freq,
            prediction_length=dataset.prediction_length
        )
    elif model == "R":
        predictor = r_forecast.RForecastPredictor(
            freq=dataset.freq,
            prediction_length=dataset.prediction_length,
            method_name=r_method
        )
    elif model == "SeasonalNaive":
        predictor = seasonal_naive.SeasonalNaivePredictor(
            freq=dataset.freq,
            prediction_length=dataset.prediction_length,
            season_length=24
        )
    else:
        predictor = estimator.train(dataset.train_ds)
        if model[0] == "DeepState":
            predictor.serialize(Path("temp"))
            predictor = Predictor.deserialize(Path("temp"), ctx=ctx)  # fix for deepstate

    # Make predictions on the test dataset, using the Predictor
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=num_samples,  # num of sample paths we want for evaluation
    )

    return dataset, list(forecast_it), list(ts_it)
