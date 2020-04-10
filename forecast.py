from gluonts.distribution import GaussianOutput, LaplaceOutput, PiecewiseLinearOutput, UniformOutput, \
    StudentTOutput
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model import deepar, canonical, deep_factor, deepstate, gp_forecaster, \
    npts, prophet, r_forecast, seasonal_naive, seq2seq, \
    transformer
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.model.predictor import Predictor
import matplotlib.pyplot as plt
import mxnet as mx
from path import Path
from gluonts.block.encoder import Seq2SeqEncoder
from mxnet import nd, gpu, gluon, autograd
from custom_models.CustomSimpleEstimator import CustomSimpleEstimator
from custom_models.CustomSimpleFeedFordwardEstimator import CustomSimpleFeedForwardEstimator
from custom_models.CustomCanonicalEstimator import CustomCanonicalRNNEstimator


def forecast_dataset(dataset,
                     epochs=100,
                     learning_rate=1e-3,
                     num_samples=100,
                     model="SimpleFeedForward",
                     r_method="ets",
                     alpha=0,
                     distrib="Gaussian",
                     quantiles=list([0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995]),
                     num_cells_simple=100,
                     num_hidden_dimensions=[10],
                     num_cells_rnn=50,
                     num_cells_ar=40,
                     num_layers_rnn=1,
                     num_layers_ar=2,
                     embedding_dimension=10,
                     num_hidden_global=50,
                     num_layers_global=1,
                     num_factors=10,
                     mlp_final_dim=20
                     ):

    if distrib == "Gaussian":
        distr_output = GaussianOutput()
    elif distrib == "Laplace":
        distr_output = LaplaceOutput()
    elif distrib == "PiecewiseLinear":
        distr_output = PiecewiseLinearOutput(num_pieces=2)
    elif distrib == "Uniform":
        distr_output = UniformOutput()
    elif distrib == "Student":
        distr_output = StudentTOutput()
    else:
        distr_output = None

    if model != "GaussianProcess":
        ctx = mx.Context("gpu")
    else:
        ctx = mx.Context("cpu")

    # Trainer
    trainer = Trainer(epochs=epochs,
                      learning_rate=learning_rate,
                      num_batches_per_epoch=100,
                      ctx=ctx,
                      hybridize=True if model[0] != "c" else False
                      )

    # Estimator (if machine learning model)
    if model == "cSimple":
        estimator = CustomSimpleEstimator(
            prediction_length=dataset.prediction_length,
            context_length=dataset.context_length,
            freq=dataset.freq,
            trainer=trainer,
            alpha=alpha,
            distr_output=distr_output,
            distr_output_type=distrib,
            num_cells=num_cells_simple,
        )
    elif model == "SimpleFeedForward":
        estimator = SimpleFeedForwardEstimator(
            prediction_length=dataset.prediction_length,
            context_length=dataset.context_length,
            freq=dataset.freq,
            trainer=trainer,
            distr_output=distr_output,
            num_hidden_dimensions=num_hidden_dimensions,
        )
    elif model == "cSimpleFeedForward":
        estimator = CustomSimpleFeedForwardEstimator(
            prediction_length=dataset.prediction_length,
            context_length=dataset.context_length,
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
            context_length=dataset.context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            distr_output=distr_output
        )
    elif model == "cCanonicalRNN":
        estimator = CustomCanonicalRNNEstimator(
            freq=dataset.freq,
            context_length=dataset.context_length,
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
            context_length=dataset.context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            distr_output=distr_output,
            num_cells=num_cells_ar,
            num_layers=num_layers_ar
        )
    elif model == "DeepFactor":
        estimator = deep_factor.DeepFactorEstimator(
            freq=dataset.freq,
            context_length=dataset.context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            distr_output=distr_output,
            num_hidden_global=num_hidden_global,
            num_layers_global=num_layers_global,
            num_factors=num_factors,
        )
    elif model == "DeepState":  # Very slow on cpu
        estimator = deepstate.DeepStateEstimator(
            freq=dataset.freq,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            cardinality=list([1]),
            use_feat_static_cat=False
        )
    elif model == "GaussianProcess":
        estimator = gp_forecaster.GaussianProcessEstimator(
            freq=dataset.freq,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            cardinality=183
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
            context_length=dataset.context_length,
            trainer=trainer,
            quantiles=quantiles,
            mlp_final_dim=mlp_final_dim
        )
    elif model == "MQRNN":
        estimator = seq2seq.MQRNNEstimator(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=dataset.context_length,
            trainer=trainer,
            quantiles=quantiles,
            mlp_final_dim=mlp_final_dim
        )
    elif model == "RNN2QR":  # Must be investigated
        estimator = seq2seq.RNN2QRForecaster(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=dataset.context_length,
            trainer=trainer,
            cardinality=dataset.cardinality,
            embedding_dimension=1,
            encoder_rnn_layer=1,
            encoder_rnn_num_hidden=1,
            decoder_mlp_layer=[1],
            decoder_mlp_static_dim=1
        )
    elif model == "SeqToSeq":  # Must be investigated
        estimator = seq2seq.Seq2SeqEstimator(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=dataset.context_length,
            trainer=trainer,
            cardinality=[1],
            embedding_dimension=1,
            decoder_mlp_layer=[1],
            decoder_mlp_static_dim=1,
            encoder=Seq2SeqEncoder()
        )
    elif model == "Transformer":  # Make the computer lag the first time
        estimator = transformer.TransformerEstimator(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=dataset.context_length,
            trainer=trainer
        )

    else:
        estimator = None

    # Predictor (directly if non machine learning model and from estimator if machine learning)
    if model == "Prophet":
        predictor = prophet.ProphetPredictor(
            freq=dataset.freq,
            prediction_length=dataset.prediction_length,
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
        if model[0] != "c":
            predictor.serialize(Path("temp"))
            predictor = Predictor.deserialize(Path("temp"), ctx=mx.cpu(0))  # fix for deepstate

    # Evaluate
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=num_samples,  # num of sample paths we want for evaluation
    )

    return list(forecast_it), list(ts_it)
