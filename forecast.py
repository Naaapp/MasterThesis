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


def forecast_dataset(dataset, epochs=100, learning_rate=1e-3, num_samples=100,
                     model="SimpleFeedForward", r_method="ets", alpha=0, distrib="Gaussian",
                     quantiles=list([0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995])):
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
    if model == "SimpleFeedForward":  # 10s / epochs for context 60*24
        estimator = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[10],
            prediction_length=dataset.prediction_length,
            context_length=dataset.context_length,
            freq=dataset.freq,
            trainer=trainer,
            distr_output=distr_output,
        )
    elif model == "cSimple":  # 10s / epochs for context 60*24
        estimator = CustomSimpleEstimator(
            prediction_length=dataset.prediction_length,
            context_length=dataset.context_length,
            freq=dataset.freq,
            trainer=trainer,
            num_cells=40,
            alpha=alpha,
            distr_output=distr_output,
            distr_output_type=distrib
        )
    elif model == "cSimpleFeedForward":  # 10s / epochs for context 60*24
        estimator = CustomSimpleFeedForwardEstimator(
            prediction_length=dataset.prediction_length,
            context_length=dataset.context_length,
            freq=dataset.freq,
            trainer=trainer,
            num_hidden_dimensions=[10],
            alpha=alpha,
            distr_output=distr_output,
            distr_output_type=distrib
        )
    elif model == "CanonicalRNN":  # 80s /epochs for context 60*24, idem for 60*1
        estimator = canonical.CanonicalRNNEstimator(
            freq=dataset.freq,
            context_length=dataset.context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            distr_output=distr_output,
        )
    elif model == "DeepAr":
        estimator = deepar.DeepAREstimator(
            freq=dataset.freq,
            context_length=dataset.context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            distr_output=distr_output,
        )
    elif model == "DeepFactor":  # 120 s/epochs if one big time serie, 1.5s if 183 time series
        estimator = deep_factor.DeepFactorEstimator(
            freq=dataset.freq,
            context_length=dataset.context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            distr_output=distr_output,
        )
    elif model == "DeepState":  # Very slow on cpu
        estimator = deepstate.DeepStateEstimator(
            freq=dataset.freq,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            cardinality=list([1]),
            use_feat_static_cat=False
        )
    elif model == "GaussianProcess":  # CPU / GPU problem
        estimator = gp_forecaster.GaussianProcessEstimator(
            freq=dataset.freq,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            cardinality=1
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
            quantiles=quantiles
        )
    elif model == "MQRNN":
        estimator = seq2seq.MQRNNEstimator(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=dataset.context_length,
            trainer=trainer,
            quantiles=quantiles
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
