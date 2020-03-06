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


def forecast_dataset(dataset, epochs=5, learning_rate=1e-3, num_samples=100,
                     model="SimpleFeedForward", r_method="ets"):
    if model != "GaussianProcess":
        ctx = mx.Context("gpu")
    else:
        ctx = mx.Context("cpu")

    # Trainer
    trainer = Trainer(
                      learning_rate=learning_rate,
                      num_batches_per_epoch=100,
                      ctx=ctx)

    # Estimator (if machine learning model)
    if model == "SimpleFeedForward":  # 10s / epochs for context 60*24
        estimator = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[10],
            prediction_length=dataset.prediction_length,
            context_length=dataset.context_length,
            freq=dataset.freq,
            trainer=trainer
        )
    elif model == "CanonicalRNN":  # 80s /epochs for context 60*24, idem for 60*1
        estimator = canonical.CanonicalRNNEstimator(
            freq=dataset.freq,
            context_length=dataset.context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
        )
    elif model == "DeepAr":
        estimator = deepar.DeepAREstimator(
            freq=dataset.freq,
            context_length=dataset.context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer
        )
    elif model == "DeepFactor":
        estimator = deep_factor.DeepFactorEstimator(
            freq=dataset.freq,
            context_length=dataset.context_length,
            prediction_length=dataset.prediction_length,
            trainer=trainer
        )
    elif model == "DeepState":  # Very slow on cpu
        estimator = deepstate.DeepStateEstimator(
            freq=dataset.freq,
            prediction_length=dataset.prediction_length,
            trainer=trainer,
            cardinality=list([1]),
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
            prediction_length=dataset.prediction_length,
            ctx=ctx
        )
    elif model == "MQCNN":
        estimator = seq2seq.MQCNNEstimator(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=dataset.context_length,
            trainer=trainer
        )
    elif model == "MQRNN":
        estimator = seq2seq.MQRNNEstimator(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=dataset.context_length,
            trainer=trainer
        )
    elif model == "RNN2QR":  # Must be investigated
        estimator = seq2seq.RNN2QRForecaster(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            context_length=dataset.context_length,
            trainer=trainer,
            cardinality=[1],
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
        predictor.serialize(Path("temp"))
        predictor = Predictor.deserialize(Path("temp"), ctx=mx.cpu(0))  # fix for deepstate

    # Evaluate
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=num_samples,  # num of sample paths we want for evaluation
    )

    return list(forecast_it), list(ts_it)
