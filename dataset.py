# Master Thesis (Théo Stassen, Université de Liège) :
# "Comparison of probabilistic forecasting deep learning models in the context of renewable energy production"
#
# - Defines the class Dataset containing training and testing sets and metadata, for an custom or imported input dataset

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas

class Dataset:

    def __init__(self, dataset=None, custom_train_dataset=None, custom_test_dataset=None,
                 start=None, freq=None, prediction_length=None,
                 learning_length=None, context_length=100, cardinality_train=None, cardinality_test=None,
                 train_static_feat=None, test_static_feat=None, num_series=None):
        if dataset is not None:
            self.learning_length = len(to_pandas(next(iter(dataset.train))))
            self.prediction_length = dataset.metadata.prediction_length
            self.freq = dataset.metadata.freq
            self.test_ds = dataset.test
            self.train_ds = dataset.train
            self.context_length = context_length
            self.cardinality = list([1])
        elif custom_train_dataset is not None:
            self.freq = freq
            self.start = start
            self.learning_length = learning_length
            self.prediction_length = prediction_length
            self.context_length = context_length
            self.cardinality_train = cardinality_train
            self.cardinality_test = cardinality_test
            self.num_series = num_series
            self.train_ds = ListDataset([{FieldName.TARGET: target,
                                          FieldName.START: start,
                                          FieldName.FEAT_STATIC_CAT: [fsc]}
                                         for (target, fsc) in zip(custom_train_dataset[:,
                                                                  :-prediction_length],
                                                                  train_static_feat)],
                                        freq=freq)
            self.test_ds = ListDataset([{FieldName.TARGET: target,
                                         FieldName.START: start,
                                         FieldName.FEAT_STATIC_CAT: [fsc]}
                                        for (target, fsc) in zip(custom_test_dataset,
                                                                 test_static_feat)],
                                       freq=freq)
