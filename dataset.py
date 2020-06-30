from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas


class Dataset:

    def __init__(self, dataset=None, custom_train_dataset=None, custom_test_dataset=None,
                 start=None, freq=None, prediction_length=None,
                 learning_length=None, context_length=100, cardinality_train=None, cardinality_test=None,
                 train_static_feat=None, test_static_feat=None):
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
            # train dataset: cut the last window of length "prediction_length",
            # add "target" and "start" fields
            # self.train_ds = ListDataset([{'target': x, 'start': start}
            #                              for x in
            #                              custom_train_dataset[:,
            #                              :-prediction_length]],
            #                             freq=freq)
            self.train_ds = ListDataset([{FieldName.TARGET: target,
                                          FieldName.START: start,
                                          FieldName.FEAT_STATIC_CAT: [fsc]}
                                         for (target, fsc) in zip(custom_train_dataset[:,
                                                                  :-prediction_length],
                                                                  train_static_feat)],
                                        freq=freq)
            # test dataset: use the whole dataset, add "target" and "start"
            self.test_ds = ListDataset([{FieldName.TARGET: target,
                                         FieldName.START: start,
                                         FieldName.FEAT_STATIC_CAT: [fsc]}
                                        for (target, fsc) in zip(custom_test_dataset,
                                                                 test_static_feat)],
                                       freq=freq)
