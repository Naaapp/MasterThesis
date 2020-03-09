from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas


class Dataset:

    def __init__(self, dataset=None, custom_dataset=None,
                 start=None, freq=None, prediction_length=None,
                 learning_length=None, context_length=100, cardinality=None):
        if dataset is not None:
            self.learning_length = len(to_pandas(next(iter(dataset.train))))
            self.prediction_length = dataset.metadata.prediction_length
            self.freq = dataset.metadata.freq
            self.test_ds = dataset.test
            self.train_ds = dataset.train
            self.context_length = context_length
            self.cardinality = list([1])
        elif custom_dataset is not None:
            self.freq = freq
            self.start = start
            self.learning_length = learning_length
            self.prediction_length = prediction_length
            self.context_length = context_length
            self.cardinality = cardinality
            # train dataset: cut the last window of length "prediction_length",
            # add "target" and "start" fields
            self.train_ds = ListDataset([{'target': x, 'start': start}
                                         for x in
                                         custom_dataset[:,
                                         :-prediction_length]],
                                        freq=freq)
            # test dataset: use the whole dataset, add "target" and "start"
            self.test_ds = ListDataset([{'target': x, 'start': start}
                                        for x in custom_dataset],
                                       freq=freq)
