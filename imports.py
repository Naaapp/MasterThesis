import pandas as pd
import numpy as np
import dataset as dt


def import_p_gestamp(data_size, imported_dataset, static_eol_type, cardinality):
    train_df = pd.read_csv("datasets/mesure_p_gestamp.csv")
    a_imported_dataset = -np.array([train_df['Active Power ' + str(1)].to_numpy()])[0][0:97920]
    n_sample = np.shape(a_imported_dataset)[0] / data_size
    a_imported_dataset = a_imported_dataset.reshape(int(n_sample), -1)
    imported_dataset = np.concatenate((imported_dataset, a_imported_dataset), axis=0)
    static_eol_type = np.append(static_eol_type, np.zeros(int(n_sample)))
    cardinality = cardinality + [n_sample]
    return imported_dataset, static_eol_type, cardinality


def import_6months_minutes(data_size, imported_dataset, static_eol_type, cardinality):
    train_df = pd.read_csv("datasets/6months-minutes.csv")
    b_imported_dataset = np.array([train_df['Active Power ' + str(1)].to_numpy()])[0][1::2]
    n_sample = np.shape(b_imported_dataset)[0] / data_size
    b_imported_dataset = b_imported_dataset / 1000
    b_imported_dataset = b_imported_dataset.reshape(int(n_sample), -1)
    imported_dataset = np.concatenate((imported_dataset, b_imported_dataset), axis=0)
    static_eol_type = np.append(static_eol_type, np.ones(int(n_sample)))
    cardinality = cardinality + [n_sample]
    return imported_dataset, static_eol_type, cardinality


def import_2eol_measurements(data_size, imported_dataset, static_eol_type, cardinality):
    train_df = pd.read_csv("datasets/2eol_measurements.csv")
    c_imported_dataset = -np.array([train_df['Active Power ' + str(1)].to_numpy()])[0]
    d_imported_dataset = -np.array([train_df['Active Power ' + str(2)].to_numpy()])[0]
    n_sample = np.shape(c_imported_dataset)[0] / data_size

    c_imported_dataset = c_imported_dataset.reshape(int(n_sample), -1)
    imported_dataset = np.concatenate((imported_dataset, c_imported_dataset), axis=0)
    static_eol_type = np.append(static_eol_type, np.ones(int(n_sample)) + 1)

    d_imported_dataset = d_imported_dataset.reshape(int(n_sample), -1)
    imported_dataset = np.concatenate((imported_dataset, d_imported_dataset), axis=0)
    static_eol_type = np.append(static_eol_type, np.array(np.ones(int(n_sample)) + 2))
    cardinality = cardinality + [n_sample]
    return imported_dataset, static_eol_type, cardinality


def import_dataset(train_dataset_names, test_dataset_names, data_size=1440):
    imported_train_dataset = np.empty((0, data_size), int)
    static_feat_train = np.empty(0, int)
    imported_test_dataset = np.empty((0, data_size), int)
    static_feat_test = np.empty(0, int)
    cardinality_train = []
    cardinality_test = []

    if "mesure_p_gestamp" in train_dataset_names:
        imported_train_dataset, static_feat_train, cardinality_train = import_p_gestamp(
            data_size, imported_train_dataset, static_feat_train, cardinality_train)
    if "6months-minutes" in train_dataset_names:
        imported_train_dataset, static_feat_train, cardinality_train = import_6months_minutes(
            data_size, imported_train_dataset, static_feat_train, cardinality_train)
    if "2eol_measurements" in train_dataset_names:
        imported_train_dataset, static_feat_train, cardinality_train = import_2eol_measurements(
            data_size, imported_train_dataset, static_feat_train, cardinality_train)

    if "mesure_p_gestamp" in test_dataset_names:
        imported_test_dataset, static_feat_test, cardinality_test = import_p_gestamp(
            data_size, imported_test_dataset, static_feat_test, cardinality_test)
    if "6months-minutes" in test_dataset_names:
        imported_test_dataset, static_feat_test, cardinality_test = import_6months_minutes(
            data_size, imported_test_dataset, static_feat_test, cardinality_test)
    if "2eol_measurements" in test_dataset_names:
        imported_test_dataset, static_feat_test, cardinality_test = import_2eol_measurements(
            data_size, imported_test_dataset, static_feat_test, cardinality_test)

    # print(imported_train_dataset)
    # print(imported_test_dataset)
    print(cardinality_train)

    prediction_length = 10
    context_length = 60 * 1  # One day
    freq = "1min"
    start = pd.Timestamp("01-04-2019", freq=freq)
    return dt.Dataset(custom_train_dataset=imported_train_dataset,
                      custom_test_dataset=imported_test_dataset,
                      train_static_feat=static_feat_train,
                      test_static_feat=static_feat_test,
                      cardinality_train=cardinality_train,
                      cardinality_test=cardinality_test,
                      start=start,
                      freq=freq,
                      prediction_length=prediction_length,
                      learning_length=context_length)
