# Load CSV
import numpy as np
import pandas as pd
import csv
import glob
from pathlib import Path
import re

# load training and testing data from CSV files


def load_data(path, num_of_inputs, num_of_categories):
    """==============read training data=============="""
    raw_data = open(path + "/training_data.csv", "rt")
    tr_d = np.loadtxt(raw_data, delimiter=",")
    training_inputs = [np.reshape(x, (num_of_inputs, 1)) for x in tr_d]
    raw_data = open(path + "/training_labels.csv", "rt")
    tr_l = np.loadtxt(raw_data, delimiter=",")

    # train_labels_flat = train_data.iloc[:,0:1].values
    # train_labels_count = np.unique(tr_l).shape[0]

    training_labels = [vectorization(y, num_of_categories) for y in tr_l]
    training_data = list(zip(training_inputs, training_labels))

    """==============read testing data=============="""
    raw_data = open(path + "/testing_data.csv", "rt")
    te_d = np.loadtxt(raw_data, delimiter=",")
    testing_inputs = [np.reshape(x, (num_of_inputs, 1)) for x in te_d]

    # test_labels = test_data.iloc[:,0:1].values
    # test_labels = dense_to_one_hot(test_labels, train_labels_count)
    # test_labels = test_labels.astype(np.uint8)

    test_data = pd.read_csv(path + "/testing_labels.csv", header=None)
    testing_labels = test_data.iloc[:, 0:1].values
    testing_labels = dense_to_one_hot(testing_labels, num_of_categories)
    testing_labels = testing_labels.astype(np.uint8)

    # raw_data = open(path+'/testing_labels.csv', 'rt')
    # testing_labels = np.loadtxt(raw_data, delimiter=",")
    # testing_labels = dense_to_one_hot(testing_labels, num_of_categories)

    testing_data = testing_inputs
    # testing_data = zip(testing_inputs, te_l)

    return (training_data, testing_data, testing_labels)


def combine_and_output_experiment_data(root: str, type: str) -> None:
    """Combine all experiment round CSV files in a directory into a single CSV file.

    Args:
        root (str): The directory containing the CSV files to combine.
        type (str): The type of data to combine (ground truth or prediction labels).
          Possible values are "groundtruth" and "prediction".
    """

    root = Path(root)
    if type == "groundtruth":
        csv_files = sorted(root.glob("Round*GroundTruth.csv"))
        label = "GroundTruth"
    elif type == "prediction":
        csv_files = sorted(root.glob("Round*Prediction.csv"))
        label = "Prediction"
    else:
        raise ValueError("Invalid type: " + type)

    # Create an empty list to hold the data from all CSV files
    data = []

    # Loop through all CSV files in the directory
    for filename in csv_files:
        # Open the CSV file and read the data
        with open(filename, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            data.extend(csv_reader)  # append data to list

    # Write the combined data to a new CSV file
    combined_path = root / f"{label}.csv"
    with open(combined_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)


def vectorization(j, num_of_categories):
    e = np.zeros((num_of_categories, 1))
    e[int(j)] = 1.0
    return e


# Convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
