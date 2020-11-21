# Naive Bayes Assignment
# Oscar Kosar-Kosarewicz
# opk18
# 11/20/2020

from sys import argv
import re
import numpy as np


def main(train_path, test_path):
    # train_path = 'data/NaiveBayes/breast_cancer.train.txt'
    # test_path = 'data/NaiveBayes/breast_cancer.test.txt'

    # train_path = 'data/NaiveBayes/led.train.txt'
    # test_path = 'data/NaiveBayes/led.test.txt'

    # scan files to get number of attributes and number of lines
    train_attributes, train_lines = scan_file(train_path)
    test_attributes, test_lines = scan_file(test_path)
    num_attributes = max(train_attributes, test_attributes)

    # read data from files
    train_labels, train_x = read_data(train_path, num_attributes, train_lines)
    test_labels, test_x = read_data(test_path, num_attributes, test_lines)

    # Train model
    class_weights = train(train_x, train_labels)

    # predict labels
    predicted_labels = predict(train_x, class_weights)
    predicted_labels_test = predict(test_x, class_weights)

    # print results
    print_summary(train_labels, predicted_labels)
    print_summary(test_labels, predicted_labels_test)


def scan_file(filepath):
    """
    Scan the file to find the number of lines and attributes
    :param filepath: input filepath
    :return: number of attributes, number of lines
    """
    max_index = 0
    num_lines = 0
    with open(filepath) as f:
        for line in f.readlines():
            max_index_in_line = int(re.search(r'(\d*):\d*$', line).group(1))
            max_index = max(max_index, max_index_in_line)
            num_lines += 1
    return max_index, num_lines


def read_data(filepath, num_attributes, num_lines):
    """
    Read data from file
    :param filepath: input file
    :param num_attributes: number of attributes
    :param num_lines: number of lines
    :return: numpy ndarray with data
    """
    labels = np.ndarray(num_lines, dtype=np.int)
    attributes = np.zeros((num_lines, num_attributes), dtype=np.int)
    with open(filepath) as f:
        for i, line in enumerate(f.readlines()):
            labels[i] = int(re.match(r'[-+]\d*', line).group())
            for index, value in re.findall(r'(\d*):(\d*)', line):
                attributes[i, int(index) - 1] = int(value)
    return labels, attributes


def train(data, labels):
    """
    Trains a Naive Bayes classifier using Laplace smoothing.

    :param data: training data
    :param labels: training labels
    :return: A 2 dimensional list containing dictionaries of probabilities for each
    attribute for each class. shape is num_classes x num_attributes.
    """
    class_weights = []
    # loop over classes/labels
    for label in np.unique(labels):
        label_count = np.sum(labels == label) + 2
        attribute_weights = []
        # loop over attributes
        for i in range(data.shape[1]):
            # get the unique values and respective counts for the attribute in rows with this label
            values, counts = np.unique(data[labels == label, i], return_counts=True)
            # Append dictionary with label probabilities for each value of this attribute
            attribute_weights.append({value: (count + 1) / label_count for value, count in zip(values, counts)})
        class_weights.append(attribute_weights)
    return class_weights


def predict(data, class_weights):
    """
    Use the class weights of the Naive Bayes classifier to predict labels.

    :param data: data for prediction
    :param class_weights: A 2 dimensional list containing dictionaries of probabilities for each
    attribute for each class. shape is num_classes x num_attributes.
    :return: predicted labels
    """
    probabilities = np.ndarray((data.shape[0], len(class_weights)))
    for i, row in enumerate(data):
        for r, label_weights in enumerate(class_weights):
            probabilities[i, r] = np.product([label_weights[j].get(value, 0) for j, value in enumerate(row)])
    result = np.argmax(probabilities, axis=1)
    result[result == 0] = -1
    return result


def print_summary(true_y, predicted_y):
    """
    Print descriptive summary of Naive Bayes performance

    :param true_y: True labels from dataset
    :param predicted_y: predicted labels from model
    :return:
    """
    matches = predicted_y[true_y == predicted_y]
    differences = predicted_y[true_y != predicted_y]
    true_positives = np.count_nonzero(matches[matches == 1])
    true_negatives = np.count_nonzero(matches[matches == -1])
    false_positives = np.count_nonzero(differences[differences == 1])
    false_negatives = np.count_nonzero(differences[differences == -1])
    print(f'{true_positives} {false_negatives} {false_positives} {true_negatives}')


if __name__ == '__main__':
    main(argv[1], argv[2])
