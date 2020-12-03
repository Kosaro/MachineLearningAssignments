# Naive Bayes Assignment
# I initially wrote and polished this assignment using numpy, but when I tested it on linprog
# a few hours before the due date I was surprised
# to find that numpy was not installed. (It is for python 2, but not python 3)
# So I hastily rewrote it to work without numpy. I apologize for messy code in this file, I also submitted the numpy
# file so that you can view the clean version
# Oscar Kosar-Kosarewicz
# opk18
# 11/20/2020

from sys import argv
import re
from collections import Counter


def main(train_path, test_path):

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
    :return: data
    """
    labels = []
    attributes = initialize_matrix(num_lines, num_attributes)
    with open(filepath) as f:
        for i, line in enumerate(f.readlines()):
            labels.append(int(re.match(r'[-+]\d*', line).group()))
            for index, value in re.findall(r'(\d*):(\d*)', line):
                attributes[i][int(index) - 1] = int(value)
    return labels, attributes


def train(data, labels):
    """
    Trains a Naive Bayes classifier using Laplace smoothing.

    :param data: training data
    :param labels: training labels
    :return: A 2 dimensional list containing dictionaries of probabilities for each
    attribute for each class. shape is num_classes x num_attributes.
    """
    counter = Counter(labels)
    class_weights = []
    # loop over classes/labels
    for label, num_labels in counter.items():
        num_labels += 2
        for l in labels:
            if l == labels:
                l += 1
        attribute_weights = []
        # loop over attributes
        for i in range(len(data[0])):
            counter = Counter()
            for j in range(len(labels)):
                if labels[j] == label:
                    counter[data[j][i]] += 1 / num_labels
            attribute_weights.append(counter)
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
    probabilities = initialize_matrix(len(data), len(class_weights))
    for i, row in enumerate(data):
        for r, label_weights in enumerate(class_weights):
            probabilities[i][r] = product([label_weights[j].get(value, 0) for j, value in enumerate(row)])
    result = []
    for l,r in probabilities:
        if l < r:
            result.append(-1)
        else:
            result.append(1)
    return result

def product(list):
    result = 1
    for x in list:
        result *= x
    return result


def print_summary(true_y, predicted_y):
    """
    Print descriptive summary of Naive Bayes performance

    :param true_y: True labels from dataset
    :param predicted_y: predicted labels from model
    :return:
    """
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for true, predicted in zip(true_y, predicted_y):
        if predicted==1:
            if true == predicted:
                true_positives += 1
            else:
                false_positives +=1
        else:
            if true == predicted:
                true_negatives += 1
            else:
                false_negatives +=1

    print('%4s %4s %4s %4s' % (true_positives, false_negatives, false_positives, true_negatives))


def initialize_matrix(x,y):
    result = []
    for i in range(x):
        arr = []
        for j in range(y):
            arr.append(0)
        result.append(arr)
    return result

if __name__ == '__main__':
    main(argv[1], argv[2])
