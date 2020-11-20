import re
import numpy as np


def scan_file(filepath):
    max_index = 0
    num_lines = 0
    with open(filepath) as f:
        for line in f.readlines():
            max_index_in_line = int(re.search(r'(\d*):\d*$', line).group(1))
            max_index = max(max_index, max_index_in_line)
            num_lines += 1
    return max_index, num_lines


def read_data(filepath, num_attributes, num_lines):
    labels = np.ndarray(num_lines, dtype=np.int)
    attributes = np.zeros((num_lines, num_attributes), dtype=np.int)
    with open(filepath) as f:
        for i, line in enumerate(f.readlines()):
            labels[i] = int(re.match(r'[-+]\d*', line).group())
            for index, value in re.findall(r'(\d*):(\d*)', line):
                attributes[i, int(index) - 1] = int(value)
    return labels, attributes


train_path = 'data/NaiveBayes/breast_cancer.train.txt'
test_path = 'data/NaiveBayes/breast_cancer.test.txt'

train_attributes, train_lines = scan_file(train_path)
test_attributes, test_lines = scan_file(test_path)
num_attributes = max(train_attributes, test_attributes)

labels_train, attributes_train = read_data(train_path, num_attributes, train_lines)
labels_test, attributes_test = read_data(test_path, num_attributes, test_lines)
