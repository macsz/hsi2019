import argparse
import json
import sys

import numpy as np
from scipy.spatial import distance
from sklearn import svm
from sklearn.svm import LinearSVC


def main(args):
    with open(args.train_db, 'r') as outfile:
        train_db = json.load(outfile)
        print('Train DB has {} keys'.format(len(train_db.keys())))

    with open(args.test_db, 'r') as outfile:
        test_db = json.load(outfile)
        print('Test DB has {} keys'.format(len(test_db.keys())))
    print('Train and test db have {} common keys'.format(len(list(set(test_db.keys()).intersection(train_db.keys())))))

    Y = []
    X = []
    all = 0
    valid = 0
    invalid = 0
    categories = []
    for train_cat in train_db.keys():
        for embedding in train_db[train_cat]:
            if train_cat not in categories:
                categories.append(train_cat)
            Y.append(train_cat)
            X.append(embedding)
            # X.append(train_db[train_cat][embedding])
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X, Y)

    for test_cat in test_db.keys():
        for embedding in test_db[test_cat]:
            dec = lin_clf.decision_function([embedding])
            result = categories[np.where(dec == dec.max())[1][0]]
            print('Testing', test_cat, result)
            all += 1
            if result == test_cat:
                valid += 1
            else:
                invalid += 1
    print('Valid:', valid)
    print('Invalid:', invalid)
    print('All:', all)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_db', type=str, help='Path to the train DB file')
    parser.add_argument('--test_db', type=str, help='Path to the test DB file (_all, _avg doesn\'t make much sense)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
