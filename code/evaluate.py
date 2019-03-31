import argparse
import json
import sys

from scipy.spatial import distance


def main(args):
    with open(args.train_db, 'r') as outfile:
        train_db = json.load(outfile)
        print('Train DB has {} keys'.format(len(train_db.keys())))

    with open(args.test_db, 'r') as outfile:
        test_db = json.load(outfile)
        print('Test DB has {} keys'.format(len(test_db.keys())))
    print('Train and test db have {} common keys'.format(len(list(set(test_db.keys()).intersection(train_db.keys())))))

    all = 0
    valid = 0
    invalid = 0

    for test_cat in test_db.keys():
        print('Category {} has {} test images'.format(test_cat, len(test_db[test_cat])))
        for test_embedding in test_db[test_cat]:
            min_dst = 99999999
            min_dst_cat = None
            for train_cat in train_db.keys():
                train_embedding = train_db[train_cat]
                dst = distance.euclidean(test_embedding, train_embedding)
                # print('{} is in distance of {} to category {}'.format(test_cat, dst, train_cat))
                if dst < min_dst:
                    # print('new minimum! {}->{}: {}->{}'.format(min_dst_cat, train_cat, min_dst, dst))
                    min_dst = dst
                    min_dst_cat = train_cat
            all += 1
            if test_cat == min_dst_cat:
                print('VALID: {} was detected as {}'.format(test_cat, min_dst_cat))
                valid += 1
            else:
                print('INVALID: {} was detected as {}'.format(test_cat, min_dst_cat))
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
