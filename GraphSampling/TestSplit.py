import sys
import getopt

import pandas as pd

if __name__ == '__main__':

    input_file = ""
    training_file = ""
    validation_file = ""
    test_file = ""
    validation_percentage = 0.05
    test_percentage = 0.05

    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["input_file=", "training_file=",
                                                      "validation_file=", "test_file=",
                                                      "validation_percentage=", "test_percentage="])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt == "--input_file":
            input_file = arg
        elif opt == "--training_file":
            training_file = arg
        elif opt == "--validation_file":
            validation_file = arg
        elif opt == "--test_file":
            test_file = arg
        elif opt == "--validation_percentage":
            validation_percentage = float(arg)
        elif opt == "--test_percentage":
            test_percentage = float(arg)

    if input_file == "":
        raise ValueError("Input file must be specified!")

    if training_file == "":
        raise ValueError("Training file must be specified!")

    if validation_file == "":
        raise ValueError("Validation file must be specified!")

    if test_file == "":
        raise ValueError("Test file must be specified!")

    if validation_percentage < 0 or test_percentage < 0 or validation_percentage + test_percentage > 1:
        raise ValueError("Validation and test percentages must be between 0 and 1!")

    edges = pd.read_csv(input_file, names=["source", "type", "target"], sep="\t")

    validation_size = int(len(edges) * validation_percentage)
    test_size = int(len(edges) * test_percentage)
    train_size = len(edges) - validation_size - test_size

    train_set = edges.sample(train_size)
    validation_and_test_set = edges.drop(train_set.index)

    validation_set = validation_and_test_set.sample(validation_size)
    test_set = validation_and_test_set.drop(validation_set.index)

    # Drop all validation/ test triples, where both entities are connected in the training set
    # Info: (In WN18RR never the case, but better to implement accordint to specification)
    # https://stackoverflow.com/questions/28901683/pandas-get-rows-which-are-not-in-other-dataframe

    train_validation_merge = validation_set.merge(train_set, on=["source", "target"], how='left', indicator=True)
    train_validation_merge = train_validation_merge[train_validation_merge['_merge'] == 'left_only']
    train_validation_merge = train_validation_merge.drop(columns=['_merge', 'type_y'])
    train_validation_merge_inverse = train_validation_merge.merge(train_set, left_on=["source", "target"],
                                                                  right_on=["target", "source"], how='left',
                                                                  indicator=True)
    train_validation_merge_inverse = train_validation_merge_inverse[
        train_validation_merge_inverse['_merge'] == 'left_only']
    train_validation_merge_inverse = train_validation_merge_inverse.drop(columns=['_merge'])
    train_validation_merge_inverse.to_csv(validation_file, sep='\t', columns=["source_x", "type_x", "target_x"],
                                          header=False, index=False)

    train_test_merge = test_set.merge(train_set, on=["source", "target"], how='left', indicator=True)
    train_test_merge = train_test_merge[train_test_merge['_merge'] == 'left_only']
    train_test_merge = train_test_merge.drop(columns=['_merge', 'type_y'])
    train_test_merge_inverse = train_test_merge.merge(train_set, left_on=["source", "target"],
                                                      right_on=["target", "source"], how='left', indicator=True)
    train_test_merge_inverse = train_test_merge_inverse[
        train_test_merge_inverse['_merge'] == 'left_only']
    train_test_merge_inverse = train_test_merge_inverse.drop(columns=['_merge'])
    train_test_merge_inverse.to_csv(test_file, sep='\t', columns=["source_x", "type_x", "target_x"], header=False,
                                    index=False)

    train_set.to_csv(training_file, sep='\t', columns=["source", "type", "target"], header=False, index=False)

