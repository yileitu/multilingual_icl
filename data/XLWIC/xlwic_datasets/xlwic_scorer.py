#!/usr/bin/env python
import parser
import sys
import os
import os.path
import codecs
from argparse import ArgumentParser

def read_submission(submission_path):
    if not os.path.exists(submission_path):
        message = "Expected submission file '{0}'"
        sys.exit(message.format(submission_path))
    with open(submission_path) as submission_file:
        submission_file_lines = list(map(lambda line: line.strip(), submission_file.readlines()))
    return submission_file_lines


def eval_lang(submission_file_lines, truth_file_path):
    with codecs.open(truth_file_path, encoding="utf8") as truth_file:
        truth_file_lines = list(map(lambda line: line.strip().split("\t")[-1], truth_file.readlines()))

    num_preditions_items = len(truth_file_lines)
    if len(submission_file_lines) < num_preditions_items:
        sys.exit(
        'Number of lines lower than test file')
    for i, l in enumerate(submission_file_lines):
        try:
            int(l)
        except ValueError:
            sys.exit("Predictions file contains malformed lines. "
                     "Lines should only contain 1 or 0 but line {} contains {}."
                     .format(i, l))
    correct = 0
    for i in range(len(truth_file_lines)):
        output_answer = submission_file_lines[i].strip()
        gold_answer = truth_file_lines[i].strip()
        if output_answer == gold_answer:
            correct += 1
    # print(lang, 100 * correct / num_preditions_items)
    return 100 * float(correct) / float(num_preditions_items)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('predictions_file', type=str)
    parser.add_argument('gold_file', type=str)
    args = parser.parse_args()
    submission_lines = read_submission(args.predictions_file)
    score = eval_lang(submission_lines, args.gold_file)
    print(f'Accuracy: {score:.4}%')
        
