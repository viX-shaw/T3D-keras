from random import shuffle
import csv
import glob
import re
import argparse

action_classes = None
def create_train_csv(fp):
    global action_classes
    action_classes = set([re.split("[0-9]+",e.split("/")[-1].split(".")[0])[0] for e in glob.glob("/content/UCF_Crime/*")])
    print(action_classes)
    train = []
    with open(fp, 'r') as f:
        _files = [e.split("/")[-1] for e in f.readlines()]
        print(_files[:5])
    for idx, entry in action_classes:
        for filename in glob.glob("/content/UCF_Crime/{}*".format(entry)):
            if idx < 5:
                print("F  ", filename.split("/")[-1])
            if filename.split("/")[-1] in _files:
                train.append([filename, idx, entry])
    with open('/content/anomaly_train.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(["path", "class", "sport"])
        mywriter.writerows(train)
    print("Training csv created successfully.")

def create_test_csv(fp):
    test = []
    with open(fp, 'r') as f:
        _files = [e.split("/")[-1] for e in f.readlines()]
    for idx, entry in action_classes:
        for filename in glob.glob("/content/UCF_Crime/{}*".format(entry)):
            if filename.split("/")[-1] in _files:
                test.append([filename, idx, entry])
    with open('/content/anomaly_test.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(["path", "class", "sport"])
        mywriter.writerows(test)
    print("Test csv created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creating train test split for Training T3D on UCF_Crime")
    parser.add_argument("--path_to_train",type=str, default="/content/Anomaly_Train.txt")
    parser.add_argument("--path_to_test",type=str, default="/content/Anomaly_Test.txt")
    params = parser.parse_args()
    create_train_csv(params.path_to_train)
    create_test_csv(params.path_to_test)

