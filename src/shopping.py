import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4
months={"Jan", "Feb", "Mar", "May", "June", "Jul", }


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    Integer_Value = ["Administrative", "Informational", "ProductRelated", "OperatingSystems", "Browser", "Region", "TrafficType"]
    Float_Value = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
    months = {"Jan": 0, "Feb": 1, "Mar": 2, "April": 3, "May": 4,"June": 5, "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}

    visitor_type = {"Returning_Visitor": 1, "New_Visitor": 0, "Other": 0}

    weekend = {"FALSE": 0, "TRUE": 1}

    evidence = []
    labels = []

    with open(filename, mode="r") as file:
        csvFile = csv.DictReader(file, delimiter=",")
        for row in csvFile:
            lines = []

            for key in row:
                if key in Integer_Value:
                    lines.append(int(row[key]))
                elif key in Float_Value:
                    lines.append(float(row[key]))
                elif key == "Month":
                    lines.append(months[row[key]])
                elif key == "Weekend":
                    lines.append(weekend[row[key]])
                elif key == "VisitorType":
                    lines.append(visitor_type[row[key]])
                else:
                    labels.append(1 if row["Revenue"] == "True" else 0)
            evidence.append(lines)
            labels.append(1 if row["Revenue"]== "True"  else 0)

    return evidence, labels

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(evidence, labels)
    return knn


def evaluate(labels, predictions):

    positive = labels.count(1)
    negative = labels.count(0)
    sensitivity = 0.0
    specificity = 0.0

    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            if labels[i] == 1:
                sensitivity +=1
            else:
                specificity +=1

    sensitivity = sensitivity/ positive
    specificity = specificity/ negative
    return sensitivity, specificity


if __name__ == "__main__":
    main()
