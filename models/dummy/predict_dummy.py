import os
import sys 
from os.path import dirname as up
from dummy_model import DummyClassifier

sys.path.insert(0, up(up(up(os.path.abspath(__file__)))))
from src.load_data import load_and_format_data


def main():
    X_train, y_train, X_test = load_and_format_data(path='../../data')
    model = DummyClassifier()
    preds = model.predict(X_test)
    preds.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()