import os
import sys

import pandas as pd
import sklearn.model_selection as skl_m

from src.model_chooser import get_model, DEFAULT_MODEL, store_model, Model
from src.text_processing import vectorize, perform_vectorization
from src.feature_extractor import perform_reduction, extract_features
from src.classifier import train


def main():
    if len(sys.argv) < 2:
        print("Please provide text as a command-line argument.")
        return

    text = ' '.join(sys.argv[1:])
    model = get_model(DEFAULT_MODEL)
    reduced_features = perform_reduction(model.ipca, perform_vectorization(model.vectorizer, [text]), False)
    result = dict()
    for target in model.gnbs:
        result.__setitem__(target, model.gnbs[target].predict(reduced_features))
    for key, value in result.items():
        print(f"{key}: {value}")


def evaluate_model(target, gnb, X_test, y_test):
    predicted = gnb.predict(X_test)
    correct_amount = 0
    incorrect_amount = 0
    for i in range(y_test.shape[0]):
        y_test_row = y_test.iloc[i]
        y_predicted = predicted[i]
        if y_test_row == y_predicted:
            correct_amount += 1
        else:
            incorrect_amount += 1
    print("\nFor " + target + " accuracy is " + str(correct_amount * 1.0 / (correct_amount + incorrect_amount)))


def create_model(model_name, dataset_length, amount_of_components):
    train_data = pd.read_csv(os.path.join('data', 'train.csv'))
    dataset_length = min(dataset_length, train_data.shape[0])
    dataset = train_data[0:dataset_length]
    train_comments = dataset['comment_text']
    comments_vectorized, vectorizer = vectorize(train_comments)
    X_train, X_test, y_train, y_test = (
        skl_m.train_test_split(comments_vectorized, dataset))
    features_reduced, ipca = extract_features(X_train, num=amount_of_components)

    gnbs = dict()
    targets = dataset.columns.to_list()
    targets.remove('id')
    targets.remove('comment_text')
    for target in targets:
        gnb = train(features_reduced, y_train[target])
        gnbs.__setitem__(target, gnb)
        evaluate_model(target, gnb, perform_reduction(ipca, X_test), y_test[target])

    store_model(model_name, Model(ipca, gnbs, vectorizer))


def evaluate_on_the_rest_of_dataset(initial_length):
    train_data = pd.read_csv(os.path.join('data', 'train.csv'))
    dataset = train_data[initial_length:]
    model = get_model(DEFAULT_MODEL)
    reduced_features = perform_reduction(model.ipca, perform_vectorization(model.vectorizer, dataset['comment_text']))
    for target in model.gnbs:
        evaluate_model(target, model.gnbs[target], reduced_features, dataset[target])


if __name__ == '__main__':
    main()
