import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

# File paths
# DATABASE_PATH is a csv file that is used to construct the model
# MODEL_PATH is the location that the file is outputted to
DATABASE_PATH = r"C:\Users\Valued Customer\PycharmProjects\InterfaceWithPredictor\database3.csv"
MODEL_PATH = r"C:\Users\Valued Customer\PycharmProjects\InterfaceWithPredictor\trained_model.sav"

# Headers
HEADERS = []


def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, test_x, train_y, test_y
    """

    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y


def handel_missing_values(dataset, missing_values_header, missing_label):
    """
    Filter missing values from the dataset
    :param dataset:
    :param missing_values_header:
    :param missing_label:
    :return:
    """

    return dataset[dataset[missing_values_header] != missing_label]


def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf


def dataset_statistics(dataset):
    """
    Basic statistics of the dataset
    :param dataset: Pandas dataframe
    :return: None, print the basic statistics of the dataset
    """
    print(dataset.describe())

def get_headers(dataset):
    return dataset.columns.values

def main():
    """
    Main function
    :return:
    """
    # Load the csv file into pandas dataframe
    dataset = pd.read_csv(DATABASE_PATH)
    # Get basic statistics of the loaded dataset
    dataset_statistics(dataset)
    HEADERS = get_headers(dataset)
    print(HEADERS)

    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.8, HEADERS[1:], HEADERS[0])

    # Train and Test dataset size details
    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)

    # Create random forest classifier instance
    trained_model = random_forest_classifier(train_x, train_y)

    # Cross Validation
    scores = cross_val_score(trained_model, train_x, train_y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print("Trained model :: ", trained_model)
    predictions = trained_model.predict(test_x.values)

    for i in range(0, 5):
        print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))

    print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions))

    joblib.dump(trained_model, MODEL_PATH)


if __name__ == "__main__":
    main()