from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd


class DataGestion:
    def __init__(self, path_csv, is_normalized):
        self.path = path_csv
        self.is_normalized = is_normalized

    def normalize(self, df):
        columns_to_normalize = list(df.columns)
        columns_to_normalize.remove("id")
        columns_to_normalize.remove("species")

        scaler = StandardScaler()
        ct = ColumnTransformer([("normalizer", scaler, columns_to_normalize)])
        df[columns_to_normalize] = ct.fit_transform(df)

    def transformation(self, attribute):
        df = pd.read_csv(self.path)
        if self.is_normalized:
            self.normalize(df)
        le = LabelEncoder().fit(df[attribute])
        df[attribute] = le.transform(df[attribute])
        return df

    def balanced_train_test_split(self, df, test_size=0.2):
        """
        Split a data frame in 4 : X_train, X_test, t_train, t_test.
        For all data frame, the number of samples of each class is equal.
        Example with size=0.2 and 100 classes of 10 samples :
        8 samples of each class in X_train and t_train
        2 samples of each class X_test and t_test
        """
        df = pd.DataFrame(df.drop(columns="id").reset_index(drop=True).sample(frac=1))
        number_tests_samples = 10 * test_size  # 10 samples per class

        selected_tests_samples_per_class = {}
        train_index = []
        test_index = []

        for class_ in pd.unique(df["species"]):
            selected_tests_samples_per_class.update({class_: 0})

        for row in df.itertuples():
            if selected_tests_samples_per_class[row.species] < number_tests_samples:
                test_index.append(row.Index)
                selected_tests_samples_per_class[row.species] += 1
            else:
                train_index.append(row.Index)

        X_train = df.loc[train_index]
        X_test = df.loc[test_index]

        t_train = X_train["species"]
        t_test = X_test["species"]

        X_train = X_train.drop(columns="species")
        X_test = X_test.drop(columns="species")

        return X_train, X_test, t_train, t_test
