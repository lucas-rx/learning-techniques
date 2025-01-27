from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self, n=5):
        self.n = n
        self.model = KNeighborsClassifier(n_neighbors=self.n)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def search_best_parameters(self, X, t, cv=5):
        param_grid = {"n_neighbors": [5, 7, 10, 12, 15]}
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X, t)
        self.C = grid_search.best_params_.get("C")
        self.kernel = grid_search.best_params_.get("kernel")
        self.gamma = grid_search.best_params_.get("gamma")
        self.model = grid_search.best_estimator_

    def print_parameters(self):
        print(
            f"""*** PARAMETERS ***
n : {self.n}
"""
        )

    def name(self):
        return "K-nearest neighbors"
