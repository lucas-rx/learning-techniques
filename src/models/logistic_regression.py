from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


class Logistic:
    def __init__(self, max_iter=1000, C=1):
        self.max_iter = max_iter
        self.C = C
        self.model = LogisticRegression(C=C)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def search_best_parameters(self, X, y, cv=5):
        param_grid = {"max_iter": [500, 1000, 2000], "C": [0.01, 0.1, 1, 10, 100, 1000]}
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X, y)
        self.max_iter = grid_search.best_params_.get("max_iter")
        self.C = grid_search.best_params_.get("C")
        self.model = grid_search.best_estimator_

    def print_parameters(self):
        print(
            f"""*** PARAMETERS ***
max_iter : {self.max_iter}
C : {self.C}
"""
        )

    def name(self):
        return "Logistic regression"
