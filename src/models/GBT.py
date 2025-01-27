from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier


class GBT:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def search_best_parameters(self, X, t, cv=5):
        param_grid = {
            "max_depth": [2, 3, 5, 7, 9],
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.01, 0.1, 1, 10],
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X, t)
        self.max_depth = grid_search.best_params_.get("max_depth")
        self.n_estimators = grid_search.best_params_.get("n_estimators")
        self.learning_rate = grid_search.best_params_.get("learning_rate")
        self.model = grid_search.best_estimator_

    def print_parameters(self):
        print(
            f"""*** PARAMETERS ***
max_depth : {self.max_depth}
n_estimators : {self.n_estimators}
learning_rate : {self.learning_rate}
"""
        )

    def name(self):
        return "Gradient Boosting Trees"
