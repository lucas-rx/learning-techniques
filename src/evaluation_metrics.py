from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class EvaluationMetrics:
    def __init__(self, t_test):
        self.t_test = t_test

    def evaluate(self, methode, pred, verbose):
        precision = precision_score(self.t_test, pred, average="macro", zero_division=0)
        recall = recall_score(self.t_test, pred, average="macro", zero_division=0)
        if precision == 0 and recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        if verbose >= 2:
            print(
                f"""*** METRICS ***
Precision : {round(precision, 3)}
Recall : {round(recall, 3)}
F1-Score : {round(f1_score, 3)}
"""
            )
        return precision, recall, f1_score

    def confusion_matrix(self, t_pred, name):
        classes = []
        ConfusionMatrixDisplay.from_predictions(
            self.t_test, t_pred, include_values=False, display_labels=classes
        )
        plt.title(name)
        plt.show()
