import argparse

from sklearn.model_selection import train_test_split
import data_gestion as dg
import evaluation_metrics as evm
import numpy as np
import time

from models import generative_model, logistic_regression, KNN, GBT, neural_network, perceptron, SVC

MODELS = [
    "generative",
    "knn",
    "gbt",
    "perceptron",
    "logistic",
    "svc",
    "neural_network"
]

def do_n_simulations(N, model_type, verbose, is_normalized, is_split_balanced):

    is_normalized = bool(is_normalized)
    is_split_balanced = bool(is_split_balanced)

    print()
    precision = []
    recall = []
    f1_score = []
    duration = []

    for i in range(N):
        
        if verbose >= 1:
            print(f"----- ITERATION {i + 1} / {N} -----\n")

        data_gestion = dg.DataGestion("./data/train.csv", is_normalized)
        data = data_gestion.transformation("species")
        if is_split_balanced:
            X_train, X_test, t_train, t_test = data_gestion.balanced_train_test_split(data, test_size=0.2)
        else:
            t = data["species"]
            X = data.drop(columns=["species", "id"]).reset_index(drop=True)
            X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2)
        model = None

        if model_type == "generative":
            model = generative_model.GenerativeModel()
        elif model_type == "knn":
            model = KNN.KNN()
        elif model_type == "gbt":
            model = GBT.GBT()
        elif model_type == "perceptron":
            model = perceptron.PerceptronModel()
        elif model_type == "logistic":
            model= logistic_regression.Logistic()
        elif model_type == "svc":
            model = SVC.SVC()
        else:
            model = neural_network.NeuralNetwork()

        ev = evm.EvaluationMetrics(t_test)

        beginning = time.time()

        model.search_best_parameters(X_train, t_train)
        if verbose >= 2:
            model.print_parameters()
        model.train(X_train, t_train)
        prediction = model.predict(X_test)
        prec, rec, f1 = ev.evaluate(model, prediction, verbose)
        delta = int(time.time() - beginning)

        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)
        duration.append(delta)

        if verbose >= 1:
            print(f"*** TIME ***\n{delta // 60} min {delta % 60} sec\n")

    print(f'''----- FINAL REPORT -----\n
Precision : 
Mean : {round(np.mean(precision), 3)}
Std : {round(np.std(precision), 3)}

Recall : 
Mean : {round(np.mean(recall), 3)}
Std : {round(np.std(recall), 3)}

F1-Score : 
Mean : {round(np.mean(f1_score), 3)}
Std : {round(np.std(f1_score), 3)}

Mean time : {int(np.mean(duration) // 60)} min {int(np.mean(duration) % 60)} sec
''')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number_iterations", type=int, default=20, help="Number of iterations")
    parser.add_argument("-m", "--model", type=str, default="neural_network", help="Model")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbose level")
    parser.add_argument("--normalize", type=int, default=1, help="Data normalization")
    parser.add_argument("--balancing_split", type=int, default=1, help="Ensure that each class is equally represented in train and test datasets")
    args = parser.parse_args()

    if args.number_iterations > 0 and args.model in MODELS:
        do_n_simulations(args.number_iterations, args.model, args.verbose, args.normalize, args.balancing_split)
    else:
        print(f"Wrong input. Number of iterations must be > 0. Model must be in {MODELS}.")
