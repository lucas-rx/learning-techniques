
# Techniques d'apprentissage

- Matéo DEMANGEON
- Martin GUITTENY
- Lucas RIOUX

## Utilisation

1. Accédez au répertoire du projet.

2. Installez les dépendances requises en utilisant pip dans votre environement virtuel.

    ```bash
    pip install -r requirements.txt
    ```

3. Depuis la racine du projet, exécutez le fichier principal.
    ```bash
    python3 src/main.py -n <nombre_d_iterations> -m <type_de_modele> -v <niveau_de_verbosité>
    ```

- `nombre_d_iterations` : Le nombre d'itérations pour les simulations.
- `type_de_modele` : Le type de modèle de classification à utiliser. Choisissez parmi : `generative`, `knn`, `gbtree`, `perceptron`, `logistic`, `svc`, `neural_network` .
- `niveau_de_verbosité` : Le niveau de verbosité pour la sortie, `0`, `1` ou `2`.

4. Forme avancée :
    ```bash
    python3 src/main.py -n <nombre_d_iterations> -m <type_de_modele> -v <niveau_de_verbosité> --normalize=<normalisation> --balancing_split=<equilibrage_des_classes>
    ```

- `normalize` : normalise les données (Z-Score) avant de les utiliser dans le modèle. Activé par défaut. Vaut `0` ou `1`.
- `balancing_split` : dans les ensembles d'entraînement et de test, il y aura autant de données de chaque classe. Activé par défaut. Vaut `0` ou `1`.
