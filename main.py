import random
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


def centroid(X, Y):
    clusters = {}
    for x, y in zip(X, Y):
        if y in clusters:
            clusters[y].append(x)
        else:
            clusters[y] = [x]
    centroids = []
    centroid_label = []
    for label, cluster in clusters.items():
        centroid = np.mean(cluster, axis=0)
        centroids.append(centroid)
        centroid_label.append(label)
    return centroids, centroid_label


def best_centroid (X, centroids, centroid_label):
    distances = []
    for c, label in zip(centroids, centroid_label):
        dist = np.linalg.norm(c - X)
        distances.append((label, dist))
    return min(distances, key=lambda x: x[1])[0]


def main():
    # Load Iris dataset
    k = 11
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    etiquetas = list(iris.target_names)
    # Randomize
    datos = [(x, y) for x, y in zip(X, Y)]
    random.shuffle(datos)
    X, Y = zip(*datos)
    # Split
    X_train = X[:100]
    Y_train = Y[:100]
    X_test = X[100:]
    Y_test = Y[100:]
    centroids, centroid_label = centroid(X_train, Y_train)

    buenas = 0
    Y_obteined = []
    # Predict
    for muestra, etiqueta in zip(X_test, Y_test):
        best_y = best_centroid(muestra, centroids, centroid_label)
        Y_obteined.append(best_y)
        if best_y == etiqueta:
            buenas += 1
        print("Etiqueta real: ", etiquetas[etiqueta], "Etiqueta predicha: ", etiquetas[best_y])
    print("Porcentaje de aciertos: ", buenas / len(Y_test) * 100, "%")

    petalo_centroid = [(x[0], x[1], etiqueta) for x, etiqueta in zip(centroids, centroid_label)]
    sepalo_centroid = [(x[2], x[3], etiqueta) for x, etiqueta in zip(centroids, centroid_label)]
    petalo_test = [(x[0], x[1], etiqueta) for x, etiqueta in zip(X_test, Y_obteined)]
    sepalo_test = [(x[2], x[3], etiqueta) for x, etiqueta in zip(X_test, Y_obteined)]
    colores = ['red', 'green', 'blue']

    for x, y, etiqueta in sepalo_centroid:
        plt.scatter(x, y, color=colores[etiqueta])
    for x, y, etiqueta in sepalo_test:
        plt.scatter(x, y, color=colores[etiqueta], alpha=0.2)
    plt.show()
    plt.clf()

    for x, y, etiqueta in petalo_centroid:
        plt.scatter(x, y, color=colores[etiqueta])
    for x, y, etiqueta in petalo_test:
        plt.scatter(x, y, color=colores[etiqueta], alpha=0.2)
    plt.show()
    plt.clf()



if __name__ == "__main__":
    main()