import os
import numpy as np
import pandas as pd
import threading
import itertools
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from sklearn.svm import SVC, LinearSVC
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import Nystroem
from sklearn import pipeline, svm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Limit OpenBLAS threads to prevent my computer from crashing
os.environ['OPENBLAS_NUM_THREADS'] = '1'

=def rbf_kernel(X, Y, gamma):
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    distances = X_norm + Y_norm - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * distances)

def select_landmarks(X, m):
    kmeans = KMeans(n_clusters=m, random_state=42)
    kmeans.fit(X)
    return kmeans.cluster_centers_

def nystroem_transform(X, Z, gamma):
    K_mm = rbf_kernel(Z, Z, gamma)
    K_mm += np.eye(K_mm.shape[0]) * 1e-6  # Ridge regularization
    U, S, V_t = np.linalg.svd(K_mm) #singlular value decomposition instead of eigendecomposition to work with non-square matrices
    K_mm_inv_sqrt = np.diag(1 / np.sqrt(S))
    K_nm = rbf_kernel(X, Z, gamma)
    return K_nm @ U @ K_mm_inv_sqrt

class LinearSVM:
    def __init__(self, C=1.0, lr=0.001, max_iter=1000, pos_weight=1.0):
        self.C = C  # Regularization parameter
        self.lr = lr  # Learning rate
        self.max_iter = max_iter  # Number of iterations
        self.pos_weight = pos_weight #Regularized weight
        self.w = None  # Weights
        self.b = 0  # Bias term

    def fit(self, X, y):
        n_features = X.shape[1]

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.max_iter):
            ela = self.lr / (1+ 0.001 * i) #dynamic learning rate to help with convergence
            for idx, x_i in enumerate(X):
                margin = y[idx] * (np.dot(x_i, self.w) + self.b)
                weight = self.pos_weight if y[idx] == 1 else 1.0 #pushing positive class weights to help with underfitting other digits

                if margin < 1:
                    # Squared hinge loss for gradient weight updates
                    grad_w = 2 * self.w - 2 * self.C * weight * (1 - margin) * y[idx] * x_i
                    grad_b = -2 * self.C * weight * (1 - margin) * y[idx]
                else:
                    grad_w = 2 * self.w
                    grad_b = 0
                self.w -= ela * grad_w
                self.b -= ela * grad_b
                self.w = np.clip(self.w, -1e2, 1e2) #limit divergence 
    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

class ovrSVM:
    #one versus rest classification for multiclass classification
    def __init__(self, C=1.0, lr=0.001, max_iter=1000):
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.classifiers = {}  # Holds the binary classifiers for each class
        self.classes_ = None  

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Identify unique classes
        for digit in self.classes_:
            y_bin = np.where(y == cls, 1, -1)
            # New LinearSVM for this class
            pos_weight = len(y) / (2 * np.sum(y_bin == 1)) #make better decision scores on imbalanced classes
            classifier = LinearSVM(C=self.C, lr=self.lr, max_iter=self.max_iter, pos_weight = pos_weight)
            classifier.fit(X, y_bin)  # Train the binary classifier
            self.classifiers[digit] = classifier
        return self

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for idx, cls in enumerate(self.classes_):
            classifier = self.classifiers[cls]
            norm_factor = np.linalg.norm(classifier.w) + 1e-6  #avoid division by zero
            scores[:, idx] = (np.dot(X, classifier.w) + classifier.b) / (norm_factor)
        # Assign class with the highest normalized score
        return self.classes_[np.argmax(scores, axis=1)]

def evaluate_model(X_train, X_test, y_train, y_test, C, gamma, landmarks):
    results = {
        "raw": {"accuracy": None, "time": None},
        "nystroem": {"accuracy": [], "time": [], "landmarks": []}
    }

    # Evaluate baseline SVM scores
    start = time.time()
    ovr = ovrSVM()
    ovr.fit(X_train, y_train)
    raw_time = time.time() - start
    raw_accuracy = accuracy_score(y_test, ovr.predict(X_test))
    results["raw"]["accuracy"] = raw_accuracy
    results["raw"]["time"] = raw_time

    def evaluate_nystroem(landmark):
        start = time.time()
        Z = select_landmarks(X_train, landmark)
        Phi_train = nystroem_transform(X_train, Z, gamma)
        Phi_test = nystroem_transform(X_test, Z, gamma)
        scaler = StandardScaler() #standardize the transformed scores
        Phi_train = scaler.fit_transform(Phi_train)
        Phi_test = scaler.transform(Phi_test)
        nystroem_ovr = ovrSVM(C=C, lr=0.001, max_iter=1000)
        nystroem_ovr.fit(Phi_train, y_train)
        runtime = time.time() - start
        accuracy = accuracy_score(y_test, nystroem_ovr.predict(Phi_test))
        return accuracy, runtime
    #parallelization to speed up timespent
    nystroem_results = Parallel(n_jobs=2)(
        delayed(evaluate_nystroem)(n) for n in n_components_list
    )

    for i, landmark in enumerate(landmarks):
        accuracy, runtime = nystroem_results[i]
        results["nystroem"]["accuracy"].append(accuracy)
        results["nystroem"]["time"].append(runtime)
        results["nystroem"]["landmarks"].append(landmark)

    return results

def baseline(X_train, X_test, y_train, y_test, gamma, C):
    #evaluating scores of sklearn models to benchmark against
    results = {}
    start = time.time()
    linear_svm = LinearSVC(C=C, max_iter=500, dual=False)
    linear_svm.fit(X_train, y_train)
    linear_time = time.time() - start
    linear_accuracy = accuracy_score(y_test, linear_svm.predict(X_test))
    results['LinearSVM'] = (linear_accuracy, linear_time)

    start = time.time()
    rbf_svm = SVC(C=C, kernel='rbf', gamma=gamma)
    rbf_svm.fit(X_train, y_train)
    rbf_time = time.time() - start
    rbf_accuracy = accuracy_score(y_test, rbf_svm.predict(X_test))
    results['RBFSVM'] = (rbf_accuracy, rbf_time)

    custom_nystroem_scores = []
    sklearn_nystroem_scores = []
    custom_nystroem_times = []
    sklearn_nystroem_times = []

    landmarks = [50, 100, 200, 500,1000,2000]  # Vary the number of components
    for landmark in landmarks:
        custom_start = time.time()
        Z = select_landmarks(X_train, landmark)  # Select landmarks
        Phi_train = nystroem_transform(X_train, Z, gamma)
        Phi_test = nystroem_transform(X_test, Z, gamma)
        svm = LinearSVC(C=C, max_iter=1000, dual=False) #testing our nystroem transformation on sklearn svm
        svm.fit(Phi_train, y_train)
        custom_time = time.time() - custom_start
        custom_accuracy = accuracy_score(y_test, svm.predict(Phi_test))

        sklearn_start = time.time()
        sklearn_nystroem = Nystroem(kernel="rbf", gamma=gamma, n_components=landmark)
        X_train_sklearn = sklearn_nystroem.fit_transform(X_train)
        X_test_sklearn = sklearn_nystroem.transform(X_test)
        sklearn_svm = LinearSVC(C=C, max_iter=1000, dual=False)
        sklearn_svm.fit(X_train_sklearn, y_train)
        sklearn_time = time.time() - sklearn_start
        sklearn_accuracy = accuracy_score(y_test, sklearn_svm.predict(X_test_sklearn))

        custom_nystroem_scores.append(custom_accuracy)
        sklearn_nystroem_scores.append(sklearn_accuracy)
        custom_nystroem_times.append(custom_time)
        sklearn_nystroem_times.append(sklearn_time)

    # Store results for plotting
    results['CustomNystroem'] = (custom_nystroem_scores, custom_nystroem_times, n_components_list)
    results['nystroem-sk'] = (sklearn_nystroem_scores, sklearn_nystroem_times, n_components_list)
    return results

# Plot Results
def plotresults(baseline, model_results):
    landmarks = [50,100,200,500,1000,2000]
    plt.figure(figsize=(12, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.axhline(y=baseline['LinearSVM'][0], color='r', linestyle='--', label="Linear SVM")
    plt.plot(landmarks, baseline['nystroem-sk'][0], marker="o", color='orange', label="Nyström (Baseline)")
    plt.plot(landmarks, baseline['CustomNystroem'][0], marker="o", color='blue', label="Nyström K-Means (sk-learn svm)")
    plt.axhline(y=baseline['RBFSVM'][0], color='g', linestyle='--', label="RBF SVM")
    plt.plot(landmarks, model_results["nystroem"]["accuracy"], marker="o", color='purple', label=" Nyström K-means w/ implemented svm ")
    plt.axhline(y=model_results["raw"]["accuracy"], color='pink', linestyle='--', label="Implemented SVM")
    plt.xlabel("Number of Landmarks (m)")
    plt.ylabel("Accuracy")
    plt.legend()

    # Runtime Plot
    plt.subplot(1, 2, 2)
    plt.axhline(y=baseline['LinearSVM'][1], color='r', linestyle='--', label="Linear SVM")
    plt.plot(landmarks, baseline['CustomNystroem'][0], marker="o", color='blue', label="Nyström (implemented)")
    plt.plot(landmarks, baseline['nystroem-sk'][1], marker="o", color='orange', label="Nyström Runtime (Baseline)")
    plt.axhline(y=baseline['RBFSVM'][1], color='g', linestyle='--', label="RBF SVM")
    plt.plot(landmarks, model_results["nystroem"]["time"], marker="o", color='purple', label=" Nyström K-means w/ implemented svm ")
    plt.axhline(y=model_results["raw"]["time"], color='pink', linestyle='--', label="Implemented SVM")
    plt.xlabel("Number of Landmarks (m)")
    plt.ylabel("Runtime (seconds)")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    mnist = pd.read_csv("mnist_train.csv")
    data = mnist.iloc[:, 1:].values / 255 #
    X, Y = shuffle(data, mnist.iloc[:, 0].values) #randomizing the order for when i take a subset of the data to train on

    X_train, y_train = X[:5000], Y[:5000]
    X_test, y_test = X[5000:7000], Y[5000:7000]


    """"
    This is a bunch of testing code that I didn't want to delete
    Z = select_landmarks(X_train, 200)
    gamma = 0.031
    Phi_train = nystroem_transform(X_train, Z, gamma)
    Phi_test = nystroem_transform(X_test, Z, gamma)
    print(f"Feature Range Pre-Nystroem: {X.min(axis=0)} to {X.max(axis=0)}")
    print(f"Feature Range Post-Nystroem: {Phi_train.min(axis=0)} to {Phi_train.max(axis=0)}")
    
    
    scaler = StandardScaler()
    Phi_train_scaled = scaler.fit_transform(Phi_train)
    Phi_test_scaled = scaler.transform(Phi_test)
    nystroem_ovr = ovrSVM(C=1.0)
    nystroem_ovr.fit(Phi_train_scaled, y_train)
    nystroem_svc = LinearSVC(C=1.0, max_iter=1000, dual=False)
    nystroem_svc.fit(Phi_train, y_train)
    ovrPred = nystroem_ovr.predict(Phi_test_scaled)
    svcPred = nystroem_svc.predict(Phi_test)
    print("Sklearn Accuracy:", accuracy_score(y_test, svcPred))
    print("OvR Accuracy:", accuracy_score(y_test, ovrPred))
    # Sklearn LinearSVC

    sklearn_svm = LinearSVC(C=1.0, max_iter=1000, dual=False)
    sklearn_svm.fit(X_train, y_train)
    sklearn_predictions = sklearn_svm.predict(X_test)
    ovr = ovrSVM()
    ovr.fit(X_train, y_train)
    ovr = ovr_sklearn.predict(X_test)
    print("Sklearn OvR Accuracy:", accuracy_score(y_test, sklearn_predictions))
    print("Custom OvR Accuracy:", accuracy_score(y_test, custom_predictions))
    print("Confusion Matrix (Sklearn):")
    print(confusion_matrix(y_test, sklearn_predictions))
    print("Sklearn Accuracy:", accuracy_score(y_test, sklearn_predictions))
    print("Custom SVM Accuracy:", accuracy_score(y_test, custom_predictions))
    print("Prediction Differences:", np.sum(sklearn_predictions != custom_predictions))
    print("Custom Matrix:",confusion_matrix(y_test,custom_predictions))

    """
    baseline_results = baseline(X_train, X_test, y_train, y_test, gamma=0.031, C=1.0)
    model_results = evaluate_model(X_train, X_test, y_train, y_test, C=1.0, gamma=0.031, landmarks=[50, 100, 200, 500, 1000, 2000])
    plotresults(baseline_results, model_results)