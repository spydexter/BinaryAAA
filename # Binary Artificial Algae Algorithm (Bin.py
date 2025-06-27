import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def compute_fitness(position, X, y, alpha=0.9, beta=0.1):
    selected_indices = np.where(position == 1)[0]
    if len(selected_indices) == 0:
        return float('inf')
    X_selected = X[:, selected_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return alpha * (1 - accuracy) + beta * (len(selected_indices) / len(position))

def initialize_population(pop_size, dim):
    return np.random.randint(2, size=(pop_size, dim))

def binary_aaa(X, y, pop_size=20, dim=100, max_iter=50, alpha=0.9, beta=0.1):
    population = initialize_population(pop_size, dim)
    energy = np.random.rand(pop_size)
    best_position = None
    best_fitness = float('inf')

    for iteration in range(max_iter):
        for i in range(pop_size):
            fitness = compute_fitness(population[i], X, y, alpha, beta)
            if fitness < best_fitness:
                best_fitness = fitness
                best_position = population[i].copy()

            rand_alga = population[np.random.randint(pop_size)]
            diff = rand_alga - population[i]
            v = np.dot(diff, energy[i])
            prob = sigmoid(v)
            new_position = np.where(np.random.rand(dim) < prob, 1, 0)

            new_fitness = compute_fitness(new_position, X, y, alpha, beta)
            if new_fitness < fitness:
                population[i] = new_position
                energy[i] += 0.05
            else:
                energy[i] -= 0.02

            if energy[i] > 1.2:
                energy[i] = 1.0
                population = np.vstack([population, new_position])
                energy = np.append(energy, 1.0)
                if len(population) > pop_size:
                    worst_index = np.argmax([compute_fitness(p, X, y) for p in population])
                    population = np.delete(population, worst_index, axis=0)
                    energy = np.delete(energy, worst_index)

        print(f"Iteration {iteration + 1}, Best Fitness: {best_fitness:.4f}")

    return best_position, best_fitness

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    dim = X.shape[1]
    best_solution, best_fit = binary_aaa(X, y, dim=dim, pop_size=20, max_iter=30)

    print("\nBest Feature Subset:", best_solution)
    print("Selected Features:", np.sum(best_solution))

    selected_features = np.where(best_solution == 1)[0]
    X_selected = X[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print(f"\n✅ Final Model Evaluation:")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
