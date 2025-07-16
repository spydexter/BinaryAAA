import os
import numpy as np
import warnings
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------
# STEP 1: Extract ResNet50 Features
# -----------------------------------
def extract_resnet_features(base_path, size=(224, 224), limit_per_class=400):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    X = []
    y = []
    labels = sorted(os.listdir(base_path))
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        folder_path = os.path.join(base_path, label)
        if not os.path.isdir(folder_path):
            continue
        count = 0
        for filename in os.listdir(folder_path):
            if count >= limit_per_class:
                break
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = image.load_img(img_path, target_size=size)
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    features = model.predict(x, verbose=0)
                    X.append(features.flatten())
                    y.append(label_map[label])
                    count += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    return np.array(X), np.array(y)

# -----------------------------
# STEP 2: Binary AAA Functions
# -----------------------------
def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def compute_fitness(position, X, y, alpha=0.6, beta=0.1):
    selected_indices = np.where(position == 1)[0]
    if len(selected_indices) == 0:
        return float('inf')
    X_selected = X[:, selected_indices]
    skf = StratifiedKFold(n_splits=3)
    scores = []
    for train_idx, test_idx in skf.split(X_selected, y):
        clf = XGBClassifier(eval_metric='mlogloss')
        clf.fit(X_selected[train_idx], y[train_idx])
        y_pred = clf.predict(X_selected[test_idx])
        acc = accuracy_score(y[test_idx], y_pred)
        scores.append(acc)
    mean_acc = np.mean(scores)
    return alpha * (1 - mean_acc) + beta * (len(selected_indices) / len(position))

def initialize_population(pop_size, dim):
    return np.random.randint(2, size=(pop_size, dim))

def binary_aaa(X, y, pop_size=30, dim=2048, max_iter=40, alpha=0.6, beta=0.1):
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

# -----------------------------
# STEP 3: Run the Pipeline
# -----------------------------
if __name__ == "__main__":
    base_path = r"C:\\Users\\deads\\OneDrive\\Desktop\\New folder (3)\\NCT-CRC-HE-100K"

    print("📥 Extracting deep features using ResNet50...")
    X, y = extract_resnet_features(base_path, size=(224, 224), limit_per_class=400)
    print("✅ Features extracted. Shape:", X.shape)

    print("🚀 Running Binary AAA for feature selection...")
    dim = X.shape[1]
    best_solution, best_fit = binary_aaa(X, y, dim=dim, pop_size=30, max_iter=40)

    print("\n✅ Best Feature Subset Found:")
    print("Selected Features:", np.sum(best_solution))

    selected_indices = np.where(best_solution == 1)[0]
    X_selected = X[:, selected_indices]

    print("\n📊 Final Evaluation with 5-Fold Stratified Cross Validation")
    skf = StratifiedKFold(n_splits=5)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
        print(f"\nFold {fold+1}")
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('xgb', XGBClassifier(eval_metric='mlogloss'))
        ], voting='soft')

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except:
            auc = 0.0

        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        all_metrics.append((acc, prec, rec, f1, auc))

    avg_metrics = np.mean(all_metrics, axis=0)
    print("\n📈 Average Performance across 5 folds:")
    print(f"Accuracy:  {avg_metrics[0] * 100:.2f}%")
    print(f"Precision: {avg_metrics[1]:.4f}")
    print(f"Recall:    {avg_metrics[2]:.4f}")
    print(f"F1 Score:  {avg_metrics[3]:.4f}")
    print(f"AUC-ROC:   {avg_metrics[4]:.4f}")
