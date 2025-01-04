from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import Draw, MACCSkeys
from prettytable import PrettyTable
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Laad CSV-train-bestand
train_data = pd.read_csv("C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/Assignment 3 - Programming/drd-3-binder-quest/train.csv")
# Laad CSV-test-bestand
test_data = pd.read_csv("C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/Assignment 3 - Programming/drd-3-binder-quest/test.csv")
# Laad submission-bestand
sample_submission = pd.read_csv("C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/Assignment 3 - Programming/drd-3-binder-quest/sample_submission.csv")

tabel = PrettyTable()
tabel.field_names = ["Type of model","N augmentation","Balanced accuracy", "Precision", "Recall", "F1"]

# Extracteer en canonicaliseer SMILES
train_data["Canonical_SMILES"] = train_data["SMILES_canonical"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if pd.notna(x) else None)

def add_rows_to_table(type_model ,y_test, y_pred, n):
    tabel.add_row([type_model,
                   n,
                   metrics.balanced_accuracy_score(y_test, y_pred),
                   metrics.precision_score(y_test, y_pred),
                   metrics.recall_score(y_test, y_pred),
                   metrics.f1_score(y_test, y_pred)])
    
def dump_model(model_type, model_name):
    joblib.dump(model_type, model_name)
    print(f"Model is opgeslagen als {model_name}")

# Functie om gerandomiseerde SMILES te genereren
def generate_randomized_smiles(smile, n):
    mol = Chem.MolFromSmiles(smile)
    return [Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True) for _ in range(n)]

def data_augmentation_with_fingerprint(n_variants):
    # Data augmentatie toepassen
    augmented_data = []
    for _, row in train_data.iterrows():
        if pd.notna(row["Canonical_SMILES"]):  # Alleen als SMILES geldig is
            randomized = generate_randomized_smiles(row["Canonical_SMILES"], n_variants)
            for rand_smile in randomized:
                # Vergelijk de canonical SMILES
                original_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(row["Canonical_SMILES"]), canonical=True)
                random_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(rand_smile), canonical=True)
                if original_canonical != random_canonical:
                    print("Structuren zijn verschillend!")
                augmented_data.append({"SMILES": rand_smile, "Active": row["target_feature"]})
      
    """
    Maak dataframe, 
    verwijder dubbelen, 
    maak lijst van smiles en labels,
    geef fingerprint van smile lijst 
    """
    augmented_df = pd.DataFrame(augmented_data)
    remove_duplicates = augmented_df.drop_duplicates(subset="SMILES")
    smiles_list = remove_duplicates["SMILES"].tolist()
    labels = remove_duplicates["Active"].tolist()
    fingerprints = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(sm)) for sm in smiles_list]
    train = [list(fp) for fp in fingerprints]
    
    # SMILES omzetten naar Molecuulobjecten
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list[:n_variants]]  # Neem een subset voor visualisatie
    # Render een rasterafbeelding
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400, 400))
    img.show()

    # Train/test-splitsing
    X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, stratify=labels, random_state=42)
    return X_train, X_test, y_train, y_test, train, labels

def models(X_train, X_test, y_train, y_test):
    randomforest = RandomForestClassifier(n_estimators=100)
    randomforest.fit(X_train, y_train)
    y_pred_randomforest = randomforest.predict(X_test)
    add_rows_to_table("Randomforest", y_test, y_pred_randomforest, i)
    dump_model(randomforest, "randomforest model")

    clf = svm.SVC(kernel="linear")
    clf.fit(X_train, y_train)
    y_pred_clf = clf.predict(X_test)
    add_rows_to_table("Support vector classifier", y_test, y_pred_clf, i)
    dump_model(clf, "support vector classifier model")

for i in range(1,21,5):
    X_train, X_test, y_train, y_test, train, labels = data_augmentation_with_fingerprint(i)
    models(X_train, X_test, y_train, y_test)
print(tabel)





X_train, X_test, y_train, y_test, train, labels = data_augmentation_with_fingerprint(10)
model = RandomForestClassifier(random_state=42)

# Parametergrid: Definieer de mogelijke waarden van hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

scoring = {
    "balanced_accuracy" : "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}
# Stratified K-Fold instellen
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV instellen
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=scoring,
    cv=kfold,  # Gebruik k-fold cross-validation
    refit="balanced_accuracy",
    verbose=2,
    n_jobs=-1  # Gebruik alle beschikbare cores
)

# Pas GridSearch toe op de trainingdata
grid_search.fit(X_train, y_train)

# Resultaten bekijken
print("Beste hyperparameters:", grid_search.best_params_)
print("Beste score:", grid_search.best_score_)

# Beste model toepassen op testdata
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Accuracy op testset:", metrics.accuracy_score(y_test, y_pred))


test_smiles_list = test_data["SMILES_canonical"].tolist()
test_fingerprints = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(sm)) for sm in test_smiles_list]
test = [list(fp) for fp in test_fingerprints]  # Omzetten naar een lijst van bits
# Beste model toepassen op submission
best_model = grid_search.best_estimator_
y_pred_submission = best_model.predict(test)
sample_submission["target_feature"] = y_pred_submission
sample_submission.to_csv("C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/sample_submission.csv", index=False)

results = pd.DataFrame(grid_search.cv_results_)
results.plot(x='param_n_estimators', y='mean_test_f1', kind='line')



# Pivot de data voor een heatmap
heatmap_data = results.pivot_table(
    index='param_max_depth',
    columns='param_n_estimators',
    values='mean_test_balanced_accuracy'
)

# Maak een heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".3f")
plt.title("Balanced Accuracy voor Hyperparameter Combinaties")
plt.xlabel("n_estimators")
plt.ylabel("max_depth")
plt.show()


# Plot verschillende scoringsmetrics
plt.figure(figsize=(12, 6))

plt.plot(results['mean_test_balanced_accuracy'], label='Balanced Accuracy', marker='o')
plt.plot(results['mean_test_accuracy'], label='Accuracy', marker='s')
plt.plot(results['mean_test_f1'], label='F1-Score', marker='^')
plt.plot(results['mean_test_precision'], label='Precision', marker='x')
plt.plot(results['mean_test_recall'], label='Recall', marker='d')

plt.title("Vergelijking van Scores voor Hyperparameter Combinaties")
plt.xlabel("Hyperparameter Combinaties (index)")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.show()




