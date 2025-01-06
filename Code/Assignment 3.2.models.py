import joblib
from rdkit import Chem
import pandas as pd
from rdkit.Chem import Draw
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from rdkit.Chem import MACCSkeys
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier

# Laad CSV-train-bestand
train_data = pd.read_csv("C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/Assignment 3 - Programming/drd-3-binder-quest/train.csv")
# Laad CSV-test-bestand
test_data = pd.read_csv("C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/Assignment 3 - Programming/drd-3-binder-quest/test.csv")
# Laad submission-bestand
sample_submission = pd.read_csv("C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/Assignment 3 - Programming/drd-3-binder-quest/sample_submission.csv")

tabel = PrettyTable()
tabel.field_names = ["Type of model", "Balanced accuracy", "Precision", "Recall", "F1"]

# Extracteer en canonicaliseer SMILES
train_data["Canonical_SMILES"] = train_data["SMILES_canonical"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if pd.notna(x) else None)

def add_rows_to_table(type_model ,y_test, y_pred):
    tabel.add_row([type_model,
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
    randomforest = RandomForestClassifier(n_estimators=1000)
    randomforest.fit(X_train, y_train)
    y_pred_randomforest = randomforest.predict(X_test)
    add_rows_to_table("Randomforest", y_test, y_pred_randomforest)
    dump_model(randomforest, "randomforest model")

    clf = svm.SVC(kernel="linear")
    clf.fit(X_train, y_train)
    y_pred_clf = clf.predict(X_test)
    add_rows_to_table("Support vector classifier", y_test, y_pred_clf)
    dump_model(clf, "support vector classifier model")

X_train, X_test, y_train, y_test, train, labels = data_augmentation_with_fingerprint(50)
models(X_train, X_test, y_train, y_test)
print(tabel)


test_smiles_list = test_data["SMILES_canonical"].tolist()
test_fingerprints = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(sm)) for sm in test_smiles_list]
test = [list(fp) for fp in test_fingerprints]  # Omzetten naar een lijst van bits

trained_model_randomforest = joblib.load("randomforest model")
y_pred_ranforest = trained_model_randomforest.predict(test)
sample_submission["target_feature"] = y_pred_ranforest
sample_submission.to_csv("C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/sample_submission.csv", index=False)

trained_model_clf = joblib.load("support vector classifier model")
y_pred_clf = trained_model_clf.predict(test)
sample_submission["target_feature"] = y_pred_clf




