import joblib
from rdkit import Chem
import pandas as pd
from rdkit.Chem import Draw
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from rdkit.Chem import MACCSkeys

# Laad CSV-train-bestand
train_data = pd.read_csv("C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/Assignment 3 - Programming/drd-3-binder-quest/train.csv")

# Laad CSV-test-bestand
test_data = pd.read_csv("C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/Assignment 3 - Programming/drd-3-binder-quest/test.csv")

# Laad submission-bestand
sample_submission = pd.read_csv("C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/Assignment 3 - Programming/drd-3-binder-quest/sample_submission.csv")



# Moleculen en labels (actief/inactief)
test_smiles_list = test_data["SMILES_canonical"].tolist()
# MACCS fingerprints genereren
test_fingerprints = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(sm)) for sm in test_smiles_list]
test = [list(fp) for fp in test_fingerprints]  # Omzetten naar een lijst van bits




# Extracteer en canonicaliseer SMILES
train_data["Canonical_SMILES"] = train_data["SMILES_canonical"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if pd.notna(x) else None)


# Functie om gerandomiseerde SMILES te genereren
def generate_randomized_smiles(smile, n_variants=2):
    mol = Chem.MolFromSmiles(smile)
    return [Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True) for _ in range(n_variants)]

# Data augmentatie toepassen
augmented_data = []
for _, row in train_data.iterrows():
    if pd.notna(row["Canonical_SMILES"]):  # Alleen als SMILES geldig is
        randomized = generate_randomized_smiles(row["Canonical_SMILES"])
        for rand_smile in randomized:
            # Vergelijk de canonical SMILES
            original_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(row["Canonical_SMILES"]), canonical=True)
            random_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(rand_smile), canonical=True)

            if original_canonical != random_canonical:
                print("Structuren zijn verschillend!")

            augmented_data.append({"SMILES": rand_smile, "Active": row["target_feature"]})

# Maak een DataFrame van de augmented data
augmented_df = pd.DataFrame(augmented_data)
remove_duplicates = augmented_df.drop_duplicates(subset="SMILES")

# Toon het resultaat
print(remove_duplicates)

# Moleculen en labels (actief/inactief)
smiles_list = remove_duplicates["SMILES"].tolist()
labels = remove_duplicates["Active"].tolist()

# MACCS fingerprints genereren
fingerprints = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(sm)) for sm in smiles_list]
train = [list(fp) for fp in fingerprints]  # Omzetten naar een lijst van bits

# Train/test-splitsing
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=42)


clf = svm.SVC(kernel="linear")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("B-Accuracy:", metrics.balanced_accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))



# Sla het getrainde model op
joblib.dump(clf, "trained_model")
print("Model is opgeslagen als trained_model")


# SMILES omzetten naar Molecuulobjecten
mols = [Chem.MolFromSmiles(smile) for smile in smiles_list[:20]]  # Neem een subset voor visualisatie

# Render een rasterafbeelding
img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400, 400))
img.show()

# Laad het model
trained_model = joblib.load("trained_model")
print("Model is geladen")

# Gebruik het geladen model voor voorspellingen
y_pred = trained_model.predict(test)
print(y_pred)

sample_submission["target_feature"] = y_pred
# Sla het bijgewerkte bestand op
uitvoer_pad = "C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/sample_submission.csv"
sample_submission.to_csv(uitvoer_pad, index=False)


