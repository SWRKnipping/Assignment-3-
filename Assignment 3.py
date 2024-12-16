from rdkit import Chem
import pandas as pd
from rdkit.Chem import Draw


# Laad CSV-bestand
file_path = "C:/Users/20234420/OneDrive - TU Eindhoven/Desktop/Assignment 3 - Programming/drd-3-binder-quest/train.csv"
data = pd.read_csv(file_path)
# Extracteer en canonicaliseer SMILES
data["Canonical_SMILES"] = data["SMILES_canonical"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)) if pd.notna(x) else None)


# Functie om gerandomiseerde SMILES te genereren
def generate_randomized_smiles(smile, n_variants=20):
    mol = Chem.MolFromSmiles(smile)
    return [Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True) for _ in range(n_variants)]

# Data augmentatie toepassen
augmented_data = []

for _, row in data.iterrows():
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



# SMILES omzetten naar Molecuulobjecten
mols = [Chem.MolFromSmiles(smile) for smile in randomized]
# Render een rasterafbeelding
img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400, 400))
img.show()




