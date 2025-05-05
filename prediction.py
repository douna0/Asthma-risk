import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Charger les données
df = pd.read_csv('data/asthma-data.csv')

# Si la colonne "Severity" existe et contient les 3 classes, on l'utilise directement.
# Si ce n'est pas le cas, crée une colonne "Severity" avec les trois valeurs possibles
df['Severity'] = df[["Severity_Mild", "Severity_Moderate", "Severity_None"]].idxmax(axis=1)

# Sélectionner les features (toutes les colonnes sauf la colonne cible)
X = df.drop(columns=['Severity'])

# La cible à prédire est la colonne "Severity"
y = df['Severity']

# Séparer les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données (facultatif, mais utile pour certains modèles)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialisation du modèle RandomForest pour la classification multiclasse
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement du modèle
model.fit(X_train_scaled, y_train)

# Prédictions
y_pred = model.predict(X_test_scaled)

# Affichage des résultats
print(f"Précision sur l'ensemble de test : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
