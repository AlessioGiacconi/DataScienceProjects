import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Caricamento dataset
file_path = 'C:\\Users\\falco\\PycharmProjects\\ProgettoSerieTemporali\\DataScienceProjects\\SerieTemporali\\Dataset\\clean_dataset.csv'
df_csv = pd.read_csv(file_path)

# Preprocessing
df_csv['IsCancelled'] = df_csv['InvoiceNo'].astype(str).str.startswith('C').astype(int)

# Aggregazione dati
total_orders_by_product = df_csv.groupby('Description')['InvoiceNo'].nunique()
cancelled_orders_by_product = df_csv[df_csv['IsCancelled'] == 1].groupby('Description')['InvoiceNo'].nunique()

# Creazione dataset per classificazione
cancellation_stats = pd.DataFrame({
    'TotalOrders': total_orders_by_product,
    'CancelledOrders': cancelled_orders_by_product
}).fillna(0)
cancellation_stats['CancellationRate'] = (cancellation_stats['CancelledOrders'] / cancellation_stats['TotalOrders']) * 100
classification_data = cancellation_stats.reset_index()
classification_data['AtRisk'] = (classification_data['CancellationRate'] > 3).astype(int)

# Definizione feature e target
X = classification_data[['TotalOrders', 'CancelledOrders']]
y = classification_data['AtRisk']

# Downsampling
majority_class = classification_data[classification_data['AtRisk'] == 0]
minority_class = classification_data[classification_data['AtRisk'] == 1]
majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)
balanced_data = pd.concat([majority_downsampled, minority_class])
X_balanced = balanced_data[['TotalOrders', 'CancelledOrders']]
y_balanced = balanced_data['AtRisk']

# Scaling
scaler = StandardScaler()
X_balanced_scaled = scaler.fit_transform(X_balanced)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Modelli individuali
rf_model = RandomForestClassifier(max_depth=5, n_estimators=25, min_samples_leaf=2, min_samples_split=2, max_features=0.7, bootstrap=False, criterion='gini',random_state=42)
svc_model = SVC(kernel='rbf', C=1, gamma=1, probability=True, random_state=42)

# Ensemble con VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[('rf', rf_model), ('svc', svc_model)],
    voting='soft'
)

# Training
ensemble_model.fit(X_train, y_train)

# Predizione e valutazione
y_pred = ensemble_model.predict(X_test)
y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]

print("Accuracy (Ensemble):", accuracy_score(y_test, y_pred))
print("Confusion Matrix (Ensemble):\n", confusion_matrix(y_test, y_pred))
print("Classification Report (Ensemble):\n", classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()