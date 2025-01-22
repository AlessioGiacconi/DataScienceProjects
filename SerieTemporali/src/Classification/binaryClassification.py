import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, \
    precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV

base_dir = Path(__file__).resolve().parent

file_path = 'C:\\Users\\falco\\PycharmProjects\\ProgettoSerieTemporali\\DataScienceProjects\\SerieTemporali\\Dataset\\clean_dataset.csv'

df_csv = pd.read_csv(file_path)

df_csv['IsCancelled'] = df_csv['InvoiceNo'].astype(str).str.startswith('C').astype(int)

# Numero totale di ordini per prodotto
total_orders_by_product = df_csv.groupby('Description')['InvoiceNo'].nunique()

# Numero di cancellazioni per prodotto
cancelled_orders_by_product = df_csv[df_csv['IsCancelled'] == 1].groupby('Description')['InvoiceNo'].nunique()

# Creare un DataFrame con i dati
cancellation_stats = pd.DataFrame({
    'TotalOrders': total_orders_by_product,
    'CancelledOrders': cancelled_orders_by_product
})

# Sostituire i NaN (prodotti senza cancellazioni) con 0
cancellation_stats['CancelledOrders'] = cancellation_stats['CancelledOrders'].fillna(0)

# Calcolare la percentuale di cancellazioni
cancellation_stats['CancellationRate'] = (cancellation_stats['CancelledOrders'] / cancellation_stats['TotalOrders']) * 100

# Ordinare i prodotti per percentuale di cancellazioni
cancellation_stats = cancellation_stats.sort_values(by='CancellationRate', ascending=False)

# Calcolare la media di Quantity e UnitPrice per prodotto
product_stats = df_csv.groupby('Description').agg({
    'Quantity': 'mean',
    'UnitPrice': lambda x: x.mode()[0],
    'Country': lambda x: x.mode()[0]  # Prendere il paese più frequente
}).reset_index()

# Unire con il dataset delle cancellazioni
cancellation_stats = cancellation_stats.reset_index()  # Assicurarsi che l'indice sia un campo
classification_data = pd.merge(cancellation_stats, product_stats, on='Description', how='left')
classification_data['AtRisk'] = (classification_data['CancellationRate'] > 3).astype(int)
classification_data['Country'] = classification_data['Country'].astype('category').cat.codes.astype('int64')

'''
# Selezionare solo le colonne numeriche
numerical_data = classification_data.select_dtypes(include=['float64', 'int64'])

# Calcolare la matrice di correlazione
correlation_matrix = numerical_data.corr()

# Stampare le correlazioni con la variabile target "AtRisk"
print(correlation_matrix['AtRisk'].sort_values(ascending=False))

# Creare una heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice di Correlazione')
plt.show()
'''

# Selezionare le feature e il target
X = classification_data[['UnitPrice', 'CancelledOrders', 'Quantity', 'Country']]
y = classification_data['AtRisk']

# =========================
# Downsampling delle classi
# =========================

# Concatenare feature e target in un unico DataFrame
data = pd.concat([X, y], axis=1)

# Separare le classi
majority_class = data[data['AtRisk'] == 0]
minority_class = data[data['AtRisk'] == 1]

# Downsampling della classe maggioritaria
majority_downsampled = resample(
    majority_class,
    replace=False,  # Senza replacement
    n_samples=len(minority_class),  # Stesso numero della classe minoritaria
    random_state=42
)

# Concatenare il dataset bilanciato
balanced_data = pd.concat([majority_downsampled, minority_class])

# Separare nuovamente feature e target
X_balanced = balanced_data.drop(columns=['AtRisk'])
y_balanced = balanced_data['AtRisk']

scaler = StandardScaler()
X_balanced_scaled = scaler.fit_transform(X_balanced)

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# =========================
# Random Forest Classifier
# =========================
rf_model = RandomForestClassifier(max_depth=3, n_estimators=50, random_state=42)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cross_val_scores_rf = cross_val_score(rf_model, X_balanced_scaled, y_balanced, cv=kf, scoring='accuracy')
print(f"Random Forest Cross-Validation Scores (k=5): {cross_val_scores_rf}")
print(f"Random Forest Mean Accuracy: {cross_val_scores_rf.mean():.4f}")

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy (Downsampling):", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix (Random Forest):\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

# =========================
# Support Vector Classifier
# =========================
svc_model = SVC(kernel='rbf', C=1, random_state=42)
cross_val_scores_svc = cross_val_score(svc_model, X_balanced_scaled, y_balanced, cv=kf, scoring='accuracy')
print(f"SVC Cross-Validation Scores (k=5): {cross_val_scores_svc}")
print(f"SVC Mean Accuracy: {cross_val_scores_svc.mean():.4f}")

svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)

print("SVC Accuracy (Downsampling):", accuracy_score(y_test, y_pred_svc))
print("Confusion Matrix (SVC):\n", confusion_matrix(y_test, y_pred_svc))
print("Classification Report (SVC):\n", classification_report(y_test, y_pred_svc))

# =========================
# Logistic Regression
# =========================
log_reg = LogisticRegression(random_state=42, C= 1, max_iter=500)

# Cross-validation
log_reg_cv_scores = cross_val_score(log_reg, X_balanced_scaled, y_balanced, cv=kf, scoring='accuracy')
print(f"Logistic Regression Cross-Validation Scores (k=5): {log_reg_cv_scores}")
print(f"Logistic Regression Mean Accuracy: {log_reg_cv_scores.mean():.4f}")

# Addestramento sul training set
log_reg.fit(X_train, y_train)

# Previsioni
y_pred_log_reg = log_reg.predict(X_test)

# Valutazione
print("Logistic Regression Accuracy (Downsampling):", accuracy_score(y_test, y_pred_log_reg))
print("Confusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred_log_reg))
print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_log_reg))

# =========================
# Decision Tree Classifier
# =========================
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)

# Cross-validation
dt_cv_scores = cross_val_score(dt_model, X_balanced_scaled, y_balanced, cv=kf, scoring='accuracy')
print(f"Decision Tree Cross-Validation Scores (k=5): {dt_cv_scores}")
print(f"Decision Tree Mean Accuracy: {dt_cv_scores.mean():.4f}")

# Addestramento sul training set
dt_model.fit(X_train, y_train)

# Previsioni
y_pred_dt = dt_model.predict(X_test)

# Valutazione
print("Decision Tree Accuracy (Downsampling):", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix (Decision Tree):\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))

# =========================
# Gradient Boosting Classifier
# =========================
gb_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)

# Cross-validation
gb_cv_scores = cross_val_score(gb_model, X_balanced_scaled, y_balanced, cv=kf, scoring='accuracy')
print(f"Gradient Boosting Cross-Validation Scores (k=5): {gb_cv_scores}")
print(f"Gradient Boosting Mean Accuracy: {gb_cv_scores.mean():.4f}")

# Addestramento sul training set
gb_model.fit(X_train, y_train)

# Previsioni
y_pred_gb = gb_model.predict(X_test)

# Valutazione
print("Gradient Boosting Accuracy (Downsampling):", accuracy_score(y_test, y_pred_gb))
print("Confusion Matrix (Gradient Boosting):\n", confusion_matrix(y_test, y_pred_gb))
print("Classification Report (Gradient Boosting):\n", classification_report(y_test, y_pred_gb))

# =========================
# Linear Discriminant Analysis (LDA)
# =========================
lda_model = LinearDiscriminantAnalysis()

# Cross-validation
lda_cv_scores = cross_val_score(lda_model, X_balanced_scaled, y_balanced, cv=kf, scoring='accuracy')
print(f"LDA Cross-Validation Scores (k=5): {lda_cv_scores}")
print(f"LDA Mean Accuracy: {lda_cv_scores.mean():.4f}")

# Addestramento sul training set
lda_model.fit(X_train, y_train)

# Previsioni
y_pred_lda = lda_model.predict(X_test)

# Valutazione
print("LDA Accuracy (Downsampling):", accuracy_score(y_test, y_pred_lda))
print("Confusion Matrix (LDA):\n", confusion_matrix(y_test, y_pred_lda))
print("Classification Report (LDA):\n", classification_report(y_test, y_pred_lda))

# Genera previsioni da ciascun modello
predictions = {
    "Random Forest": y_pred_rf,
    "SVC": y_pred_svc,
    "Logistic Regression": y_pred_log_reg,
    "Decision Tree": y_pred_dt,
    "Gradient Boosting": y_pred_gb,
    "LDA": y_pred_lda
}

# Crea un DataFrame con le previsioni
predictions_df = pd.DataFrame(predictions)

# Calcola la matrice di correlazione
correlation_matrix = predictions_df.corr()

# Visualizza la heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlazioni tra le Previsioni dei Modelli")
plt.show()

# =========================
# Random Forest Grid Search
# =========================
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10],
    'max_features': [0.3, 0.7, 1],
    'bootstrap': [False],
    'min_samples_split': [2, 3, 10],
    'min_samples_leaf': [2, 3, 10],
    'criterion': ['gini']
}

rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=5,
    scoring='accuracy'
)

rf_grid_search.fit(X_train, y_train)

# Migliori parametri e punteggio
print("Best Random Forest Parameters:", rf_grid_search.best_params_)
print("Best Random Forest CV Accuracy:", rf_grid_search.best_score_)

# Previsioni con il modello ottimizzato
rf_best_model = rf_grid_search.best_estimator_
y_pred_rf_gs = rf_best_model.predict(X_test)

# Valutazione Random Forest
print("Random Forest Accuracy (Test):", accuracy_score(y_test, y_pred_rf_gs))
print("Confusion Matrix (Random Forest):\n", confusion_matrix(y_test, y_pred_rf_gs))
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf_gs))

# =========================
# DT Grid Search
# =========================
dt_param_grid = {
    'max_depth': [3, 5, 10],
    'max_features': [0.3, 0.7, 1],
    'min_samples_split': [2, 3, 10],
    'min_samples_leaf': [2, 3, 10],
}

dt_grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=dt_param_grid,
    cv=5,
    scoring='accuracy'
)

dt_grid_search.fit(X_train, y_train)

# Migliori parametri e punteggio
print("Best DT Parameters:", dt_grid_search.best_params_)
print("Best DT CV Accuracy:", dt_grid_search.best_score_)

# Previsioni con il modello ottimizzato
dt_best_model = dt_grid_search.best_estimator_
y_pred_dt_gs = dt_best_model.predict(X_test)

# Valutazione DT
print("DT Accuracy (Test):", accuracy_score(y_test, y_pred_dt_gs))
print("Confusion Matrix (DT):\n", confusion_matrix(y_test, y_pred_dt_gs))
print("Classification Report (DT):\n", classification_report(y_test, y_pred_dt_gs))


# Modelli individuali
rf_model = RandomForestClassifier(max_depth=10, n_estimators=50, min_samples_leaf=2, min_samples_split=10, max_features=0.3, bootstrap=False, criterion='gini',random_state=42)
dt_model = DecisionTreeClassifier(max_depth=3, max_features=0.7, min_samples_split=2, min_samples_leaf=2, random_state=42)

# Ensemble con VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[('rf', rf_model), ('dt', dt_model)],
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

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Calcolo dei dati per la learning curve
train_sizes, train_scores, test_scores = learning_curve(
    ensemble_model, X_balanced_scaled, y_balanced, cv=5, scoring='accuracy', n_jobs=-1, random_state=42
)

# Calcolare media e deviazione standard
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

sns.set_theme(style="darkgrid")

# Creazione del grafico
plt.figure(figsize=(10, 6))

# Linea e intervallo per i punteggi di training
plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Score", color="tab:blue")
plt.fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="tab:blue"
)

# Linea e intervallo per i punteggi di validazione
plt.plot(train_sizes, test_scores_mean, 'o-', label="Validation Score", color="tab:green")
plt.fill_between(
    train_sizes,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="tab:green"
)

# Etichette e dettagli del grafico
plt.title("Learning Curve for VotingClassifier", fontsize=16)
plt.xlabel("Training Instances", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.6)
plt.show()






