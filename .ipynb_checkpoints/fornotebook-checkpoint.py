# Using Lending Club Loan Data,
# https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, recall_score
import matplotlib.pyplot as plt

# Load sample dataset
df = pd.read_csv('loan_sample.csv', low_memory=False)

# Load sample dataset
df = pd.read_csv('loan_sample.csv', low_memory=False)

# Filter statuses
target_statuses = ['Fully Paid', 'Charged Off', 'Default']
df_filtered = df[df['loan_status'].isin(target_statuses)].copy()

# Encode target
df_filtered['default'] = df_filtered['loan_status'].isin(['Charged Off', 'Default']).astype(int)

# Class distribution code
counts = df_filtered['default'].value_counts()
percent = counts / counts.sum() * 100

for cls in counts.index:
    print(f"Class {cls}: {counts[cls]} ({percent[cls]:.2f}%)")

# Variable selection code
selected_vars = [
    'annual_inc',
    'application_type',
    'dti',
    'home_ownership',
    'purpose',
    'default',  # target
    'issue_d',  # for time-based splitting
]
df_model = df_filtered[selected_vars].copy()

# Numeric variables description code
df_model.describe()

# Missing values code
missing_counts = df_model.isna().sum().sort_values(ascending=False)
total_rows = len(df_model)

missing_vars = missing_counts[missing_counts > 0]

if len(missing_vars) > 0:
    print("Variables with missing values:")
    for col, count in missing_vars.items():
        percentage = (count / total_rows) * 100
        print(f"{col}: {count} ({percentage:.2f}%)")
else:
    print("No missing values found")

# Replace missing values code
numeric_cols = df_model.select_dtypes(include=['number']).columns
df_model[numeric_cols] = df_model[numeric_cols].fillna(df_model[numeric_cols].median())

# One hot encoding code
categorical_cols = ['purpose', 'home_ownership', 'application_type']
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# Time-based Train/test split
df_sorted = df_model.sort_values('issue_d')

split_point = int(0.8 * len(df_sorted))

X_train = df_sorted.iloc[:split_point].drop(['default', 'issue_d'], axis=1)
y_train = df_sorted.iloc[:split_point]['default']
X_test = df_sorted.iloc[split_point:].drop(['default', 'issue_d'], axis=1)
y_test = df_sorted.iloc[split_point:]['default']

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train model and make predictions
model = LogisticRegression(class_weight=None)
model.fit(X_train_scaled, y_train)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.2).astype(int)


# Calculate recall
recall_0 = recall_score(y_test, y_pred, pos_label=0)
recall_1 = recall_score(y_test, y_pred, pos_label=1)

print(f"Class 0 recall: {recall_0:.2f}\n"
      f"Class 1 recall: {recall_1:.2f}")


# AUC and Gini
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC: {auc:.3f}")

gini = 2 * auc - 1
print(f"Gini coefficient: {gini:.3f}")


# ROC plot
plt.figure(figsize=(8,6))
# plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.3f})')

# plot KS point
ks_index = (tpr - fpr).argmax()
plt.scatter(fpr[ks_index], tpr[ks_index], color='red', s=30, label='KS point', zorder=5)

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# KS point
ks = max(tpr - fpr)
print(f'KS: {ks:.3f}')


# PSI code details
# Predicted probabilities for the training set
y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]

n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)

df_PSI_test = pd.DataFrame({'y_pred_proba': y_pred_proba})

df_PSI_test['bin'] = pd.cut(df_PSI_test['y_pred_proba'], bins=bin_edges, include_lowest=True)

df_test_bins = (
    df_PSI_test.groupby('bin')
    .agg(count=('y_pred_proba', 'count'))
    .reset_index()
)

df_test_bins['fraction_test'] = df_test_bins['count'] / df_test_bins['count'].sum()

df_PSI_train = pd.DataFrame({'y_pred_proba_train': y_pred_proba_train})

df_PSI_train['bin'] = pd.cut(df_PSI_train['y_pred_proba_train'], bins=bin_edges, include_lowest=True)

df_train_bins = (
    df_PSI_train.groupby('bin')
    .agg(count=('y_pred_proba_train', 'count'))
    .reset_index()
)

df_train_bins['fraction_train'] = df_train_bins['count'] / df_train_bins['count'].sum()

def calculate_psi(train_frac, test_frac):
    """
    train_frac: array of fractions per bin in the baseline population
    test_frac: array of fractions per bin in the new population
    """
    epsilon = 1e-8  # to avoid division by zero
    psi_values = (train_frac - test_frac) * np.log((train_frac + epsilon) / (test_frac + epsilon))
    return np.sum(psi_values)

psi = calculate_psi(df_train_bins['fraction_train'], df_test_bins['fraction_test'])
print(f'PSI ({n_bins} bins): {psi:.3f}')


# PSI plot code
plt.figure(figsize=(8,6))
x_mid = df_test_bins['bin'].apply(lambda x: x.mid).to_numpy()
plt.plot(x_mid, df_train_bins['fraction_train'].to_numpy(), label='Train', marker='o')
plt.plot(x_mid, df_test_bins['fraction_test'].to_numpy(), label='Test', marker='o')

plt.xticks(rotation=45)
plt.xticks(np.arange(0, 1.01, 0.1))  # ticks at 0.0, 0.1, 0.2, ..., 1.0
plt.xlabel("Probability bin")
plt.ylabel("Fraction of samples")
plt.title("Distribution of predicted probabilities (Train vs Test)")
plt.legend()
plt.tight_layout()
plt.show()

# Calibration plot code
df_calibration = pd.DataFrame({
    'y_test': y_test,
    'y_pred_proba': y_pred_proba
})

n_bins_cali = 10
bin_edges = np.linspace(0, 1, n_bins_cali + 1)

df_calibration["bin"] = pd.cut(df_calibration['y_pred_proba'], bins=bin_edges, include_lowest=True)
calibration_table = df_calibration.groupby('bin').agg(
    avg_pred=('y_pred_proba', 'mean'),          # Average predicted PD in the bin
    obs_default_rate=('y_test', 'mean'),        # Fraction of defaults in the bin
).reset_index() # make the bins a column instead of index, and use standard index

plt.figure(figsize=(8,6))
plt.plot([0,1], [0,1], 'k--', label='Perfect calibration')
plt.plot(calibration_table['avg_pred'].to_numpy(), calibration_table['obs_default_rate'].to_numpy(), marker='o', label='Model')
plt.xlabel('Average Predicted PD')
plt.ylabel('Observed Default Rate')
plt.title('Calibration Plot')
plt.legend()
plt.show()

# feature importance code
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0].round(3)
}).sort_values('Coefficient', key=abs, ascending=False)
top_n_features = 5
feature_importance.head(top_n_features)

top_features = feature_importance.head(top_n_features)
top_features.style.hide(axis='index')
