import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
# Step 1: Load data
df = pd.read_csv('./used_cars.csv')
df.head()
# Clean 'price' and 'milage' columns
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df['milage'] = df['milage'].str.replace(',', '').str.replace(' mi.', '').astype(float)
# Drop less useful columns
df = df.drop(columns=['model', 'engine', 'clean_title', 'ext_col', 'int_col'])
# Separate target and features
target_col = 'price'
X = df.drop(columns=[target_col])
y = df[target_col]
# Define feature types
numerical_features = ['model_year', 'milage']
categorical_features = X.select_dtypes(include='object').columns.tolist()
# Pipelines for preprocessing
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])
# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# Fit and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
# Print shapes to confirm
print(f"Train: {X_train_tensor.shape}, Val: {X_val_tensor.shape}, Test: {X_test_tensor.shape}")

# ######################################################################################################

# Model Trained by Anthony Mingus

# Apply log transformation to the target, avoids large MSE values
y = np.log1p(df[target_col])  # log(price + 1)

# Split the data into training, validation, and test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Fit and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Wrap features as sequences of length 1
X_train_seq = X_train_tensor.unsqueeze(1)
X_val_seq = X_val_tensor.unsqueeze(1)

# LSTM model
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout_rate=0.3):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0  # dropout only works if more than one layer
        )
        self.dropout = nn.Dropout(dropout_rate)  # apply after LSTM
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # dropout before fully connected layer
        return self.fc(out)


# Hyperparameters
input_size = X_train_tensor.shape[1]
hidden_size = 32
num_epochs = 1000
batch_size = 64
learning_rate = 0.001

# Dataloaders
train_dataset = torch.utils.data.TensorDataset(X_train_seq, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(X_val_seq, y_val_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

# Initialize
model = LSTMRegressor(input_size, hidden_size, num_layers=2, dropout_rate=0.3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
val_losses = []
val_mae_list = []

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        output = model(X_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_seq)
        val_loss = criterion(val_preds, y_val_tensor).item()
        val_losses.append(val_loss)

        # Reverse the log for MAE
        preds_exp = torch.expm1(val_preds).squeeze().numpy()
        targets_exp = torch.expm1(y_val_tensor).squeeze().numpy()
        val_mae = mean_absolute_error(targets_exp, preds_exp)
        val_mae_list.append(val_mae)

        print(f"Epoch {epoch+1}/{num_epochs}, Val MSE (log-space): {val_loss:.4f}, MAE: ${val_mae:.2f}")


# Plot MSE loss (log-scale)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), val_losses, marker='o')
plt.title("Validation MSE Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)

# Plot MAE (in real dollars)
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_mae_list, marker='o', color='orange')
plt.title("Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE ($)")
plt.grid(True)

plt.tight_layout()
plt.show()

import seaborn as sns

# Original prices
raw_prices = df['price']

# Log-transformed prices
log_prices = np.log1p(raw_prices)

# Plot side-by-side histograms
plt.figure(figsize=(12, 5))

# Raw prices
plt.subplot(1, 2, 1)
sns.histplot(raw_prices, bins=50, kde=True)
plt.title("Raw Price Distribution")
plt.xlabel("Price ($)")
plt.ylabel("Count")

# Log-transformed prices
plt.subplot(1, 2, 2)
sns.histplot(log_prices, bins=50, kde=True, color='orange')
plt.title("Log-Transformed Price Distribution")
plt.xlabel("log(Price + 1)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# Calculate metrics
r2 = r2_score(targets_exp, preds_exp)
mae = mean_absolute_error(targets_exp, preds_exp)

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=targets_exp, y=preds_exp, alpha=0.6)
plt.plot([targets_exp.min(), targets_exp.max()],
         [targets_exp.min(), targets_exp.max()],
         'r--', label='Perfect Prediction')

plt.title(f"Predicted vs Actual Car Prices\nR² = {r2:.3f}, MAE = ${mae:.0f}")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Final evaluation
model.eval()
with torch.no_grad():
    final_preds = model(X_val_seq).squeeze().numpy()
    y_val_np = y_val_tensor.squeeze().numpy()

    # Reverse the log transform
    preds_exp = np.expm1(final_preds)
    targets_exp = np.expm1(y_val_np)

    mae = mean_absolute_error(targets_exp, preds_exp)
    r2 = r2_score(targets_exp, preds_exp)

    print(f"\nFinal Evaluation on Validation Set:")
    print(f"MAE (in dollars): ${mae:.2f}")
    print(f"R² Score: {r2:.4f}")
