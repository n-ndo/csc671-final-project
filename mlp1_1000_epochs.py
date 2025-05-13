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
from matplotlib.ticker import FuncFormatter

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

#Model 1
class CarPriceMLP(nn.Module):
    def __init__(self, input_size):
        super(CarPriceMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, X_train, y_train, X_val, y_val, epochs=100):
    train_losses, val_losses = [], []
    train_mae_list, val_mae_list = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # Metrics
        train_mae = mean_absolute_error(y_train.detach().numpy(), output.detach().numpy())

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            val_mae = mean_absolute_error(y_val.numpy(), val_output.numpy())

        # Record losses
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_mae_list.append(train_mae)
        val_mae_list.append(val_mae)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}, "
                  f"Train MAE = {train_mae:.2f}, Val MAE = {val_mae:.2f}")
    best_val_mae = min(val_mae_list)
    print(f"Best Val MAE = ${best_val_mae:,.2f}")
    return train_losses, val_losses, train_mae_list, val_mae_list, best_val_mae

# Initialize model
input_size = X_train_tensor.shape[1]
model = CarPriceMLP(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Run training
train_losses, val_losses, train_mae_list, val_mae_list, best_val_mae = train_model(
    model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, epochs=1000
)
# Plotting
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss (MSE)')
plt.plot(val_losses, label='Val Loss (MSE)')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()

# MAE Plot
plt.subplot(1, 2, 2)
plt.plot(train_mae_list, label='Train MAE')
plt.plot(val_mae_list, label='Val MAE')
plt.title("MAE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend()

plt.tight_layout()
plt.show()

# Predicted Price vs Actual Price Plot
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze().numpy()
    actuals = y_test_tensor.squeeze().numpy()

plt.figure(figsize=(7, 6))
plt.scatter(actuals, predictions, alpha=0.5)
plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Predicted vs Actual Car Prices Test Set (MAE = ${best_val_mae:,.2f})")

def format_func(x, _): return f'{int(x):,}'
plt.xticks(np.arange(0, 3100000, 500000))
plt.yticks(np.arange(0, 3100000, 500000))
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))

plt.grid(True)
plt.tight_layout()
plt.show()
