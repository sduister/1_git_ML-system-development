import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ========================================
# ⚙️ Configuration
# ========================================
NUM_ELLIPSES = 1000
NUM_POINTS = 30
LATENT_DIM = 2
EPOCHS = 100
A_MIN, A_MAX = 0.5, 3.0
B_MIN, B_MAX = 0.5, 3.0
INPUT_DIM = 2 * (NUM_POINTS + 1)

# ========================================
# 📏 Geometry calculation
# ========================================
def compute_geometry(points):
    y, z = points[:, 0], points[:, 1]
    area = 0.5 * np.abs(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))
    perimeter = np.sum(np.sqrt(np.sum(np.diff(points, axis=0, append=[points[0]])**2, axis=1)))
    return area, perimeter

# ========================================
# 📦 Data generation
# ========================================
def generate_ellipse(a, b, num_points):
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return np.vstack([np.stack([x, y], axis=1), [a * 1, 0]])

def generate_dataset(n, num_points, a_range, b_range):
    ellipses, targets = [], []
    for _ in range(n):
        a = np.random.uniform(*a_range)
        b = np.random.uniform(*b_range)
        shape = generate_ellipse(a, b, num_points)
        area, perimeter = compute_geometry(shape)
        ellipses.append(shape)
        targets.append([area, perimeter])
    return np.array(ellipses), np.array(targets)

ellipses, targets = generate_dataset(NUM_ELLIPSES, NUM_POINTS, (A_MIN, A_MAX), (B_MIN, B_MAX))

# ========================================
# 📐 Plot Original Ellipses (Overlay)
# ========================================
plt.figure(figsize=(8, 8))
for shape in ellipses:
    plt.plot(shape[:, 0], shape[:, 1], alpha=0.6)
plt.gca().set_aspect("equal")
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Overlay of Random Ellipses")
plt.show()

# ========================================
# 🧼 Normalize
# ========================================
ellipses_normalized = ellipses.copy()
ellipses_normalized[:, :, 0] = (ellipses[:, :, 0] + A_MAX) / (2 * A_MAX)
ellipses_normalized[:, :, 1] = (ellipses[:, :, 1] + B_MAX) / (2 * B_MAX)
X = ellipses_normalized.reshape(NUM_ELLIPSES, -1)
X_tensor = torch.tensor(X, dtype=torch.float32)

# ========================================
# 🧠 Autoencoder: Search Space, Model, Training
# ========================================

# 🔧 Hyperparameter Search Space
hidden_sizes = [32, 64]
num_layers_list = [1, 2]
activation_funcs = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh
}
param_grid = list(itertools.product(hidden_sizes, num_layers_list, activation_funcs.items()))

# 🧠 Autoencoder Definition
def build_autoencoder(input_dim, latent_dim, layer_size, num_layers, activation_fn):
    def layer(in_dim, out_dim): return [nn.Linear(in_dim, out_dim), activation_fn()]
    enc, dec = [], []
    in_dim = input_dim
    for _ in range(num_layers):
        enc += layer(in_dim, layer_size)
        in_dim = layer_size
    enc.append(nn.Linear(layer_size, latent_dim))
    in_dim = latent_dim
    for _ in range(num_layers):
        dec += layer(in_dim, layer_size)
        in_dim = layer_size
    dec.append(nn.Linear(layer_size, input_dim))
    return nn.Sequential(*enc), nn.Sequential(*dec)

# 🔍 Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, targets, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor), batch_size=32, shuffle=False)

# 🧪 Grid Search Training
best_val_loss = float("inf")
best_model = None
best_encoder = None
best_decoder = None
best_config = None
best_train_loss_hist = []
best_val_loss_hist = []

for hidden_size, num_layers, (act_name, act_fn) in param_grid:
    encoder, decoder = build_autoencoder(INPUT_DIM, LATENT_DIM, hidden_size, num_layers, act_fn)
    model = nn.Sequential(encoder, decoder)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb[0]), xb[0])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb[0].size(0)
        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb in val_loader:
                loss = loss_fn(model(xb[0]), xb[0])
                val_loss += loss.item() * xb[0].size(0)
        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)

    if val_loss_history[-1] < best_val_loss:
        best_val_loss = val_loss_history[-1]
        best_model = model
        best_encoder = encoder
        best_decoder = decoder
        best_config = (hidden_size, num_layers, act_name)
        best_train_loss_hist = train_loss_history
        best_val_loss_hist = val_loss_history

print(f"Best config: hidden_size={best_config[0]}, num_layers={best_config[1]}, activation={best_config[2]}")
print(f"Best validation loss: {best_val_loss:.6f}")

# 📉 Plot Autoencoder Loss
plt.figure(figsize=(6, 4))
plt.plot(best_train_loss_hist, label="Train Loss")
plt.plot(best_val_loss_hist, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Training")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ========================================
# 🌲 Surrogate Model (Random Forest)
# ========================================
with torch.no_grad():
    Z = best_encoder(X_train_tensor).cpu().numpy()

scaler_z = MinMaxScaler(feature_range=(-1, 1))
Z_scaled = scaler_z.fit_transform(Z)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(Z_scaled, y_train_scaled)
pred_scaled = rf.predict(Z_scaled)
pred = scaler_y.inverse_transform(pred_scaled)

# ========================================
# 📊 Surrogate scatter plots with metrics
# ========================================
true_area, true_peri = y_train[:, 0], y_train[:, 1]
pred_area, pred_peri = pred[:, 0], pred[:, 1]

# Metrics
r2_area = r2_score(true_area, pred_area)
r2_peri = r2_score(true_peri, pred_peri)
mape_area = mean_absolute_percentage_error(true_area, pred_area) * 100
mape_peri = mean_absolute_percentage_error(true_peri, pred_peri) * 100

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(true_area, pred_area, alpha=0.6)
plt.plot([min(true_area), max(true_area)], [min(true_area), max(true_area)], 'r--')
plt.xlabel("True Area")
plt.ylabel("Predicted Area")
plt.title("Area: True vs Predicted")
plt.grid(True)
plt.text(0.98, 0.02,
         f"$R^2$: {r2_area:.3f}\nMAPE: {mape_area:.2f}%",
         transform=plt.gca().transAxes,
         fontsize=10, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))

plt.subplot(1, 2, 2)
plt.scatter(true_peri, pred_peri, alpha=0.6)
plt.plot([min(true_peri), max(true_peri)], [min(true_peri), max(true_peri)], 'r--')
plt.xlabel("True Perimeter")
plt.ylabel("Predicted Perimeter")
plt.title("Perimeter: True vs Predicted")
plt.grid(True)
plt.text(0.98, 0.02,
         f"$R^2$: {r2_peri:.3f}\nMAPE: {mape_peri:.2f}%",
         transform=plt.gca().transAxes,
         fontsize=10, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))

plt.tight_layout()
plt.show()

# ========================================
# 🧱 Convex Hull Filtering Setup
# ========================================
from scipy.spatial import ConvexHull, Delaunay
import cma

# Build convex hull from scaled latent vectors
hull = ConvexHull(Z_scaled)
delaunay = Delaunay(Z_scaled[hull.vertices])
z_min, z_max = Z_scaled.min(axis=0), Z_scaled.max(axis=0)

# ========================================
# 🎯 CMA-ES Optimization with Soft Perimeter Penalty
# ========================================
PERIMETER_TARGET = 13.5
PENALTY_STRENGTH = 100.0  # Tune this as needed

def penalized_objective(z_scaled):
    # Reject latent points outside the convex hull
    if delaunay.find_simplex([z_scaled]) < 0:
        return 1e6

    # De-normalize latent vector
    z = scaler_z.inverse_transform([z_scaled])[0]
    z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)

    # Decode and de-normalize coordinates
    with torch.no_grad():
        decoded = best_decoder(z_tensor).numpy().reshape(NUM_POINTS + 1, 2)
    decoded[:, 0] = (decoded[:, 0] * 2 * A_MAX) - A_MAX
    decoded[:, 1] = (decoded[:, 1] * 2 * B_MAX) - B_MAX

    # Evaluate geometry
    area, perimeter = compute_geometry(decoded)

    # Apply soft constraint as penalty term
    perimeter_penalty = ((perimeter - PERIMETER_TARGET) / PERIMETER_TARGET) ** 2
    total_loss = -area + PENALTY_STRENGTH * perimeter_penalty

    return total_loss

# ========================================
# 🧠 CMA-ES Optimization
# ========================================
x0 = np.mean(Z_scaled, axis=0)
sigma0 = 0.2
opts = {
    "bounds": [z_min.tolist(), z_max.tolist()],
    "popsize": 20,
    "tolfun": 1e-6,
    "maxfevals": 10000,
    "verb_disp": 1
}
es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
es.optimize(penalized_objective)

# ========================================
# 🔓 Decode best latent vector
# ========================================
best_z_scaled = es.result.xbest
best_z = scaler_z.inverse_transform([best_z_scaled])[0]
z_tensor = torch.tensor(best_z, dtype=torch.float32).unsqueeze(0)
decoded = best_decoder(z_tensor).detach().numpy().reshape(NUM_POINTS + 1, 2)

# ✅ De-normalize decoded coordinates
decoded[:, 0] = (decoded[:, 0] * 2 * A_MAX) - A_MAX
decoded[:, 1] = (decoded[:, 1] * 2 * B_MAX) - B_MAX
area, perimeter = compute_geometry(decoded)

# ========================================
# 🖼️ Plot optimized shape
# ========================================
plt.figure(figsize=(6, 6))
plt.plot(decoded[:, 0], decoded[:, 1], color='green')
plt.gca().set_aspect("equal")
plt.title(f"Optimized Shape (Max Area, Perimeter ≈ {perimeter:.2f})")
plt.grid(True)
plt.tight_layout()
plt.show()