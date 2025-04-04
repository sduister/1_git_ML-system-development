import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# ========================================
# ⚙️ Configuration
# ========================================
NUM_ELLIPSES = 1000
NUM_POINTS = 100
LATENT_DIM = 2
EPOCHS = 300
A_MIN, A_MAX = 0.5, 3.0
B_MIN, B_MAX = 0.5, 3.0

# ========================================
# 📏 Geometry calculation
# ========================================
def compute_geometry(points):
    y, z = points[:, 0], points[:, 1]
    area = 0.5 * np.abs(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))
    perimeter = np.sum(np.sqrt(np.sum(np.diff(points, axis=0, append=[points[0]])**2, axis=1)))
    return area, perimeter

# ========================================
# 📦 1. DATA GENERATION
# ========================================
def generate_ellipse(a, b, num_points):
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    points = np.stack([x, y], axis=1)
    return np.vstack([points, points[0]])

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
# 📊 Plot overlay of all ellipses
# ========================================
plt.figure(figsize=(6, 6))
for shape in ellipses:
    plt.plot(shape[:, 0], shape[:, 1], alpha=0.3)
plt.gca().set_aspect("equal")
plt.title(f"Overlay of {NUM_ELLIPSES} Training Ellipses")
plt.xlabel("Y")
plt.ylabel("Z")
plt.grid(True)
plt.tight_layout()
plt.show()

# ========================================
# 🧼 Preprocessing
# ========================================
ellipses_normalized = ellipses.copy()
ellipses_normalized[:, :, 0] /= A_MAX
ellipses_normalized[:, :, 1] /= B_MAX
flattened = ellipses_normalized.reshape(NUM_ELLIPSES, -1)

# ========================================
# 🧠 2. AUTOENCODER
# ========================================
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x): return self.model(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, z): return self.model(z)

X = torch.tensor(flattened, dtype=torch.float32)
encoder = Encoder(X.shape[1], LATENT_DIM)
decoder = Decoder(LATENT_DIM, X.shape[1])
model = nn.Sequential(encoder, decoder)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

loss_history = []
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, X)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

# ========================================
# 📉 Plot loss over epochs
# ========================================
plt.figure(figsize=(6, 4))
plt.plot(loss_history, label="Reconstruction Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Loss Over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ========================================
# 🌲 3. SURROGATE MODEL
# ========================================
with torch.no_grad():
    Z = encoder(X).cpu().numpy()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(Z, targets)
pred_targets = rf.predict(Z)

# ========================================
# 📊 4. EVALUATION
# ========================================
true_area = targets[:, 0]
true_peri = targets[:, 1]
pred_area = pred_targets[:, 0]
pred_peri = pred_targets[:, 1]

def eval_metrics(true, pred):
    return {
        "MAPE": mean_absolute_percentage_error(true, pred),
        "MSE": mean_squared_error(true, pred),
        "R²": r2_score(true, pred)
    }

# ========================================
# 📈 5. PLOTS: True vs Predicted
# ========================================
metrics_area = eval_metrics(true_area, pred_area)
metrics_peri = eval_metrics(true_peri, pred_peri)

plt.figure(figsize=(12, 5))

# Area
plt.subplot(1, 2, 1)
plt.scatter(true_area, pred_area, alpha=0.6)
plt.plot([min(true_area), max(true_area)], [min(true_area), max(true_area)], 'r--')
plt.xlabel("True Area")
plt.ylabel("Predicted Area")
plt.title("Area: True vs Predicted")
plt.grid(True)
metrics_text_area = (
    f"MAPE: {metrics_area['MAPE']:.4f}%\n"
    f"MSE: {metrics_area['MSE']:.4f}\n"
    f"R²: {metrics_area['R²']:.4f}"
)
plt.gca().text(0.7, 0.05, metrics_text_area, transform=plt.gca().transAxes,
               fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# Perimeter
plt.subplot(1, 2, 2)
plt.scatter(true_peri, pred_peri, alpha=0.6)
plt.plot([min(true_peri), max(true_peri)], [min(true_peri), max(true_peri)], 'r--')
plt.xlabel("True Perimeter")
plt.ylabel("Predicted Perimeter")
plt.title("Perimeter: True vs Predicted")
plt.grid(True)
metrics_text_peri = (
    f"MAPE: {metrics_peri['MAPE']:.4f} %\n"
    f"MSE: {metrics_peri['MSE']:.4f}\n"
    f"R²: {metrics_peri['R²']:.4f}"
)
plt.gca().text(0.7, 0.05, metrics_text_peri, transform=plt.gca().transAxes,
               fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()
