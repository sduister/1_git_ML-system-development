import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# === SETUP ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# === CONFIGURATION ===
NUM_ELLIPSOIDS = 100
THETA_RES = 50
PHI_RES = 50
NUM_SECTIONS = 10
POINTS_PER_SECTION = 50
x_slices = np.linspace(-1.0, 1.0, NUM_SECTIONS)
a_range = (1.0, 2.0)
b_range = (0.5, 2.0)
c_range = (0.5, 2.0)
colors = plt.cm.plasma(np.linspace(0, 1, NUM_ELLIPSOIDS))

# === FUNCTION TO GENERATE TRIANGULAR MESH ELLIPSOID ===
def generate_ellipsoid_mesh(a, b, c, theta_res, phi_res):
    theta = np.linspace(0, np.pi, theta_res)
    phi = np.linspace(0, 2 * np.pi, phi_res)
    theta, phi = np.meshgrid(theta, phi)
    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    faces = []
    for i in range(phi_res - 1):
        for j in range(theta_res - 1):
            idx = i * theta_res + j
            faces.append([idx, idx + 1, idx + theta_res])
            faces.append([idx + 1, idx + theta_res + 1, idx + theta_res])
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)

# === 1. DATA GENERATION ===
a_list = np.random.uniform(*a_range, NUM_ELLIPSOIDS)
b_list = np.random.uniform(*b_range, NUM_ELLIPSOIDS)
c_list = np.random.uniform(*c_range, NUM_ELLIPSOIDS)

ellipsoid_meshes = []
volumes = []
surface_areas = []

for a, b, c in zip(a_list, b_list, c_list):
    mesh = generate_ellipsoid_mesh(a, b, c, THETA_RES, PHI_RES)
    ellipsoid_meshes.append(mesh)
    volumes.append(mesh.volume)
    surface_areas.append(mesh.area)

# === 2. SLICE TO CROSS-SECTIONS ===
sliced_tensors = []
for mesh in ellipsoid_meshes:
    slices = []
    for x_target in x_slices:
        section = mesh.section(plane_origin=[x_target, 0, 0], plane_normal=[1, 0, 0])
        if section is None:
            yz_slice = np.zeros((POINTS_PER_SECTION, 2))
        else:
            try:
                points = np.vstack(section.discrete)
                if len(points) < POINTS_PER_SECTION:
                    pad = POINTS_PER_SECTION - len(points)
                    yz_slice = np.pad(points[:, 1:3], ((0, pad), (0, 0)), mode='edge')
                elif len(points) > POINTS_PER_SECTION:
                    indices = np.round(np.linspace(0, len(points) - 1, POINTS_PER_SECTION)).astype(int)
                    yz_slice = points[indices, 1:3]
                else:
                    yz_slice = points[:, 1:3]
            except Exception:
                yz_slice = np.zeros((POINTS_PER_SECTION, 2))
        slices.append(yz_slice)
    sliced_tensors.append(np.stack(slices))

data_tensor = np.stack(sliced_tensors)
tensor_data = torch.tensor(data_tensor, dtype=torch.float32)
dataset = TensorDataset(tensor_data)

# === 3. VISUALIZATION (YZ, XZ, and SLICED XZ) ===
fig, (ax_yz, ax_xz, ax_xz_slices) = plt.subplots(1, 3, figsize=(18, 6))

# YZ view
for mesh, color in zip(ellipsoid_meshes, colors):
    patches = [Polygon(mesh.vertices[face][:, [1, 2]], closed=True) for face in mesh.faces]
    ax_yz.add_collection(PatchCollection(patches, facecolor=color, alpha=0.05))
ax_yz.set_title("YZ View")
ax_yz.set_xlabel("Y")
ax_yz.set_ylabel("Z")
ax_yz.set_aspect('equal')
ax_yz.autoscale_view()

# XZ view
for mesh, color in zip(ellipsoid_meshes, colors):
    patches = [Polygon(mesh.vertices[face][:, [0, 2]], closed=True) for face in mesh.faces]
    ax_xz.add_collection(PatchCollection(patches, facecolor=color, alpha=0.05))
ax_xz.set_title("XZ View")
ax_xz.set_xlabel("X")
ax_xz.set_ylabel("Z")
ax_xz.set_aspect('equal')
ax_xz.autoscale_view()

# XZ view with slice lines
for mesh, color in zip(ellipsoid_meshes, colors):
    patches = [Polygon(mesh.vertices[face][:, [0, 2]], closed=True) for face in mesh.faces]
    ax_xz_slices.add_collection(PatchCollection(patches, facecolor=color, alpha=0.05))
for x in x_slices:
    ax_xz_slices.axvline(x=x, color='red', linestyle='--', linewidth=1)
ax_xz_slices.set_title("XZ View with Slice Lines")
ax_xz_slices.set_xlabel("X")
ax_xz_slices.set_ylabel("Z")
ax_xz_slices.set_aspect('equal')
ax_xz_slices.autoscale_view()

plt.tight_layout()
plt.show()

# === 4. AUTOENCODER TRAINING ===
latent_dim = 30
hidden_size = 256
activation_name = 'leaky_relu'
num_layers = 1
EPOCHS = 50
batch_size = 16
train_ratio = 0.8

train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

def get_activation(name):
    return {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.2),
        'tanh': nn.Tanh()
    }[name]

def build_autoencoder(latent_dim, hidden_size, activation_name, num_layers):
    act = get_activation(activation_name)
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            convs = [
                nn.Conv2d(2, 16, kernel_size=3, padding=1), act,
                nn.Conv2d(16, 32, kernel_size=3, padding=1), act,
                nn.Flatten()
            ]
            linear_in = 32 * NUM_SECTIONS * POINTS_PER_SECTION
            encoder_layers = [nn.Linear(linear_in, hidden_size), act]
            if num_layers == 2:
                encoder_layers += [nn.Linear(hidden_size, hidden_size), act]
            encoder_layers.append(nn.Linear(hidden_size, latent_dim))

            decoder_layers = [nn.Linear(latent_dim, hidden_size), act]
            if num_layers == 2:
                decoder_layers += [nn.Linear(hidden_size, hidden_size), act]
            decoder_layers += [
                nn.Linear(hidden_size, linear_in), act,
                nn.Unflatten(1, (32, NUM_SECTIONS, POINTS_PER_SECTION)),
                nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), act,
                nn.ConvTranspose2d(16, 2, kernel_size=3, padding=1)
            ]
            self.encoder = nn.Sequential(*convs, *encoder_layers)
            self.decoder = nn.Sequential(*decoder_layers)
        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            z = self.encoder(x)
            out = self.decoder(z)
            return out.permute(0, 2, 3, 1)
    return Autoencoder()

model = build_autoencoder(latent_dim, hidden_size, activation_name, num_layers).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for (batch,) in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for (batch,) in test_loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            val_loss += loss_fn(recon, batch).item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)

# === LOSS CURVE PLOT ===
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", linewidth=2)
plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# === LATENT SPACE EXTRACTION ===
model.eval()
latent_vectors = []

with torch.no_grad():
    for (batch,) in DataLoader(dataset, batch_size=batch_size):
        batch = batch.to(DEVICE)
        z = model.encoder(batch.permute(0, 3, 1, 2))
        latent_vectors.append(z.cpu().numpy())

X_latent = np.concatenate(latent_vectors, axis=0)
y_surface = np.array(surface_areas)

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X_latent, y_surface, test_size=0.2, random_state=42)

# === MANUAL GRID SEARCH (NO CV) ===
param_grid = [
    {'n_estimators': 50, 'max_depth': None, 'min_samples_leaf': 1},
    {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 1},
    {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 2},
    {'n_estimators': 200, 'max_depth': 20, 'min_samples_leaf': 2},
]

best_model = None
best_mse = float('inf')
best_params = None

print("Training Random Forest models...")

for params in tqdm(param_grid):
    model_try = RandomForestRegressor(random_state=42, **params)
    model_try.fit(X_train, y_train)
    val_pred = model_try.predict(X_test)
    mse = mean_squared_error(y_test, val_pred)
    if mse < best_mse:
        best_mse = mse
        best_model = model_try
        best_params = params

# === EVALUATION ===
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("Best RF hyperparameters:", best_params)
print(f"Validation MSE: {best_mse:.4f} | R2: {r2:.4f} | MAPE: {mape:.4f}")

# === PLOT ACTUAL VS PREDICTED ===
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Actual Surface Area")
plt.ylabel("Predicted Surface Area")
plt.title("Surface Area Prediction (Latent Space → RF)")
plt.grid(True)
plt.tight_layout()
plt.show()
