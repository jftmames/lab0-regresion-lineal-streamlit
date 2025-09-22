import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass

# -----------------------------
# Configuración general
# -----------------------------
st.set_page_config(page_title="Lab 0 · Regresión lineal", page_icon="📈", layout="centered")
st.title("Lab 0 · Regresión lineal con PyTorch (Streamlit)")
st.caption("Explora entrenamiento vs. inferencia, datos sintéticos y el ciclo forward → loss → backward → update.")

# -----------------------------
# Utilidades de datos y modelo
# -----------------------------
@dataclass
class DataSpec:
    n_samples: int
    slope: float
    intercept: float
    noise_std: float
    seed: int

def make_data(spec: DataSpec):
    rng = np.random.default_rng(spec.seed)
    x = rng.uniform(-5, 5, size=(spec.n_samples, 1)).astype(np.float32)
    noise = rng.normal(0, spec.noise_std, size=(spec.n_samples, 1)).astype(np.float32)
    y = spec.slope * x + spec.intercept + noise
    return x, y

class LinearReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)
    def forward(self, x):
        return self.lin(x)

# -----------------------------
# Estado de la app
# -----------------------------
if "model" not in st.session_state:
    st.session_state.model = LinearReg()
if "loss_history" not in st.session_state:
    st.session_state.loss_history = []
if "trained" not in st.session_state:
    st.session_state.trained = False

# -----------------------------
# Sidebar: parámetros
# -----------------------------
st.sidebar.header("Datos sintéticos")
n_samples = st.sidebar.slider("n_samples", 20, 1000, 200, step=10)
slope = st.sidebar.slider("Pendiente (m)", -5.0, 5.0, 2.0, step=0.1)
intercept = st.sidebar.slider("Intercepto (b)", -5.0, 5.0, 3.0, step=0.1)
noise_std = st.sidebar.slider("Ruido (σ)", 0.0, 3.0, 0.7, step=0.1)
seed = st.sidebar.number_input("Seed", min_value=0, value=42, step=1)

data_spec = DataSpec(n_samples=n_samples, slope=slope, intercept=intercept, noise_std=noise_std, seed=seed)
X_np, y_np = make_data(data_spec)

st.sidebar.header("Entrenamiento")
optimizer_name = st.sidebar.selectbox("Optimizador", ["SGD", "Adam"], index=1)
learning_rate = st.sidebar.select_slider("Learning rate", options=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1], value=1e-2)
epochs = st.sidebar.slider("Épocas", 1, 1000, 200, step=10)
batch_size = st.sidebar.select_slider("Batch size", options=[len(X_np), 16, 32, 64, 128, 256], value=64)

col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
train_clicked = col_btn1.button("Entrenar")
reset_clicked = col_btn2.button("Reset modelo")
step_clicked = col_btn3.button("1 paso")

if reset_clicked:
    st.session_state.model = LinearReg()
    st.session_state.loss_history = []
    st.session_state.trained = False

# -----------------------------
# Tensores
# -----------------------------
X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)

# -----------------------------
# Entrenamiento (una función para reutilizar)
# -----------------------------
@torch.no_grad()
def predict_line(model, x_min=-5, x_max=5, n=100):
    xs = torch.linspace(x_min, x_max, n).reshape(-1, 1)
    ys = model(xs)
    return xs.numpy(), ys.numpy()

def make_optimizer(model, name, lr):
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    return optim.Adam(model.parameters(), lr=lr)

criterion = nn.MSELoss()


def train_one_epoch(model, X, y, batch_size, optimizer):
    model.train()
    # Mezclar
    perm = torch.randperm(X.shape[0])
    Xs = X[perm]
    ys = y[perm]
    total_loss = 0.0
    for i in range(0, X.shape[0], batch_size):
        xb = Xs[i:i+batch_size]
        yb = ys[i:i+batch_size]
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.shape[0]
    return total_loss / X.shape[0]

# Crear optimizador cada vez que entrenamos (para que cambie con los controles)
optimizer = make_optimizer(st.session_state.model, optimizer_name, learning_rate)

# Botones
if train_clicked:
    st.session_state.trained = True
    with st.spinner("Entrenando..."):
        for _ in range(epochs):
            avg_loss = train_one_epoch(st.session_state.model, X, y, batch_size, optimizer)
            st.session_state.loss_history.append(avg_loss)

if step_clicked:
    st.session_state.trained = True
    avg_loss = train_one_epoch(st.session_state.model, X, y, batch_size, optimizer)
    st.session_state.loss_history.append(avg_loss)

# -----------------------------
# Resultados y visualizaciones
# -----------------------------
col1, col2 = st.columns(2, vertical_alignment="center")
with col1:
    st.subheader("Datos y línea aprendida")
    fig, ax = plt.subplots()
    ax.scatter(X_np, y_np, alpha=0.6)
    xs, ys = predict_line(st.session_state.model, x_min=float(X_np.min())-0.5, x_max=float(X_np.max())+0.5)
    ax.plot(xs, ys)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Pérdida (MSE)")
    fig2, ax2 = plt.subplots()
    if st.session_state.loss_history:
        ax2.plot(np.arange(1, len(st.session_state.loss_history)+1), st.session_state.loss_history)
        ax2.set_xlabel("Iteración/Epoch")
        ax2.set_ylabel("Loss")
    else:
        ax2.text(0.5, 0.5, "Entrena para ver la curva", ha='center', va='center', transform=ax2.transAxes)
    st.pyplot(fig2, clear_figure=True)

# Parámetros aprendidos vs. verdaderos
with st.expander("Parámetros aprendidos (frente a los verdaderos)"):
    w = st.session_state.model.lin.weight.detach().item()
    b = st.session_state.model.lin.bias.detach().item()
    colp1, colp2 = st.columns(2)
    with colp1:
        st.metric("Pendiente aprendida (w)", f"{w:.3f}", delta=f"vs m={slope:.3f}")
    with colp2:
        st.metric("Intercepto aprendido (b)", f"{b:.3f}", delta=f"vs b={intercept:.3f}")

# Inferencia
st.subheader("Inferencia")
new_x = st.number_input("x nuevo", value=1.0, step=0.1)
with torch.no_grad():
    y_hat = st.session_state.model(torch.tensor([[float(new_x)]], dtype=torch.float32)).item()
st.write(f"Predicción del modelo: **y ≈ {y_hat:.3f}**")

# -----------------------------
# Ayuda / Guía didáctica
# -----------------------------
with st.expander("Guía didáctica (qué observar)"):
    st.markdown(
        """
        - **Entrenamiento vs. Inferencia:** Entrenar acumula iteraciones en la curva de *loss*; cambiar `x nuevo` ilustra inferencia.
        - **Efecto del ruido (σ):** A mayor ruido, más difícil ajustar una línea perfecta; observa la *loss* final.
        - **Optimizador y LR:** `Adam` suele converger más rápido que `SGD` con LR pequeños; prueba `1 paso` para ver el efecto incremental.
        - **Batch size:** Cambiarlo altera la estabilidad de la actualización; lotes muy pequeños hacen la curva más ruidosa.
        """
    )

st.divider()
st.caption("Requisitos: streamlit, torch, matplotlib, numpy. Ejecuta con:  `streamlit run app.py`")
