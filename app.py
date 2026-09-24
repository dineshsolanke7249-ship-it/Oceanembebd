import streamlit as st
import torch
import torch.nn as nn


# -----------------------------
# AI Model
# -----------------------------
class DeepOceanModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


# Load trained model
model = DeepOceanModel()

model.load_state_dict(
    torch.load(
        "models/deep_ocean_model.pth",
        weights_only=True
    )
)

model.eval()


# -----------------------------
# Application interface
# -----------------------------
st.set_page_config(
    page_title="Ocean Deep Temperature Predictor",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Ocean Deep Temperature Predictor")

st.write(
    "Predict ocean temperature at different depths "
    "using surface ocean conditions."
)

st.divider()


# -----------------------------
# Input section
# -----------------------------
st.subheader("🌍 Ocean Location and Surface Data")

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input(
        "Latitude",
        value=10.0
    )

    longitude = st.number_input(
        "Longitude",
        value=70.0
    )

with col2:
    sst = st.number_input(
        "Surface Temperature (°C)",
        value=28.5
    )

    sss = st.number_input(
        "Surface Salinity",
        value=35.2
    )


st.divider()


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Ocean Temperature", type="primary"):

    results = []

    for depth in range(100, 1001, 50):

        input_data = torch.tensor(
            [[
                latitude,
                longitude,
                sst,
                sss,
                float(depth)
            ]],
            dtype=torch.float32
        )

        with torch.no_grad():
            prediction = model(input_data)

        temperature = prediction.item()

        results.append(
            (depth, temperature)
        )


    # -------------------------
    # Results
    # -------------------------
    st.subheader("🌊 Predicted Temperature")

    for depth, temperature in results:

        st.write(
            f"**{depth} m → {temperature:.2f} °C**"
        )