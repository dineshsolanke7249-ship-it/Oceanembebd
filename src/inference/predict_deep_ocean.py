import torch
import torch.nn as nn


# --------------------------------------------------
# Define the same model used during training
# --------------------------------------------------

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


# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

model = DeepOceanModel()

model.load_state_dict(
    torch.load(
        "models/deep_ocean_model.pth",
        weights_only=True
    )
)

model.eval()


# --------------------------------------------------
# Get input data from the user
# --------------------------------------------------

print("======================================")
print(" Deep Ocean Temperature Prediction")
print("======================================")
print()

latitude = float(input("Enter latitude: "))
longitude = float(input("Enter longitude: "))
SST = float(input("Enter surface temperature (SST °C): "))
SSS = float(input("Enter surface salinity (SSS): "))

print()
print("Generating predictions from 100 m to 1000 m...")
print()


# --------------------------------------------------
# Predict temperature at different depths
# --------------------------------------------------

print("--------------------------------------")
print("Depth (m)    Predicted Temperature")
print("--------------------------------------")

for depth in range(100, 1001, 50):

    input_data = torch.tensor(
        [[
            latitude,
            longitude,
            SST,
            SSS,
            float(depth)
        ]],
        dtype=torch.float32
    )

    with torch.no_grad():
        prediction = model(input_data)

    temperature = prediction.item()

    print(
        f"{depth:4d} m       {temperature:6.2f} °C"
    )

print("--------------------------------------")
print()
print("Prediction completed.")
