import torch
import torch.nn as nn


# Same model architecture used during training
class OceanModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Create model
model = OceanModel()

# Load trained weights
model.load_state_dict(
    torch.load("models/ocean_model.pth", weights_only=True)
)

model.eval()


# Example ocean data: [SST, SSS]
input_data = torch.tensor(
    [[25.0, 35.0]],
    dtype=torch.float32
)


# Make prediction
with torch.no_grad():
    prediction = model(input_data)


print("Input SST and SSS:")
print(input_data)

print("\nModel output:")
print(prediction)