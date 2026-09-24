import pandas as pd
import torch
import torch.nn as nn

# Load training data
data = pd.read_csv("data/raw/deep_ocean_training.csv")

# Inputs:
# latitude, longitude, SST, SSS, depth
X = torch.tensor(
    data[["latitude", "longitude", "SST", "SSS", "depth"]].values,
    dtype=torch.float32
)

# Target:
# deep-ocean temperature
y = torch.tensor(
    data[["temperature"]].values,
    dtype=torch.float32
)


# Define the model
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


# Create model
model = DeepOceanModel()

# Loss and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the model
for epoch in range(1000):

    prediction = model(X)

    loss = loss_function(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}/1000, "
            f"Loss: {loss.item():.6f}"
        )


# Save the trained model
torch.save(
    model.state_dict(),
    "models/deep_ocean_model.pth"
)

print("\nTraining completed!")
print("Model saved to models/deep_ocean_model.pth")