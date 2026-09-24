import pandas as pd
import torch
import torch.nn as nn

# Load data
data = pd.read_csv("data/processed/clean_data.csv")

X = torch.tensor(
    data[["SST", "SSS"]].values,
    dtype=torch.float32
)


# Define the model
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


model = OceanModel()

# Loss and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(500):
    output = model(X)

    loss = loss_function(output, X)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/500, Loss: {loss.item():.6f}")


# Save the trained model
torch.save(model.state_dict(), "models/ocean_model.pth")

print("Training completed!")
print("Model saved to models/ocean_model.pth")