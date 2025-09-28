import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
import copy
import csv
import pickle
import sys

torch.manual_seed(0)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # Store dimensions
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder with convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=4, stride=2, padding=1
            ),  # Output: (64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),  # Output: (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Calculate the flattened size after the Conv2D layers
        conv_output_dim = self._get_conv_output_dim()

        # Latent space mappings
        self.flatten_mu = nn.Linear(conv_output_dim, latent_dim)
        self.flatten_logvar = nn.Linear(conv_output_dim, latent_dim)

        # Decoder layers with transposed convolutional layers
        self.decode_linear = nn.Linear(latent_dim, conv_output_dim)
        self.decode_conv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # Output: (64, 14, 14)
        self.decode_conv2 = nn.ConvTranspose2d(
            64, 1, kernel_size=4, stride=2, padding=1
        )  # Output: (1, 28, 28)

    def _get_conv_output_dim(self):
        # Pass a dummy tensor through the encoder to get the output dimension after Conv2D layers
        dummy_input = torch.zeros(1, 1, 28, 28)  # Assume 28x28 input for calculation
        output = self.encoder(dummy_input)
        return int(output.numel())  # Total number of elements in the output

    def encode(self, x):
        # Reshape the flattened input back to 28x28
        x = x.view(
            x.size(0), 1, 28, 28
        )  # Reshape to (batch_size, channels, height, width)
        # Pass through the convolutional encoder
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten for the linear layers
        mu = self.flatten_mu(x)
        logvar = self.flatten_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Decode from latent space back to image space
        x = self.decode_linear(z)
        x = x.view(x.size(0), 128, 7, 7)  # Reshape for ConvTranspose layers
        x = F.relu(self.decode_conv1(x))
        reconstruction = torch.sigmoid(self.decode_conv2(x))
        return reconstruction

    def forward(self, x):
        # Forward pass for VAE
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def loss_function(recon_x, x, mu, logvar, beta=1):
    BCE = nn.functional.binary_cross_entropy(
        recon_x.view(-1, 784), x.view(-1, 784), reduction="sum"
    )
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD * beta


# Custom Dataset class
class NPZDataset(Dataset):
    def __init__(self, file_path, has_labels=True, transform=None):
        data = np.load(file_path)
        self.has_labels = has_labels
        self.images = data["data"]
        if has_labels:
            self.labels = data["labels"]
            # print(self.labels[:15])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32).reshape(28, 28) / 255.0
        if self.has_labels:
            label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).float()

        if self.has_labels:
            return image.view(-1), label
        else:
            return image.view(-1)


# Extract means from VAE and train GMM
def extract_latent_means(model, dataloader, device):
    model.eval()
    latent_means = []
    labels = []
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latent_means.append(mu)

            labels.extend(label)
    return torch.cat(latent_means), torch.tensor(labels)
    
def extract_only_means(model, dataloader, device):
    model.eval()
    latent_means = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latent_means.append(mu)

    return torch.cat(latent_means)    


# Hyperparameters
input_dim = 784  # 28x28 for MNIST flattened
hidden_dim = 200
latent_dim = 2
learning_rate = 1e-3
num_epochs = 200
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the Gaussian Mixture Model (GMM)
class GaussianMixtureModel:
    def __init__(
        self, n_components, n_iter=1000, tol=1e-6, device="cpu", class_mapping=None
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.device = device
        self.class_mapping = (
            class_mapping if class_mapping else list(range(n_components))
        )

    def initialize_params(self, X, initial_means):
        n_samples, n_features = X.shape
        self.means = initial_means.to(self.device)
        self.covariances = torch.stack(
            [torch.eye(n_features).to(self.device) for _ in range(self.n_components)]
        )
        self.weights = (
            torch.ones(self.n_components, device=self.device) / self.n_components
        )

    def gaussian(self, X, mean, covariance):
        n = X.shape[1]
        # Add a small value to the diagonal of the covariance matrix for stability
        covariance += 1e-6 * torch.eye(n, device=self.device)
        diff = X - mean
        exponent = -0.5 * torch.sum((diff @ torch.linalg.inv(covariance)) * diff, dim=1)
        return torch.exp(exponent) / torch.sqrt(
            ((2 * torch.pi) ** n) * torch.linalg.det(covariance)
        )

    def fit(self, X, initial_means):
        X = X.to(self.device)
        self.initialize_params(X, initial_means)
        log_likelihoods = []

        for i in range(self.n_iter):
            responsibilities = torch.zeros(
                X.shape[0], self.n_components, device=self.device
            )
            for k in range(self.n_components):
                responsibilities[:, k] = self.weights[k] * self.gaussian(
                    X, self.means[k], self.covariances[k]
                )
            responsibilities /= responsibilities.sum(dim=1, keepdim=True)

            nk = responsibilities.sum(dim=0)
            self.weights = nk / X.shape[0]
            self.means = (responsibilities.T @ X) / nk[:, None]
            for k in range(self.n_components):
                diff = X - self.means[k]
                # Regularize the covariance update as well
                self.covariances[k] = (
                    responsibilities[:, k][:, None] * diff
                ).T @ diff / nk[k] + 1e-6 * torch.eye(diff.shape[1], device=self.device)

            # CORRECTED HERE !!!!
            # Calculate total probability (weighted sum of Gaussian probabilities) for each sample
            weighted_probs = torch.zeros(X.shape[0], device=self.device)
            for k in range(self.n_components):
                weighted_probs += self.weights[k] * self.gaussian(
                    X, self.means[k], self.covariances[k]
                )

            # Compute log-likelihood as the sum of log of these probabilities
            log_likelihood = torch.sum(torch.log(weighted_probs))
            log_likelihoods.append(log_likelihood.item())

            # Check for convergence
            if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                break

        return log_likelihoods

    def predict(self, X):
        X = X.to(self.device)
        responsibilities = torch.zeros(
            X.shape[0], self.n_components, device=self.device
        )
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self.gaussian(
                X, self.means[k], self.covariances[k]
            )
        cluster_indices = responsibilities.argmax(dim=1)

        # Map the cluster indices to class names
        predicted_classes = [
            self.class_mapping[idx] for idx in cluster_indices.cpu().numpy()
        ]
        return predicted_classes


# Define functions for training, reconstruction, and classification
def train_vae_gmm(
    train_loader,
    val_loader,
    vae,
    latent_dim,
    device,
    save_vae_path,
    save_gmm_path,
    n_components=3,
    num_epochs=200,
    learning_rate=1e-3,
):
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state_dict = None

    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        avg_loss = train_loss / len(train_loader.dataset)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict = copy.deepcopy(vae.state_dict())

        print(f"Epoch: {epoch+1}, Loss: {avg_loss}")

    vae.load_state_dict(best_state_dict)
    torch.save(vae.state_dict(), save_vae_path)

    # Extract latent means for GMM initialization
    val_latents, val_labels = extract_latent_means(vae, val_loader, device)

    # Initialize cluster means for each class using validation data
    initial_means = []
    class_mapping = []

    # For each unique label (1, 4, 8), calculate the mean of latent vectors
    for class_id in torch.unique(val_labels):
        class_latents = val_latents[val_labels == class_id]
        initial_means.append(class_latents.mean(dim=0))
        class_mapping.append(class_id.item())  # Map cluster index to class label

    # Stack the means to get the initial cluster centers for the GMM
    initial_means = torch.stack(initial_means)

    # Print class mapping for clarity
    print("Cluster to class mapping:", class_mapping)

    # Train GMM on training data using the initialized means and class mapping
    train_latents = extract_only_means(vae, train_loader, device)
    gmm = GaussianMixtureModel(
        n_components=n_components, device=device, class_mapping=class_mapping
    )
    gmm.fit(train_latents, initial_means)

    with open(save_gmm_path, "wb") as f:
        pickle.dump(gmm, f)
    # torch.save(gmm, save_gmm_path)


def test_reconstruction(vae, test_loader, device, output_file="vae_reconstructed.npz"):
    vae.eval()
    reconstructed_images = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, _, _ = vae(data)
            reconstructed_images.extend(recon_batch.cpu().numpy().reshape(-1, 28, 28))
    np.savez(output_file, data=reconstructed_images)


def test_classifier(vae, gmm, test_loader, device, output_file="vae.csv"):
    vae.eval()
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            mu, _ = vae.encode(data)
            preds = gmm.predict(mu)
            all_preds.extend(preds)
    # pd.DataFrame({"Predicted_Label": all_preds}).to_csv(output_file, index=False)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Predicted_Label"])
        for label in all_preds:
            # print(label, end=", ")
            writer.writerow([label])


def main():
    if len(sys.argv) < 4:
        print("Insufficient arguments provided.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 2

    dataset_path = sys.argv[1]

    if sys.argv[3] == "train":
        if len(sys.argv) < 6:
            print("Not enough arguments for train")
            return
        val_dataset_path = sys.argv[2]
        #   mode = sys.argv[3]
        vae_path = sys.argv[4]
        gmm_path = sys.argv[5]

        train_dataset = NPZDataset(dataset_path, has_labels=False)
        val_dataset = NPZDataset(val_dataset_path)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        vae = VAE(input_dim=784, latent_dim=latent_dim).to(device)
        train_vae_gmm(
            train_loader,
            val_loader,
            vae,
            latent_dim,
            device,
            vae_path,
            gmm_path,
        )
        return

    mode = sys.argv[2]
    vae_path = sys.argv[3]
    gmm_path = sys.argv[4] if len(sys.argv) > 4 else None

    if mode == "test_reconstruction":
        test_dataset = NPZDataset(dataset_path, has_labels=False)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
        vae = VAE(input_dim=784, latent_dim=latent_dim).to(device)
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        test_reconstruction(vae, test_loader, device)

    elif mode == "test_classifier":
        if not gmm_path:
            print("GMM parameters path is required for classification testing.")
            return
        test_dataset = NPZDataset(dataset_path, has_labels=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        vae = VAE(input_dim=784, latent_dim=latent_dim).to(device)
        vae.load_state_dict(torch.load(vae_path, map_location=device))

        with open(gmm_path, "rb") as f:
            gmm = pickle.load(f)

        test_classifier(vae, gmm, test_loader, device)


if __name__ == "__main__":
    main()

