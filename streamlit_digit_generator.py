
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Generator model (same as training)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        input_dim = noise_dim + num_classes
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), dim=1)
        img = self.model(x)
        return img.view(img.size(0), *self.img_shape)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("models/cgan_generator.pth", map_location=device))
generator.eval()

# Streamlit app
st.title("MNIST Handwritten Digit Generator")
digit = st.selectbox("Select a digit to generate", list(range(10)))
generate_button = st.button("Generate Images")

if generate_button:
    z = torch.randn(5, 100).to(device)
    labels = torch.tensor([digit]*5, dtype=torch.long).to(device)
    with torch.no_grad():
        gen_imgs = generator(z, labels).cpu()
    gen_imgs = (gen_imgs + 1) / 2  # unnormalize from [-1, 1] to [0, 1]

    grid = make_grid(gen_imgs, nrow=5, normalize=True).permute(1, 2, 0).numpy()
    st.image(grid, caption=f"Generated images for digit {digit}", use_column_width=True)
