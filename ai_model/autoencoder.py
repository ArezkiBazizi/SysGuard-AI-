"""
Définition de l'Autoencoder SysGuard-AI en PyTorch.

Architecture en sablier : 414 → 207 → 100 → 207 → 414
Activations : ReLU (couches cachées) + Sigmoid (sortie)
Régularisation : Dropout (p=0.2) entre couches cachées

Fichier partagé entre l'entraînement et l'inférence.
"""

import torch
import torch.nn as nn

INPUT_DIM = 414


class SysGuardAutoencoder(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, dropout: float = 0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 207),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(207, 100),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(100, 207),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(207, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
