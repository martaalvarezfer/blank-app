import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
import os

# --- Variables y configuraciones ---
num_classes = 10
# Lista de nombres de clases (ajusta según tu problema)
classnames = [f"Clase {i}" for i in range(num_classes)]
Images_size = 224
Images_types = ['jpg', 'jpeg', 'png']
Disp_Models = ["Modelo A", "Modelo B"]  # Opciones dummy

# --- Definición de la arquitectura de la CNN ---
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, base_model, num_classes, unfreezed_layers=0):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Congelar los parámetros del modelo base
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Descongelar las últimas capas si se requiere
        if unfreezed_layers > 0:
            for layer in list(self.base_model.children())[-unfreezed_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Nueva capa fully connected personalizada
        self.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

        # Reemplazar la capa fc original por una identidad
        self.base_model.fc = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- Dataset personalizado para la imagen cargada ---
class CustomImageDataset(Dataset):
    def __init__(self, image, transform=None):
        self.image = image
        self.transform = transform

    def __len__(self):
        return 1  # Solo una imagen

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.image)
        else:
            image = self.image
        # Etiqueta dummy (no se utiliza en la inferencia)
        label = 0
        return image, label

# --- Función principal de la app ---
def main():
    # Configuración de la página
    st.set_page_config(page_title="ML2 - CNN", layout="centered")
    st.title("Clasificación de Imágenes con CNNs")
    
    # Mensaje de bienvenida y explicación
    with st.container():
        st.markdown("""
            ¡Bienvenido!  
            Actualmente, <span style='color:red;'>estamos en construcción</span> para clasificar imágenes.
            Pasos:
            1. Sele
