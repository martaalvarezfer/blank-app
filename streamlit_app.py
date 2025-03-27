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
            1. Selecciona el modelo que deseas usar en la parte izquierda  
            2. Sube la imagen  
            3. Verás la predicción para ella
        """, unsafe_allow_html=True)

    # Configuraciones en la barra lateral
    with st.sidebar:
        st.header("Configuraciones")
        _ = st.selectbox(
            "Modelo a Utilizar:",
            Disp_Models,
            help="Selecciona el modelo de CNN (simulación)."
        )

    # Carga de la imagen
    with st.container():
        image_file = st.file_uploader("Cargar Imagen", type=Images_types)

    if image_file is not None:
        with st.spinner('Procesando imagen...'):
            # Cargar la imagen con PIL y asegurar RGB
            image = Image.open(image_file).convert("RGB")
            img_size = Images_size

            # Transformaciones: redimensionar y convertir a tensor
            streamlit_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
                # Si durante el entrenamiento usaste normalización, añádela aquí:
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Crear un Dataset y DataLoader para la imagen cargada
            streamlit_data = CustomImageDataset(image, transform=streamlit_transforms)
            streamlit_loader = DataLoader(streamlit_data, batch_size=1, shuffle=False)

        # --- Cargar el modelo guardado y clasificar la imagen ---
        with st.spinner('Cargando el modelo y clasificando la imagen...'):
            model_path = "models/mi_modelo.pt"  # Ruta del modelo guardado
            if not os.path.exists(model_path):
                st.error("No se encontró el modelo guardado en: " + model_path)
                return

            # Inicializar el modelo; en este ejemplo usamos ResNet18 como base.
            # Si usaste otra arquitectura (p.ej., ResNeXt101), cámbiala aquí.
            base_model = models.resnet18(pretrained=False)
            model = CNN(base_model, num_classes)
            # Cargar pesos en CPU para mayor compatibilidad
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            model.eval()

            # Realizar la inferencia
            with torch.no_grad():
                for img, _ in streamlit_loader:
                    outputs = model(img)
                    _, top_class = torch.max(outputs, dim=1)
                    predicted_label = top_class.item()
                    class_name = classnames[predicted_label]
                    prob = outputs[0][predicted_label].item()

        # Mostrar el resultado
        st.success(f'### Clase predicha: {class_name} (Confianza: {round(prob, 5)})')
        st.image(image, caption='Imagen cargada', use_container_width=True)
    else:
        st.info("Por favor, carga una imagen para realizar la clasificación.")

if __name__ == "__main__":
    main()
