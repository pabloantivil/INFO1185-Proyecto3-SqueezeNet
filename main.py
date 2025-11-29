"""
Script principal del proyecto ShuffleNet Transfer Learning.
Punto de entrada del proyecto.

Autor: Benja
Proyecto: INFO1185 - Transfer Learning con ShuffleNet
Curso: INFO1185
Año: 2025

Uso:
    python main.py
"""

import sys
import os

# Agregar src/ al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from src.data_preparation import get_data_loaders
from src.model import load_shufflenet_simple


def main():
    """
    Función principal del proyecto.
    Configura datos, modelo y parámetros de entrenamiento.
    """
    print("=" * 70)
    print("🚀 PROYECTO SHUFFLENET - TRANSFER LEARNING")
    print("   Clasificación de 5 Clases de Vegetales")
    print("=" * 70)
    
    # ==========================================
    # 1️⃣ CONFIGURACIÓN
    # ==========================================
    print("\n📋 CONFIGURACIÓN DEL PROYECTO")
    print("-" * 70)
    
    # Parámetros
    DATA_DIR = "./archive"
    NUM_CLASSES = 5         # jalepeno, chilli pepper, carrot, corn, cucumber
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    
    # Dispositivo (GPU si está disponible, sino CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Dispositivo: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"✅ Clases: {NUM_CLASSES}")
    print(f"✅ Batch size: {BATCH_SIZE}")
    print(f"✅ Learning rate: {LEARNING_RATE}")
    print(f"✅ Épocas: {NUM_EPOCHS}")
    
    # ==========================================
    # 2️⃣ PREPARACIÓN DE DATOS
    # ==========================================
    print("\n" + "=" * 70)
    print("📦 PASO 1: PREPARACIÓN DE DATOS")
    print("=" * 70)
    
    try:
        train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE
        )
        print("✅ Datos preparados exitosamente!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: No se encontró el directorio '{DATA_DIR}'")
        print("\n📝 INSTRUCCIONES:")
        print("   1. Asegúrate de que la carpeta 'archive' esté en la raíz del proyecto")
        print("   2. Dentro debe tener: train/, validation/, test/")
        return None, None, None, None, None, None, None
    
    # ==========================================
    # 3️⃣ CREACIÓN DEL MODELO
    # ==========================================
    print("\n" + "=" * 70)
    print("🤖 PASO 2: CARGA DE SHUFFLENET PREENTRENADO")
    print("=" * 70)
    
    model = load_shufflenet_simple(
        num_classes=num_classes,
        pretrained=True,
        freeze_features=True  # Congelar feature extractor
    )
    
    # Mover modelo al dispositivo
    model = model.to(device)
    print(f"✅ Modelo movido a {device}")
    
    # ==========================================
    # 4️⃣ CONFIGURACIÓN DEL ENTRENAMIENTO
    # ==========================================
    print("\n" + "=" * 70)
    print("⚙️  PASO 3: CONFIGURACIÓN DEL ENTRENAMIENTO")
    print("=" * 70)
    
    # Función de pérdida
    criterion = nn.CrossEntropyLoss()
    print("✅ Loss function: CrossEntropyLoss")
    
    # Optimizador (solo para parámetros entrenables)
    optimizer = optim.Adam(model.get_trainable_params(), lr=LEARNING_RATE)
    print(f"✅ Optimizer: Adam (lr={LEARNING_RATE})")
    
    # Scheduler (opcional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    print("✅ Scheduler: StepLR (step=5, gamma=0.1)")
    
    # ==========================================
    # 5️⃣ RESUMEN FINAL
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE LA CONFIGURACIÓN")
    print("=" * 70)
    
    print("\n🔵 DATASET:")
    print("   ✔ Dataset pre-dividido en train/val/test")
    print("   ✔ 5 clases seleccionadas:")
    for i, cls in enumerate(class_names, 1):
        print(f"      {i}. {cls}")
    print("   ✔ Transformaciones de entrenamiento:")
    print("     • Resize a 224×224")
    print("     • Random horizontal flip")
    print("     • Random rotation (±15°)")
    print("     • ColorJitter (brillo, contraste, saturación)")
    print("     • Normalización de ImageNet")
    print("   ✔ Transformaciones de val/test:")
    print("     • Resize a 224×224")
    print("     • Normalización de ImageNet")
    print("     • SIN data augmentation")
    
    print("\n🔵 MODELO SHUFFLENET:")
    print("   ✔ Base: ShuffleNet V2 x1.0 preentrenado en ImageNet")
    print("   ✔ Feature extractor: CONGELADO")
    print("   ✔ Clasificador: VERSIÓN 1 - SIMPLE")
    print("     • 1 capa Fully Connected")
    print("     • SIN BatchNorm")
    print("     • SIN Dropout")
    
    print("\n🔵 CONFIGURACIÓN DE ENTRENAMIENTO:")
    print(f"   ✔ Loss: CrossEntropyLoss")
    print(f"   ✔ Optimizer: Adam")
    print(f"   ✔ Learning rate: {LEARNING_RATE}")
    print(f"   ✔ Scheduler: StepLR")
    print(f"   ✔ Épocas: {NUM_EPOCHS}")
    print(f"   ✔ Dispositivo: {device}")
    
    print("\n" + "=" * 70)
    print("✅ CONFIGURACIÓN COMPLETA - LISTO PARA ENTRENAR")
    print("=" * 70)
    
    # ==========================================
    # 6️⃣ PRUEBA DE INFERENCIA
    # ==========================================
    if train_loader is not None:
        print("\n" + "=" * 70)
        print("🧪 PRUEBA DE INFERENCIA")
        print("=" * 70)
        
        model.eval()
        with torch.no_grad():
            # Obtener un batch de prueba
            images, labels = next(iter(train_loader))
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            print(f"✅ Batch procesado: {images.shape}")
            print(f"   - Predicciones: {predicted[:5].cpu().numpy()}")
            print(f"   - Labels reales: {labels[:5].cpu().numpy()}")
            print(f"   - Accuracy en batch: {(predicted == labels).sum().item() / len(labels) * 100:.2f}%")
            
            # Mostrar mapeo de clases
            print(f"\n📋 Mapeo de clases (índice → nombre):")
            for i, cls in enumerate(class_names):
                print(f"   {i} → {cls}")
    else:
        print("\n⚠️  No hay datos disponibles para prueba de inferencia.")
        print("   Asegúrate de que la carpeta 'archive' esté en la raíz del proyecto.")
    
    print("\n" + "=" * 70)
    print("✅ PROYECTO LISTO PARA ENTRENAMIENTO")
    print("=" * 70)
    
    return model, train_loader, val_loader, test_loader, criterion, optimizer, device


if __name__ == "__main__":
    # Ejecutar pipeline completo
    model, train_loader, val_loader, test_loader, criterion, optimizer, device = main()
