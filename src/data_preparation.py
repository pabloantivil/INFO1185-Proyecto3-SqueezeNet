"""
Módulo de preparación de datos para el proyecto ShuffleNet.
Maneja la carga, filtrado y transformación del dataset.

Autor: Benja
Proyecto: INFO1185 - Transfer Learning con ShuffleNet
Curso: INFO1185
Año: 2025
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os


class DataPreparation:
    """
    Clase para preparar y cargar el dataset.
    Filtra 5 clases específicas: jalepeno, chilli pepper, carrot, corn, cucumber.
    """
    
    def __init__(self, data_dir="./archive", batch_size=32):
        """
        Inicializa el preparador de datos.
        
        Args:
            data_dir (str): Directorio raíz del dataset (default: ./archive)
            batch_size (int): Tamaño del batch para los DataLoaders
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Las 5 clases que necesitamos
        self.selected_classes = [
            'jalepeno',
            'chilli pepper',
            'carrot',
            'corn',
            'cucumber'
        ]
        
        # Rutas de train, val y test
        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'validation')
        self.test_dir = os.path.join(data_dir, 'test')
    
    def get_train_transforms(self):
        """
        Transformaciones para entrenamiento con data augmentation.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_val_test_transforms(self):
        """
        Transformaciones para validación y test sin augmentation.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def filter_classes(self, dataset):
        """
        Filtra el dataset para usar solo las 5 clases seleccionadas.
        
        Args:
            dataset: Dataset de PyTorch
            
        Returns:
            Subset con solo las clases seleccionadas
        """
        # Crear mapeo de nombres de clases a índices
        class_to_idx = dataset.class_to_idx
        
        # Obtener índices de las clases que queremos
        selected_indices = [class_to_idx[cls] for cls in self.selected_classes if cls in class_to_idx]
        
        # Filtrar samples que pertenecen a nuestras clases
        filtered_indices = [
            i for i, (path, label) in enumerate(dataset.samples)
            if label in selected_indices
        ]
        
        # Crear subset
        subset = Subset(dataset, filtered_indices)
        
        return subset, len(filtered_indices)
    
    def create_dataloaders(self):
        """
        Crea los DataLoaders para train, validation y test.
        
        Returns:
            tuple: (train_loader, val_loader, test_loader, num_classes, class_names)
        """
        print("="*70)
        print("📦 PREPARANDO DATOS")
        print("="*70)
        
        # Crear datasets con transformaciones
        train_dataset_full = datasets.ImageFolder(
            root=self.train_dir,
            transform=self.get_train_transforms()
        )
        
        val_dataset_full = datasets.ImageFolder(
            root=self.val_dir,
            transform=self.get_val_test_transforms()
        )
        
        test_dataset_full = datasets.ImageFolder(
            root=self.test_dir,
            transform=self.get_val_test_transforms()
        )
        
        print(f"\n📊 Dataset completo:")
        print(f"   - Total de clases: {len(train_dataset_full.classes)}")
        print(f"   - Train: {len(train_dataset_full)} imágenes")
        print(f"   - Val: {len(val_dataset_full)} imágenes")
        print(f"   - Test: {len(test_dataset_full)} imágenes")
        
        # Filtrar solo las 5 clases que necesitamos
        print(f"\n🔍 Filtrando solo las 5 clases requeridas...")
        train_dataset, train_count = self.filter_classes(train_dataset_full)
        val_dataset, val_count = self.filter_classes(val_dataset_full)
        test_dataset, test_count = self.filter_classes(test_dataset_full)
        
        print(f"\n✅ Dataset filtrado (5 clases):")
        print(f"   - Clases: {self.selected_classes}")
        print(f"   - Train: {train_count} imágenes")
        print(f"   - Val: {val_count} imágenes")
        print(f"   - Test: {test_count} imágenes")
        
        # Crear DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"\n📦 DataLoaders creados:")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Train batches: {len(train_loader)}")
        print(f"   - Val batches: {len(val_loader)}")
        print(f"   - Test batches: {len(test_loader)}")
        print("="*70)
        
        return train_loader, val_loader, test_loader, 5, self.selected_classes


# Función de conveniencia
def get_data_loaders(data_dir="./archive", batch_size=32):
    """
    Función rápida para obtener los data loaders.
    
    Args:
        data_dir (str): Directorio del dataset
        batch_size (int): Tamaño de batch
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes, class_names)
    """
    data_prep = DataPreparation(data_dir=data_dir, batch_size=batch_size)
    return data_prep.create_dataloaders()


if __name__ == "__main__":
    print("="*70)
    print("🧪 PRUEBA DE CARGA DE DATOS")
    print("="*70)
    
    # Crear DataLoaders
    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders(
        data_dir="./archive",
        batch_size=32
    )
    
    print(f"\n✅ Todo funcionando correctamente!")
    print(f"   - Número de clases: {num_classes}")
    print(f"   - Nombres de clases: {class_names}")
    
    # Obtener un batch de ejemplo
    images, labels = next(iter(train_loader))
    print(f"\n📦 Batch de ejemplo:")
    print(f"   - Shape de imágenes: {images.shape}")
    print(f"   - Shape de labels: {labels.shape}")
    print(f"   - Labels en el batch: {labels[:5].tolist()}")
    
    print("\n" + "="*70)
    print("✅ DATASET LISTO PARA USAR")
    print("="*70)
