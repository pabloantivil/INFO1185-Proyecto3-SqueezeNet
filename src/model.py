"""
Módulo para la creación del modelo ShuffleNet con Transfer Learning.
Incluye la carga del modelo preentrenado y el clasificador simple.

Autor: Benja
Proyecto: INFO1185 - Transfer Learning con ShuffleNet
Curso: INFO1185
Año: 2025
"""

import torch
import torch.nn as nn
from torchvision import models


class ShuffleNetSimple(nn.Module):
    """
    ShuffleNet con clasificador simple (Versión 1).
    
    Características:
    - Modelo base: ShuffleNet V2 preentrenado en ImageNet
    - Clasificador: Una sola capa Fully Connected
    - SIN BatchNorm
    - SIN Dropout
    """
    
    def __init__(self, num_classes=5, pretrained=True, freeze_features=True):
        """
        Inicializa el modelo ShuffleNet con clasificador simple.
        
        Args:
            num_classes (int): Número de clases de salida (default: 5)
            pretrained (bool): Si cargar pesos preentrenados de ImageNet (default: True)
            freeze_features (bool): Si congelar las capas convolucionales (default: True)
        """
        super(ShuffleNetSimple, self).__init__()
        
        # Cargar ShuffleNet V2 preentrenado en ImageNet
        print("🔄 Cargando ShuffleNet V2 preentrenado...")
        
        # Usar weights parameter (nuevo API de torchvision >= 0.13)
        try:
            if pretrained:
                self.shufflenet = models.shufflenet_v2_x1_0(
                    weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
                )
            else:
                self.shufflenet = models.shufflenet_v2_x1_0(weights=None)
        except:
            # Fallback para versiones antiguas de torchvision
            self.shufflenet = models.shufflenet_v2_x1_0(pretrained=pretrained)
        
        print("✅ ShuffleNet V2 cargado exitosamente!")
        
        # Congelar o no las capas convolucionales (feature extractor)
        if freeze_features:
            print("❄️  Congelando capas convolucionales (feature extractor)...")
            for param in self.shufflenet.parameters():
                param.requires_grad = False
            print("✅ Capas convolucionales congeladas!")
        else:
            print("🔥 Capas convolucionales entrenable (fine-tuning completo)")
        
        # Obtener el tamaño de entrada del clasificador original
        # En ShuffleNet V2 x1.0, la última capa conv produce 1024 features
        in_features = self.shufflenet.fc.in_features
        
        # 🎯 VERSIÓN 1: CLASIFICADOR SIMPLE
        # Solo una capa Fully Connected
        # SIN BatchNorm
        # SIN Dropout
        self.shufflenet.fc = nn.Linear(in_features, num_classes)
        
        print(f"\n🎯 CLASIFICADOR SIMPLE (Versión 1) creado:")
        print(f"   - Input features: {in_features}")
        print(f"   - Output classes: {num_classes}")
        print(f"   - Capas: 1 Linear")
        print(f"   - BatchNorm: NO")
        print(f"   - Dropout: NO")
    
    def forward(self, x):
        """
        Forward pass del modelo.
        
        Args:
            x (torch.Tensor): Tensor de entrada [batch_size, 3, 224, 224]
        
        Returns:
            torch.Tensor: Logits de salida [batch_size, num_classes]
        """
        return self.shufflenet(x)
    
    def get_trainable_params(self):
        """
        Obtiene los parámetros entrenables del modelo.
        
        Returns:
            list: Lista de parámetros que requieren gradiente
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self):
        """
        Cuenta los parámetros del modelo.
        
        Returns:
            dict: Diccionario con total, entrenables y congelados
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }
    
    def print_model_info(self):
        """
        Imprime información detallada del modelo.
        """
        params = self.count_parameters()
        print("\n" + "="*60)
        print("📊 INFORMACIÓN DEL MODELO")
        print("="*60)
        print(f"Parámetros totales:      {params['total']:,}")
        print(f"Parámetros entrenables:  {params['trainable']:,}")
        print(f"Parámetros congelados:   {params['frozen']:,}")
        print(f"Porcentaje entrenable:   {params['trainable']/params['total']*100:.2f}%")
        print("="*60)


def load_shufflenet_simple(num_classes=5, pretrained=True, freeze_features=True):
    """
    Función de conveniencia para cargar el modelo ShuffleNet simple.
    
    Args:
        num_classes (int): Número de clases (default: 5)
        pretrained (bool): Si usar pesos preentrenados (default: True)
        freeze_features (bool): Si congelar feature extractor (default: True)
    
    Returns:
        ShuffleNetSimple: Modelo listo para entrenar
    """
    model = ShuffleNetSimple(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_features=freeze_features
    )
    
    model.print_model_info()
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 CREACIÓN DEL MODELO SHUFFLENET - VERSIÓN SIMPLE")
    print("=" * 60)
    
    # Configuración
    NUM_CLASSES = 5  # jalapeño, chili pepper, carrot, corn, cucumber
    
    # Crear modelo con clasificador simple
    print("\n🏗️  Creando modelo con clasificador simple...")
    model = load_shufflenet_simple(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_features=True
    )
    
    # Probar forward pass
    print("\n🧪 Probando forward pass...")
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch de 1 imagen
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Forward pass exitoso!")
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Logits: {output.squeeze().tolist()}")
    
    # Verificar que solo el clasificador es entrenable
    print("\n🔍 Verificando parámetros entrenables...")
    trainable_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
    
    print(f"✅ Capas entrenables: {trainable_layers}")
    
    print("\n" + "=" * 60)
    print("✅ MODELO CREADO EXITOSAMENTE!")
    print("=" * 60)
