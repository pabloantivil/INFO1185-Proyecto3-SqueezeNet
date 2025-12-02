# INFO1185-Proyecto3-SqueezeNet

Transfer Learning para Clasificación de Vegetales utilizando **SqueezeNet 1.1** preentrenado en ImageNet.

**Autores:** Benja Espinoza y Pablo Antivil  
**Curso:** INFO1185 - Inteligencia Artificial III  
**Año:** 2025

---

## 🎯 Objetivo

Implementar y comparar **tres variantes de clasificadores** usando transfer learning con SqueezeNet:
- **Versión 1 (Simple)**: Arquitectura básica sin regularización
- **Versión 2A (Extendido)**: 4 capas FC sin BatchNorm/Dropout
- **Versión 2B (Regularizado)**: 4 capas FC con BatchNorm y Dropout (p=0.3)

---

## 🏆 Resultados Obtenidos

| Modelo | Test Accuracy | Val Accuracy | Épocas | Parámetros Entrenables |
|--------|--------------|--------------|--------|----------------------|
| **V1 (Simple)** | **98.00%** 🏆 | 97.87% | 14 | 265,221 |
| **V2A (Sin Reg.)** | 92.00% | 95.74% | 12 | 427,525 |
| **V2B (Con Reg.)** | 94.00% | 97.87% | 19 | 428,293 |

**Hallazgo principal:** El modelo más simple (V1) superó a los complejos, demostrando que con Transfer Learning y datasets pequeños (438 samples), arquitecturas simples pueden ser óptimas.

---

## 🥕 Dataset

**5 Clases:**
- Jalapeño (jalepeno)
- Chili Pepper
- Carrot
- Corn
- Cucumber

**Estructura:**
```
archive/
├── train/          (438 imágenes de las 5 clases)
├── validation/     (47 imágenes)
└── test/           (50 imágenes)
```

---

## 📁 Estructura del Proyecto

```
INFO1185-Proyecto3-SqueezeNet/
├── SqueezeNet_Transfer_Learning.ipynb  # Notebook principal (Jupyter/Colab)
├── archive/                            # Dataset (no versionado)
│   ├── train/          (438 imágenes de las 5 clases)
│   ├── validation/     (47 imágenes)
│   └── test/           (50 imágenes)
├── squeezenet_modelo_final.pth         # Modelo V1 guardado
├── squeezenet_version_2a.pth           # Modelo V2A guardado
├── squeezenet_version_2b.pth           # Modelo V2B guardado
├── requirements.txt                    # Dependencias de Python
├── .gitignore                          # Archivos ignorados por Git
└── README.md                           # Documentación
```

**Nota:** El proyecto fue migrado a un único notebook de Jupyter para facilitar su ejecución en Google Colab.

---

## 🚀 Instalación y Ejecución

### Opción 1: Google Colab (Recomendado)

1. Abrir `SqueezeNet_Transfer_Learning.ipynb` en Google Colab
2. Subir el dataset a Google Drive o usar Kaggle API
3. Ejecutar todas las celdas en orden

### Opción 2: Entorno Local

```bash
# 1. Clonar repositorio
git clone https://github.com/pabloantivil/INFO1185-Proyecto3-SqueezeNet.git
cd INFO1185-Proyecto3-SqueezeNet

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# 3. Instalar PyTorch (CPU o GPU según disponibilidad)
# CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Instalar otras dependencias
pip install -r requirements.txt

# 5. Abrir Jupyter Notebook
jupyter notebook SqueezeNet_Transfer_Learning.ipynb
```

---

## ✅ Estado del Proyecto

### Tareas Completadas
- ✅ Preparación de datos (train/val/test)
- ✅ Data augmentation y normalización ImageNet
- ✅ DataLoaders optimizados (batch_size=32)
- ✅ SqueezeNet 1.1 preentrenado (feature extractor congelado)
- ✅ Implementación de 3 clasificadores personalizados (V1, V2A, V2B)
- ✅ Loop de entrenamiento con Early Stopping
- ✅ Validación y cálculo de métricas (Accuracy, Loss)
- ✅ Evaluación en test set
- ✅ Guardado de modelos (.pth)
- ✅ Análisis comparativo de resultados

### Resultados Finales
- **V1 (Simple):** 98% Test Accuracy (mejor desempeño)
- **V2A (Sin regularización):** 92% Test Accuracy
- **V2B (Con regularización):** 94% Test Accuracy (+2% mejora sobre V2A)

**Conclusión:** Transfer Learning con SqueezeNet demostró excelente generalización. El modelo simple (V1) superó arquitecturas complejas debido al tamaño reducido del dataset (438 muestras) y la calidad de las features preentrenadas.

---

## 🧠 Información del Modelo

### SqueezeNet 1.1 - Transfer Learning

**Arquitectura Base:**
- **Parámetros totales:** ~1.2M (modelo completo)
- **Parámetros congelados:** ~0.7M (feature extractor)
- **Extractor de características:** 512 features
- **Componentes clave:** Fire Modules (squeeze + expand layers)
- **Pretrained:** ImageNet (1000 clases)

**Clasificadores Personalizados:**

1. **Versión 1 (Simple):**
   - Linear(512 → 5)
   - **Parámetros entrenables:** 265,221
   - **Regularización:** Ninguna
   - **Test Accuracy:** 98%

2. **Versión 2A (Extendida sin regularización):**
   - Linear(512 → 256) → ReLU → Linear(256 → 128) → ReLU → Linear(128 → 5)
   - **Parámetros entrenables:** 427,525
   - **Regularización:** Ninguna
   - **Test Accuracy:** 92%

3. **Versión 2B (Con regularización):**
   - Linear(512 → 256) → BatchNorm1d → Dropout(0.3) → ReLU → Linear(256 → 128) → BatchNorm1d → Dropout(0.3) → ReLU → Linear(128 → 5)
   - **Parámetros entrenables:** 428,293
   - **Regularización:** BatchNorm + Dropout (p=0.3)
   - **Test Accuracy:** 94%

**Configuración de Entrenamiento:**
- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 32
- **Early Stopping:** Patience = 7
- **Data Augmentation:** Flip horizontal, rotación, normalización ImageNet

---

## 📖 Referencias

- [SqueezeNet Paper](https://arxiv.org/abs/1602.07360)
- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [SqueezeNet Documentation](https://pytorch.org/vision/stable/models/squeezenet.html)

---

**Curso INFO1185 - 2025**
