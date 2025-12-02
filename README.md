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

| Modelo | Test Accuracy | Val Accuracy | Test Loss | Épocas | Parámetros Entrenables |
|--------|--------------|--------------|-----------|--------|----------------------|
| **V2B (Regularizado)** | **98.00%** 🏆 | 97.87% | 0.0449 | 19 | 428,293 |
| **V2A (Sin Reg.)** | 94.00% | 95.74% | 0.1831 | 12 | 427,525 |
| **V1 (Simple)** | 92.00% | 97.87% | 0.1819 | 14 | 265,221 |

**Hallazgo principal:** El modelo con BatchNorm + Dropout (V2B) logró el mejor desempeño, confirmando la teoría de que la regularización mejora la generalización. **Orden de desempeño: V2B > V2A > V1**

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
├── ANALISIS_Y_DISCUSION.md             # Análisis detallado del proyecto
├── requirements.txt                    # Dependencias de Python
├── .gitignore                          # Archivos ignorados por Git
└── README.md                           # Documentación
```

**Nota:** El proyecto fue implementado en un único notebook de Jupyter para facilitar su ejecución en Google Colab.

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

## ✨ Parte 1 - Implementado

### Preparación de Datos
- ✅ Dataset pre-dividido (train/val/test)
- ✅ Transformaciones con data augmentation
- ✅ Normalización ImageNet
- ✅ DataLoaders optimizados

### Modelo SqueezeNet
- ✅ SqueezeNet 1.1 preentrenado
- ✅ Feature extractor congelado (512 features)
- ✅ Tres variantes de clasificadores
- ✅ Transfer Learning efectivo

---

## 🧬 Información del Modelo

```
Arquitectura:     SqueezeNet 1.1
Features:         512 (del feature extractor)
Parámetros V1:    265,221 entrenables
Parámetros V2A:   427,525 entrenables
Parámetros V2B:   428,293 entrenables

Clasificador V1:  Conv2d + Linear (simple)
Clasificador V2:  4 capas FC (512→256→128→5)
BatchNorm:        Solo V2B
Dropout:          Solo V2B (p=0.3)
```

---

## 🔜 Parte 2 - Pendiente

- [ ] Clasificador Versión 2 (complejo)
- [ ] Loop de entrenamiento
- [ ] Validación y métricas
- [ ] Evaluación en test
- [ ] Comparación de versiones

---

## 📖 Referencias

- [SqueezeNet Paper](https://arxiv.org/abs/1602.07360)
- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [SqueezeNet Documentation](https://pytorch.org/vision/stable/models/squeezenet.html)

---

**Curso INFO1185 - Inteligencia Artificial III - 2024**
