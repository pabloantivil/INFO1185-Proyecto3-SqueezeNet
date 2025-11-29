# INFO1185-Proyecto3-ShuffleNet

Transfer Learning para Clasificación de Frutas y Verduras utilizando ShuffleNet V2 preentrenado en ImageNet.

**Autor:** Benja y Pablo 
**Curso:** INFO1185  
**Año:** 2025

---

## 🎯 Objetivo

Implementar y comparar dos clasificadores usando transfer learning con ShuffleNet:
- **Versión 1 (Simple)**: Una capa Fully Connected
- **Versión 2 (Complejo)**: Múltiples capas con BatchNorm y Dropout

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
INFO1185-Proyecto3-ShuffleNet/
├── src/
│   ├── __init__.py              # Inicialización del paquete
│   ├── data_preparation.py      # Preparación y carga de datos
│   └── model.py                 # Modelo ShuffleNet
├── archive/                     # Dataset (no versionado)
├── main.py                      # Script principal de ejecución
├── requirements.txt             # Dependencias de Python
├── .gitignore                   # Archivos ignorados por Git
└── README.md                    # Documentación
```

---

## 🚀 Instalación

```powershell
# 1. Clonar repositorio
git clone https://github.com/pabloantivil/INFO1185-Proyecto3-ShuffleNet.git
cd INFO1185-Proyecto3-ShuffleNet

# 2. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Instalar PyTorch (CPU)
pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Instalar otras dependencias
pip install -r requirements.txt
```

---

## 🎮 Uso

```powershell
# Ejecutar proyecto completo
python main.py

# Probar módulos individuales
python -m src.data_preparation
python -m src.model
```

---

## ✨ Parte 1 - Implementado

### Preparación de Datos
- ✅ Dataset pre-dividido (train/val/test)
- ✅ Transformaciones con data augmentation
- ✅ Normalización ImageNet
- ✅ DataLoaders optimizados

### Modelo ShuffleNet
- ✅ ShuffleNet V2 x1.0 preentrenado
- ✅ Feature extractor congelado
- ✅ Clasificador simple (1 capa Linear)
- ✅ 5,125 parámetros entrenables (0.41%)

---

## � Información del Modelo

```
Arquitectura:     ShuffleNet V2 x1.0
Parámetros:       1,258,729 total
Entrenables:      5,125 (solo clasificador)
Congelados:       1,253,604

Clasificador:     Linear(1024 → 5)
BatchNorm:        NO
Dropout:          NO
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

- [ShuffleNet V2 Paper](https://arxiv.org/abs/1807.11164)
- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

**Curso INFO1185 - 2025**
