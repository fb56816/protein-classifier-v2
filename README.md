---
title: Clasificador de Proteinas Pfam
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
---

# Clasificador de Familias de Proteínas

Modelo de machine learning para clasificar proteínas en familias basándose en su secuencia de aminoácidos.

## Dataset

El dataset usado es [PFam Seed Random Split](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split) de Google AI, que contiene:
- ~1 millón de secuencias de proteínas
- ~18,000 familias de proteínas
- Secuencias en formato de aminoácidos (20 letras estándar)

## Instalación

```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv env
env\Scripts\activate  # Windows
# source env/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

## Descargar datos

### Opción A: Descargar manualmente (recomendado)

1. Ve a https://www.kaggle.com/datasets/googleai/pfam-seed-random-split
2. Haz clic en "Download" (puede requerir crear cuenta gratuita)
3. Extrae los archivos en la carpeta `data/`
4. El script detecta automáticamente archivos `.csv` o `.fasta`

### Opción B: Usar la API de Kaggle

```bash
# Instalar kaggle
pip install kaggle

# Configurar tu API token (descárgalo de https://www.kaggle.com/account)
# Coloca kaggle.json en ~/.kaggle/kaggle.json (Linux/Mac) o %HOMEPATH%\.kaggle\kaggle.json (Windows)

# Ejecutar script de descarga
python descargar_datos.py
```

### Opción C: Usar datos de ejemplo (sin descargar nada)

Si solo quieres probar el código, puedes usar secuencias de ejemplo modificando `train.py` para usar datos simulados.

## Entrenamiento

```bash
python train.py
```

El modelo se guarda en `modelos/clasificador_proteinas.joblib`.

## Predicción

```bash
python predecir.py
```

Puedes ingresar secuencias de proteínas interactivamente o modificar el script para usar tus propias secuencias.

## Cómo funciona

1. **Extracción de características**: Convierte cada secuencia en un vector numérico basado en:
   - Composición de los 20 aminoácidos (frecuencia relativa)
   - Longitud de la secuencia
   - Carga neta (aminoácidos positivos vs negativos)

2. **Modelo**: Random Forest con 100 árboles
   - Funciona bien en CPU
   - Proporciona importancia de características
   - Robusto a overfitting

3. **Salida**: Familia de proteína (accesión PFAM como PF00001)

## Estructura del proyecto

```
protein_classifier/
├── data/                      # Coloca aquí los archivos descargados (.csv o .fasta)
├── modelos/
│   ├── clasificador_proteinas.joblib  # Modelo entrenado
│   └── label_encoder.joblib           # Codificador de familias
├── train.py                   # Entrenamiento (detecta .csv y .fasta automáticamente)
├── predecir.py                # Predicción interactiva
├── descargar_datos.py         # Descarga desde Kaggle (opcional)
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## Ejemplo de secuencia

```
>sp|P12345|PROTEIN_HUMAN Nombre de proteína
MKWVTFISLLFLFSSAYSRG...
```

Las secuencias usan los 20 aminoácidos estándar: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y

## Referencias

- Dataset: https://www.kaggle.com/datasets/googleai/pfam-seed-random-split
- PFam: https://www.ebi.ac.uk/interpro/entry/pfam/
- Documentación scikit-learn: https://scikit-learn.org/
