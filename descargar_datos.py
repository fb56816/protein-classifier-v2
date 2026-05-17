"""
Descarga el dataset de Pfam desde Kaggle.
Requiere tener la API token de Kaggle configurado.
"""

import os
import sys

def descargar():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Instalando kaggle...")
        os.system("pip install kaggle")
        from kaggle.api.kaggle_api_extended import KaggleApi

    # Crear directorio data
    os.makedirs("data", exist_ok=True)

    # Inicializar API
    api = KaggleApi()
    api.authenticate()

    # Descargar dataset
    print("Descargando dataset Pfam Seed Random Split...")
    print("Esto puede tardar unos minutos dependiendo de tu conexión.")

    api.dataset_download_files(
        dataset="googleai/pfam-seed-random-split",
        path="data",
        unzip=True
    )

    print("\n¡Descarga completada!")
    print("Archivos en: data/")

    # Listar archivos descargados
    archivos = os.listdir("data")
    print(f"Archivos descargados: {archivos}")

if __name__ == "__main__":
    print("=== Descargador de Dataset Pfam ===\n")

    # Verificar si tiene Kaggle configurado
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_json):
        print("Primero necesitas configurar tu API token de Kaggle:")
        print()
        print("1. Ve a https://www.kaggle.com/account")
        print("2. Haz clic en 'Create New API Token'")
        print("3. Descarga el archivo kaggle.json")
        print("4. Colócalo en:")
        print(f"   {kaggle_json}")
        print()
        print("O ejecuta:")
        print("   kaggle config set -n key -v <TU_KEY>")
        print("   kaggle config set -n username -v <TU_USUARIO>")
        print()
        sys.exit(1)

    descargar()
