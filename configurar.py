"""
Script interactivo para seleccionar familias y configurar el entrenamiento.
"""

import pandas as pd
import glob
import os

RUTA_DATASET = r"C:\Users\Fabián\Downloads\archive\random_split\random_split"

print("=== Cargador de Familias de Proteínas ===\n")

# Cargar datos
print("Cargando datos del dataset...")
archivos_train = glob.glob(os.path.join(RUTA_DATASET, "train", "data-*"))

if not archivos_train:
    print("ERROR: No se encontraron archivos en", RUTA_DATASET)
    exit(1)

print(f"Encontrados {len(archivos_train)} archivos. Leyendo muestra...")

# Leer primeros 3 archivos para tener una muestra representativa
dfs = []
for archivo in archivos_train[:3]:
    df_temp = pd.read_csv(archivo)
    dfs.append(df_temp)

df_muestra = pd.concat(dfs, ignore_index=True)
print(f"Muestra cargada: {len(df_muestra)} secuencias\n")

# Contar familias
conteo = df_muestra['family_accession'].value_counts()

print("=" * 60)
print(f"TOP 50 FAMILIAS MÁS COMUNES (de {len(conteo)} totales)")
print("=" * 60)

for i, (familia, count) in enumerate(conteo.head(50).items()):
    print(f"  {i+1:2d}. {familia:15s} -> {count:4d} ejemplos")

print("\n" + "=" * 60)
print("SELECCIÓN DE FAMILIAS")
print("=" * 60)

print("\nOpciones:")
print("  1. Usar las TOP N familias más comunes")
print("  2. Seleccionar familias específicas por nombre")
print("  3. Usar todas las familias con al menos X ejemplos")

opcion = input("\nSelecciona una opción (1-3): ").strip()

familias_seleccionadas = []

if opcion == "1":
    n = int(input("¿Cuántas familias top quieres usar? (ej: 5, 10, 20): ").strip())
    familias_seleccionadas = conteo.head(n).index.tolist()
    print(f"\nSeleccionadas: {len(familias_seleccionadas)} familias")

elif opcion == "2":
    print("\nIngresa los nombres de familia (uno por línea, escribe 'fin' para terminar):")
    print("Ejemplo: PF00001.20")
    while True:
        familia = input("> ").strip()
        if familia.lower() == 'fin':
            break
        if familia in conteo.index:
            familias_seleccionadas.append(familia)
            print(f"  Agregada: {familia} ({conteo[familia]} ejemplos)")
        else:
            print(f"  Familia no encontrada. Verifica el nombre.")

elif opcion == "3":
    min_ejemplos = int(input("¿Mínimo ejemplos por familia? (ej: 50, 100): ").strip())
    familias_seleccionadas = conteo[conteo >= min_ejemplos].index.tolist()
    print(f"\nSeleccionadas: {len(familias_seleccionadas)} familias")

else:
    print("Opción no válida. Usando top 10 familias por defecto.")
    familias_seleccionadas = conteo.head(10).index.tolist()

# Guardar configuración
print("\n" + "=" * 60)
print("GUARDANDO CONFIGURACIÓN")
print("=" * 60)

config = {
    'familias': familias_seleccionadas,
    'num_familias': len(familias_seleccionadas),
}

# Crear archivo de configuración
with open("config_familias.txt", "w") as f:
    for familia in familias_seleccionadas:
        f.write(f"{familia}\n")

print(f"\nConfiguración guardada en 'config_familias.txt'")
print(f"Familias seleccionadas: {len(familias_seleccionadas)}")
print("\nFamilias:")
for fam in familias_seleccionadas:
    print(f"  - {fam}")

# Mostrar estimación de precisión
print("\n" + "=" * 60)
print("ESTIMACIÓN DE PRECISIÓN")
print("=" * 60)

# Estimación muy aproximada basada en complejidad
num_clases = len(familias_seleccionadas)
if num_clases <= 5:
    estimacion = "70-90% (problema fácil)"
elif num_clases <= 10:
    estimacion = "50-75% (problema moderado)"
elif num_clases <= 20:
    estimacion = "30-55% (problema difícil)"
else:
    estimacion = "10-35% (problema muy difícil)"

print(f"\nCon {num_clases} familias, precisión estimada: {estimacion}")
print("\nPara mejorar precisión: usa menos familias o más datos de entrenamiento")

print("\n" + "=" * 60)
print("SIGUIENTE PASO: Ejecutar train.py")
print("=" * 60)
print("\nEl script train.py usará automáticamente las familias seleccionadas.")
print("Ejecuta: py train.py")
