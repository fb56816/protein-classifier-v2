"""
Entrenamiento mejorado para clasificación de familias de proteínas.
Usa embeddings ESM-2 + features manuales para alcanzar >80% de precisión.
Soporta todas las familias del Pfam con suficiente ejemplos.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import joblib
import os
import glob
import json
import argparse

AMINOACIDOS = 'ACDEFGHIKLMNPQRSTVWY'

PROPIEDADES_AA = {
    'A': {'hydro': 1.8, 'polar': 0, 'charge': 0, 'size': 0, 'flex': 1, 'aromatic': 0},
    'C': {'hydro': 2.5, 'polar': 0, 'charge': 0, 'size': 0, 'flex': 0, 'aromatic': 0},
    'D': {'hydro': -3.5, 'polar': 1, 'charge': -1, 'size': 0, 'flex': 1, 'aromatic': 0},
    'E': {'hydro': -3.5, 'polar': 1, 'charge': -1, 'size': 1, 'flex': 1, 'aromatic': 0},
    'F': {'hydro': 2.8, 'polar': 0, 'charge': 0, 'size': 1, 'flex': 0, 'aromatic': 1},
    'G': {'hydro': -0.4, 'polar': 0, 'charge': 0, 'size': -1, 'flex': 1, 'aromatic': 0},
    'H': {'hydro': -3.2, 'polar': 1, 'charge': 0.5, 'size': 1, 'flex': 0, 'aromatic': 1},
    'I': {'hydro': 4.5, 'polar': 0, 'charge': 0, 'size': 1, 'flex': 0, 'aromatic': 0},
    'K': {'hydro': -3.9, 'polar': 1, 'charge': 1, 'size': 1, 'flex': 1, 'aromatic': 0},
    'L': {'hydro': 3.8, 'polar': 0, 'charge': 0, 'size': 1, 'flex': 0, 'aromatic': 0},
    'M': {'hydro': 1.9, 'polar': 0, 'charge': 0, 'size': 1, 'flex': 0, 'aromatic': 0},
    'N': {'hydro': -3.5, 'polar': 1, 'charge': 0, 'size': 0, 'flex': 1, 'aromatic': 0},
    'P': {'hydro': -1.6, 'polar': 0, 'charge': 0, 'size': 0, 'flex': -1, 'aromatic': 0},
    'Q': {'hydro': -3.5, 'polar': 1, 'charge': 0, 'size': 1, 'flex': 1, 'aromatic': 0},
    'R': {'hydro': -4.5, 'polar': 1, 'charge': 1, 'size': 1, 'flex': 1, 'aromatic': 0},
    'S': {'hydro': -0.8, 'polar': 1, 'charge': 0, 'size': -1, 'flex': 1, 'aromatic': 0},
    'T': {'hydro': -0.7, 'polar': 1, 'charge': 0, 'size': 0, 'flex': 1, 'aromatic': 0},
    'V': {'hydro': 4.2, 'polar': 0, 'charge': 0, 'size': 0, 'flex': 0, 'aromatic': 0},
    'W': {'hydro': -0.9, 'polar': 0, 'charge': 0, 'size': 1, 'flex': 0, 'aromatic': 1},
    'Y': {'hydro': -1.3, 'polar': 1, 'charge': 0, 'size': 1, 'flex': 0, 'aromatic': 1},
}

GRUPOS_AA = {
    'nonpolar': 'AILMFVW',
    'polar_uncharged': 'NQSTY',
    'positively_charged': 'KRH',
    'negatively_charged': 'DE',
    'aromatic': 'FWY',
    'aliphatic': 'ILV',
    'tiny': 'AGS',
    'special': 'CP',
}

RUTA_DATASET = r"C:\Users\Fabián\Downloads\archive\random_split\random_split"
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
RUTA_LOCAL = os.path.join(_SCRIPT_DIR, "data")


def extraer_caracteristicas_manuales(secuencia):
    if not secuencia or pd.isna(secuencia) or len(secuencia) == 0:
        return [0.0] * 530

    seq = secuencia.upper()
    n = len(seq)
    feats = []

    composicion = [seq.count(aa) / n for aa in AMINOACIDOS]
    feats.extend(composicion)

    dipeptidos = []
    for aa1 in AMINOACIDOS:
        for aa2 in AMINOACIDOS:
            dip = aa1 + aa2
            count = seq.count(dip)
            dipeptidos.append(count / max(n - 1, 1))
    feats.extend(dipeptidos)

    for prop_name in ['hydro', 'polar', 'charge', 'size', 'flex', 'aromatic']:
        valores = [PROPIEDADES_AA.get(aa, {}).get(prop_name, 0) for aa in seq]
        if valores:
            feats.append(np.mean(valores))
            feats.append(np.std(valores))
            feats.append(np.min(valores))
            if prop_name == 'hydro':
                ventana = 9
                hydro_mean_vals = []
                for i in range(max(1, n - ventana + 1)):
                    window_vals = valores[i:i + ventana]
                    hydro_mean_vals.append(np.mean(window_vals))
                if hydro_mean_vals:
                    feats.append(np.max(hydro_mean_vals))
                    feats.append(np.min(hydro_mean_vals))
                    feats.append(np.max(hydro_mean_vals) - np.min(hydro_mean_vals))
                else:
                    feats.extend([0, 0, 0])
            else:
                feats.extend([0, 0, 0])
        else:
            feats.extend([0, 0, 0, 0, 0, 0])

    feats.extend([
        n,
        np.log1p(n),
        (seq.count('K') + seq.count('R') + seq.count('H')) / n,
        (seq.count('D') + seq.count('E')) / n,
        seq.count('C') / n,
        seq.count('P') / n,
    ])

    for grupo_nombre, grupo_aas in GRUPOS_AA.items():
        freq = sum(seq.count(aa) for aa in grupo_aas) / n
        feats.append(freq)

    tercio = n // 3
    for i in range(3):
        inicio = i * tercio
        fin = (i + 1) * tercio if i < 2 else n
        subseq = seq[inicio:fin]
        if len(subseq) > 0:
            for aa in AMINOACIDOS:
                feats.append(subseq.count(aa) / len(subseq))
        else:
            feats.extend([0.0] * 20)

    return feats


def cargar_datos():
    print("Cargando datos...")

    rutas_busqueda = [
        os.path.join(RUTA_DATASET, "train"),
        os.path.join(RUTA_LOCAL, "random_split", "random_split", "train"),
        os.path.join(RUTA_LOCAL, "train"),
    ]

    archivos_train = []
    for ruta in rutas_busqueda:
        encontrados = glob.glob(os.path.join(ruta, "data-*"))
        if encontrados:
            archivos_train = encontrados
            print(f"Datos encontrados en: {ruta}")
            break

    if not archivos_train:
        archivos_train = glob.glob(os.path.join(RUTA_LOCAL, "**", "train", "data-*"), recursive=True)

    if not archivos_train:
        print("ERROR: No se encontraron archivos de entrenamiento")
        print(f"Rutas buscadas: {rutas_busqueda}")
        exit(1)

    print(f"Encontrados {len(archivos_train)} archivos")

    dfs = []
    for i, archivo in enumerate(archivos_train):
        try:
            df_temp = pd.read_csv(archivo)
            dfs.append(df_temp)
            if (i + 1) % 20 == 0:
                print(f"  Leídos {i+1}/{len(archivos_train)} archivos ({sum(len(d) for d in dfs):,} filas)")
        except Exception as e:
            print(f"  Error leyendo {archivo}: {e}")

    if not dfs:
        print("ERROR: No se pudieron leer los archivos")
        exit(1)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total datos: {len(df):,} filas, {df['family_accession'].nunique():,} familias")
    return df


def seleccionar_familias(df, min_ejemplos=50, max_familias=1000):
    conteo = df['family_accession'].value_counts()
    familias_validas = conteo[conteo >= min_ejemplos].index[:max_familias].tolist()

    config_path = os.path.join(_SCRIPT_DIR, "config_familias.txt")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            familias_config = [linea.strip() for linea in f if linea.strip()]
        if len(familias_config) > 5:
            familias_config_validas = [f for f in familias_config if f in conteo.index]
            if len(familias_config_validas) > 5:
                familias_validas = familias_config_validas
                print(f"Usando {len(familias_validas)} familias desde config_familias.txt")

    print(f"Familias seleccionadas: {len(familias_validas)}")
    print(f"Total muestras: {sum(conteo[f] for f in familias_validas if f in conteo.index):,}")
    return familias_validas


def main():
    parser = argparse.ArgumentParser(description='Entrenar clasificador de proteínas')
    parser.add_argument('--no-embeddings', action='store_true',
                        help='Usar solo features manuales (sin ESM-2)')
    parser.add_argument('--esm-model', type=str, default='esm2_t6_8M_UR50D',
                        choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D',
                                 'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D'],
                        help='Modelo ESM-2 a usar')
    parser.add_argument('--max-familias', type=int, default=0,
                        help='Max familias (0=todas las disponibles)')
    parser.add_argument('--min-ejemplos', type=int, default=50)
    parser.add_argument('--muestra', type=int, default=0,
                        help='Max muestras total (0=sin limite)')
    parser.add_argument('--max-per-class', type=int, default=200,
                        help='Max ejemplos por familia al submuestrear')
    parser.add_argument('--modelo', type=str, default='auto',
                        choices=['auto', 'xgboost', 'sgd', 'gboost'],
                        help='Modelo: auto|xgboost|sgd|gboost')
    args = parser.parse_args()

    use_embeddings = not args.no_embeddings

    df = cargar_datos()

    if 'sequence' in df.columns:
        df['seq_usable'] = df['sequence']
    else:
        df['seq_usable'] = df['aligned_sequence'].str.replace('.', '', regex=False)

    max_familias = args.max_familias if args.max_familias > 0 else len(df['family_accession'].unique())
    familias = seleccionar_familias(df, min_ejemplos=args.min_ejemplos, max_familias=max_familias)

    df_filtrado = df[df['family_accession'].isin(familias)].copy()
    print(f"\nDataset filtrado: {len(df_filtrado):,} secuencias en {len(familias)} familias")

    max_per_class = args.max_per_class
    needs_sampling = False
    for fam, grp in df_filtrado.groupby('family_accession'):
        if len(grp) > max_per_class:
            needs_sampling = True
            break

    if needs_sampling:
        print(f"Submuestreando (max {max_per_class} por familia)")
        dfs_sampled = []
        for fam, grp in df_filtrado.groupby('family_accession'):
            n_sample = min(len(grp), max_per_class)
            dfs_sampled.append(grp.sample(n=n_sample, random_state=42))
        df_filtrado = pd.concat(dfs_sampled, ignore_index=True)
        print(f"Después de submuestreo: {len(df_filtrado):,} secuencias")

    MUESTRA = args.muestra if args.muestra > 0 else len(df_filtrado)
    if len(df_filtrado) > MUESTRA:
        print(f"Limitando a {MUESTRA:,} muestras totales (estratificado)")
        df_filtrado = df_filtrado.groupby('family_accession').sample(
            n=max(1, MUESTRA // len(familias)), random_state=42, replace=True
        )
        print(f"Después de límite: {len(df_filtrado):,} secuencias")

    df_filtrado = df_filtrado.reset_index(drop=True)

    print("\nTop 15 familias:")
    print(df_filtrado['family_accession'].value_counts().head(15))

    print("\nExtrayendo features manuales...")
    manual_feats = []
    batch_size = 10000
    total = len(df_filtrado)
    for i in range(0, total, batch_size):
        batch = df_filtrado['seq_usable'].iloc[i:i+batch_size]
        batch_feats = batch.apply(extraer_caracteristicas_manuales)
        manual_feats.extend(batch_feats.tolist())
        if (i + batch_size) % 50000 == 0 or i + batch_size >= total:
            print(f"  {min(i+batch_size, total):,}/{total:,} secuencias")

    X_manual = np.array(manual_feats, dtype=np.float32)
    y = df_filtrado['family_accession'].values
    secuencias = df_filtrado['seq_usable'].tolist()

    print(f"Features manuales: {X_manual.shape}")

    if use_embeddings:
        try:
            from embeddings import ProteinEmbedder, has_embeddings_cache, load_embeddings_cache

            cache_dir = "modelos/embeddings_cache"
            if has_embeddings_cache(cache_dir):
                print("\nCargando embeddings desde cache...")
                X_emb, cached_labels, metadata = load_embeddings_cache(cache_dir)
                if len(X_emb) == len(y):
                    print(f"Cache válido: {X_emb.shape}")
                else:
                    print(f"Cache desactualizado ({len(X_emb)} vs {len(y)}). Regenerando...")
                    X_emb = None

            if not has_embeddings_cache(cache_dir) or (use_embeddings and 'X_emb' not in dir()):
                print(f"\nGenerando embeddings ESM-2 ({args.esm_model})...")
                embedder = ProteinEmbedder(model_name=args.esm_model)

                emb_batch_size = 4 if embedder.device.type == 'cuda' else 2
                X_emb = embedder.embed_batch(secuencias, batch_size=emb_batch_size)

                np.save(os.path.join(cache_dir, "embeddings.npy"), X_emb)
                np.save(os.path.join(cache_dir, "labels.npy"), y)

            print(f"Embeddings: {X_emb.shape}")

            print("\nCombinando embeddings + features manuales...")
            scaler_manual = StandardScaler()
            X_manual_scaled = scaler_manual.fit_transform(X_manual)

            scaler_emb = StandardScaler()
            X_emb_scaled = scaler_emb.fit_transform(X_emb)

            X = np.hstack([X_emb_scaled, X_manual_scaled])
            print(f"Features combinados: {X.shape}")

        except Exception as e:
            print(f"\nError con embeddings: {e}")
            print("Usando solo features manuales...")
            use_embeddings = False
            X = X_manual
            scaler_manual = StandardScaler()
            X = scaler_manual.fit_transform(X)
            scaler_emb = None
    else:
        X = X_manual
        scaler_manual = StandardScaler()
        X = scaler_manual.fit_transform(X)
        scaler_emb = None

    print("\nCodificando etiquetas...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_clases = len(label_encoder.classes_)
    print(f"Clases: {num_clases}")

    print("\nDividiendo datos (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Entrenamiento: {len(X_train):,} | Test: {len(X_test):,}")

    print("\nEntrenando modelo...")
    model_type = args.modelo
    if model_type == 'auto':
        model_type = 'sgd' if num_clases > 500 else 'gboost' if num_clases <= 100 else 'xgboost'

    if model_type == 'gboost' and num_clases <= 100:
        print("Usando GradientBoostingClassifier")
        modelo = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, min_samples_split=5, min_samples_leaf=3,
            max_features='sqrt', random_state=42, verbose=1,
        )
        modelo.fit(X_train, y_train)
    elif model_type == 'sgd':
        print(f"Usando SGDClassifier ({num_clases} clases, ultra eficiente)")
        from sklearn.linear_model import SGDClassifier
        modelo = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=1e-4,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        modelo.fit(X_train, y_train)
    else:
        print(f"Usando XGBoost ({num_clases} clases)")
        from xgboost import XGBClassifier
        modelo = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.3,
            subsample=0.7, colsample_bytree=0.3, min_child_weight=20,
            num_class=num_clases, objective='multi:softprob',
            eval_metric='mlogloss', tree_method='hist',
            n_jobs=-1, random_state=42, verbosity=1,
        )
        modelo.fit(X_train, y_train)

    print("\n=== EVALUACIÓN ===")
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nPrecisión en test: {acc:.4f} ({acc*100:.1f}%)")

    print("\nReporte por familia (resumido):")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs("modelos", exist_ok=True)
    joblib.dump(modelo, "modelos/clasificador_proteinas.joblib", compress=3)
    joblib.dump(label_encoder, "modelos/label_encoder.joblib", compress=3)
    if scaler_manual is not None:
        joblib.dump(scaler_manual, "modelos/scaler_manual.joblib", compress=3)
    if scaler_emb is not None:
        joblib.dump(scaler_emb, "modelos/scaler_emb.joblib", compress=3)

    feature_info = {
        'num_features': X.shape[1],
        'use_embeddings': use_embeddings,
        'esm_model': args.esm_model if use_embeddings else None,
        'manual_features': X_manual.shape[1],
        'embedding_dim': X_emb.shape[1] if use_embeddings and 'X_emb' in dir() else 0,
        'familias': list(label_encoder.classes_),
        'accuracy': float(acc),
        'num_clases': num_clases,
            'model_type': model_type,
    }
    with open("modelos/feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    print(f"\n[OK] Modelo guardado en 'modelos/clasificador_proteinas.joblib'")
    print(f"[OK] LabelEncoder en 'modelos/label_encoder.joblib'")

    if hasattr(modelo, 'feature_importances_'):
        print("\n=== Importancia de características (top 15) ===")
        importancias = modelo.feature_importances_
        top_idx = np.argsort(importancias)[::-1][:15]
        if use_embeddings and 'X_emb' in dir():
            emb_dim = X_emb.shape[1]
            for idx in top_idx:
                if idx < emb_dim:
                    print(f"  ESM_dim_{idx}: {importancias[idx]:.4f}")
                else:
                    manual_idx = idx - emb_dim
                    print(f"  Manual_feat_{manual_idx}: {importancias[idx]:.4f}")
        else:
            nombres = (
                [f'comp_{aa}' for aa in AMINOACIDOS] +
                [f'dip_{aa1}{aa2}' for aa1 in AMINOACIDOS for aa2 in AMINOACIDOS] +
                ['hydro_mean', 'hydro_std', 'hydro_min', 'hydro_max_win', 'hydro_min_win', 'hydro_range_win'] +
                ['polar_mean', 'polar_std', 'polar_min',
                 'flex_mean', 'flex_std', 'flex_min',
                 'aromatic_mean', 'aromatic_std', 'aromatic_min',
                 'charge_mean', 'charge_std', 'charge_min',
                 'size_mean', 'size_std', 'size_min'] +
                ['longitud', 'log_longitud', 'carga_pos', 'carga_neg', 'freq_C', 'freq_P'] +
                [f'grupo_{g}' for g in GRUPOS_AA.keys()] +
                [f'tercio{i}_{aa}' for i in range(3) for aa in AMINOACIDOS]
            )
            for idx in top_idx:
                nombre = nombres[idx] if idx < len(nombres) else f'feat_{idx}'
                print(f"  {nombre}: {importancias[idx]:.4f}")

    print(f"\n{'='*50}")
    print(f"RESUMEN FINAL")
    print(f"{'='*50}")
    print(f"Familias: {num_clases}")
    print(f"Features: {X.shape[1]} {'(embeddings + manuales)' if use_embeddings else '(solo manuales)'}")
    print(f"Precisión: {acc*100:.1f}%")
    if acc >= 0.80:
        print("OBJETIVO ALCANZADO: >= 80% de precisión")
    else:
        print(f"Faltan {(0.80 - acc)*100:.1f}% para el objetivo de 80%")
        if not use_embeddings:
            print("Prueba con embeddings: python train.py (sin --no-embeddings)")
        if args.esm_model == 'esm2_t6_8M_UR50D':
            print("Prueba modelo ESM más grande: python train.py --esm-model esm2_t12_35M_UR50D")


if __name__ == "__main__":
    main()
