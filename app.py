"""
Interfaz visual para el Clasificador de Familias de Proteínas.
Incluye información biológica detallada de aminoácidos y familias Pfam.
Ejecutar con: streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import os
import sys
import torch
import torch.nn as nn
import pickle
from datos_biologicos import AMINOACIDOS_INFO, PFAM_DESCRIPCIONES, get_pfam_info

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

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

TIPO_EMOJI = {
    'No polar': '🔴',
    'Polar': '🔵',
    'Cargado positivamente': '⚡',
    'Cargado negativamente': '🔻',
    'Aromático': '🟣',
    'Especial': '🟡',
}


def extraer_caracteristicas_manuales(secuencia):
    if not secuencia or len(secuencia) == 0:
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


def get_aa_tipo(code):
    info = AMINOACIDOS_INFO.get(code, {})
    tipo = info.get('tipo', '')
    if 'cargado negativamente' in tipo.lower() or 'ácido' in tipo.lower():
        return 'Cargado negativamente'
    elif 'cargado positivamente' in tipo.lower() or 'básico' in tipo.lower():
        return 'Cargado positivamente'
    elif 'aromático' in tipo.lower():
        return 'Aromático'
    elif 'especial' in tipo.lower() or 'imino' in tipo.lower():
        return 'Especial'
    elif 'polar' in tipo.lower():
        return 'Polar'
    else:
        return 'No polar'


def format_seq_colored(secuencia, max_len=80):
    lines = []
    for i in range(0, min(len(secuencia), max_len), 10):
        chunk = secuencia[i:i+10]
        labeled = " ".join(f"{aa}" for aa in chunk)
        lines.append(labeled)
    result = "\n".join(lines)
    if len(secuencia) > max_len:
        result += f"\n... ({len(secuencia)} aa total)"
    return result


st.set_page_config(
    page_title="Clasificador de Proteínas v3",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


class ProteinMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


@st.cache_resource
def cargar_modelo():
    model_path = "modelos/pfam_model_5513.pt"
    use_pytorch = os.path.exists(model_path)

    if use_pytorch:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        input_size = checkpoint['input_size']
        num_classes = checkpoint['num_classes']
        modelo = ProteinMLP(input_size, num_classes)
        modelo.load_state_dict(checkpoint['model_state_dict'])
        modelo.eval()

        with open("modelos/pfam_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        with open("modelos/pfam_scaler.pkl", "rb") as f:
            scaler_manual = pickle.load(f)

        info = {
            'num_features': input_size,
            'use_embeddings': False,
            'accuracy': checkpoint.get('accuracy', 0.92),
            'num_clases': num_classes,
            'model_type': 'pytorch_mlp',
        }
        return modelo, encoder, info, scaler_manual, None, None, False
    else:
        modelo = joblib.load("modelos/clasificador_proteinas.joblib")
        encoder = joblib.load("modelos/label_encoder.joblib")

        info = {}
        info_path = "modelos/feature_info.json"
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)

        scaler_manual = None
        if os.path.exists("modelos/scaler_manual.joblib"):
            scaler_manual = joblib.load("modelos/scaler_manual.joblib")

        use_emb = info.get('use_embeddings', False)
        return modelo, encoder, info, scaler_manual, None, None, use_emb


modelo, encoder, model_info, scaler_manual, scaler_emb, embedder, use_embeddings = cargar_modelo()


def predecir_familia(secuencia):
    manual_feats = np.array([extraer_caracteristicas_manuales(secuencia)], dtype=np.float32)

    if scaler_manual is not None:
        manual_feats = scaler_manual.transform(manual_feats)

    features = manual_feats

    if model_info.get('model_type') == 'pytorch_mlp':
        with torch.no_grad():
            input_tensor = torch.tensor(features, dtype=torch.float32)
            logits = modelo(input_tensor)
            probabilidades = torch.softmax(logits, dim=1).numpy()[0]
            prediccion_idx = int(np.argmax(probabilidades))
    else:
        prediccion_idx = modelo.predict(features)[0]
        probabilidades = modelo.predict_proba(features)[0]

    familia = encoder.inverse_transform([prediccion_idx])[0]

    top_indices = np.argsort(probabilidades)[::-1][:5]
    top_predicciones = [
        (encoder.inverse_transform([i])[0], float(probabilidades[i]))
        for i in top_indices
    ]

    return familia, top_predicciones


# ── SIDEBAR ──

with st.sidebar:
    st.header("Panel de Control")

    st.subheader("Modelo Actual")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Familias", f"{len(encoder.classes_):,}")
        st.metric("Features", model_info.get('num_features', 'N/A'))
    with col_m2:
        st.metric("Precision", f"{model_info.get('accuracy', 0.92):.1%}")
        st.metric("Embeddings", "No")

    st.divider()
    st.subheader("Buscar Familia Pfam")
    familias = encoder.classes_
    search_fam = st.text_input("Nombre de familia:", placeholder="PF00001...")
    filtered = [f for f in familias if search_fam.upper() in f.upper()] if search_fam else []
    if search_fam:
        if filtered:
            for fam in filtered[:30]:
                pfam_info = get_pfam_info(fam)
                if pfam_info:
                    st.write(f"**{fam}** - {pfam_info.get('nombre', '')}")
                else:
                    st.write(f"  {fam}")
            if len(filtered) > 30:
                st.write(f"  ... y {len(filtered) - 30} mas")
        else:
            st.warning("No se encontraron familias")

    st.divider()
    st.subheader("Buscar Aminoacido")
    aa_search = st.text_input("Codigo o nombre:", placeholder="A, Ala, Alanina...")
    if aa_search:
        matches = []
        for code, info in AMINOACIDOS_INFO.items():
            if (aa_search.upper() in code.upper()
                or aa_search.lower() in info['nombre'].lower()
                or aa_search.lower() in info['abreviatura'].lower()):
                matches.append((code, info))
        if matches:
            for code, info in matches:
                tipo = get_aa_tipo(code)
                emoji = TIPO_EMOJI.get(tipo, '⬜')
                esencial = "Si" if info.get('esencial') else "No"
                st.write(f"{emoji} **{code}** - {info['nombre']} ({info['abreviatura']})")
                st.caption(f"{info['tipo']} | PM: {info['peso_molecular']} Da | Carga: {info['carga']} | Esencial: {esencial}")
                with st.expander(f"Mas info sobre {info['nombre']}"):
                    st.write(f"**Descripcion:** {info.get('descripcion', '')}")
                    st.write(f"**Donde se encuentra:** {info.get('donde_se_encuentra', '')}")
                    st.write(f"**Funcion biologica:** {info.get('funcion_biologica', '')}")
                    st.write(f"**Propiedades quimicas:** {info.get('propiedades_quimicas', '')}")
                    st.write(f"**Hidrofobicidad:** {info.get('hidrofobicidad', '')}")
                    st.write(f"**Enfermedades relacionadas:** {info.get('enfermedades_relacionadas', '')}")
        else:
            st.warning("No encontrado")
    else:
        st.caption("Leyenda de tipos:")
        for tipo, emoji in TIPO_EMOJI.items():
            st.write(f"{emoji} {tipo}")

# ── MAIN AREA ──

st.title("Clasificador de Familias de Proteinas")
st.caption("IA para clasificacion de proteinas | 5,513 familias Pfam | Precision: 92% | PyTorch MLP")

st.divider()

st.subheader("Ingresa tu secuencia proteica")

secuencia_input = st.text_area(
    "Secuencia de aminoacidos:",
    height=120,
    placeholder="Ej: MKWVTFISLLFLFSSAYSRGVFRDTHKSEIAHRFKDLGEEHFKGLV...",
    help="Ingresa la secuencia usando los 20 aminoacidos estandar (A C D E F G H I K L M N P Q R S T V W Y)"
)

if secuencia_input.strip():
    secuencia_limpia = ''.join(
        c for c in secuencia_input.upper()
        if c in 'ACDEFGHIKLMNPQRSTVWY'
    )

    if secuencia_limpia:
        st.write("**Secuencia analizada:**")
        st.code(format_seq_colored(secuencia_limpia), language=None)

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Longitud", f"{len(secuencia_limpia)} aa")
        with col_s2:
            peso = len(secuencia_limpia) * 110
            st.metric("Peso aprox.", f"{peso:,} Da")
        with col_s3:
            pct_charged = ((secuencia_limpia.count('K') + secuencia_limpia.count('R') + secuencia_limpia.count('H') + secuencia_limpia.count('D') + secuencia_limpia.count('E')) / len(secuencia_limpia)) * 100
            st.metric("Cargados", f"{pct_charged:.1f}%")
        with col_s4:
            pct_hydrophobic = sum(secuencia_limpia.count(aa) for aa in 'AILMFVW') / len(secuencia_limpia) * 100
            st.metric("Hidrofobicos", f"{pct_hydrophobic:.1f}%")

if st.button("Predecir Familia", type="primary", use_container_width=True):
    if secuencia_input.strip():
        secuencia_limpia = ''.join(
            c for c in secuencia_input.upper()
            if c in 'ACDEFGHIKLMNPQRSTVWY'
        )

        if len(secuencia_limpia) < 10:
            st.error("La secuencia es muy corta (minimo 10 aminoacidos)")
        else:
            with st.spinner("Analizando secuencia con IA..."):
                familia, top = predecir_familia(secuencia_limpia)

            st.success(f"Familia Predicha: **{familia}**  |  Confianza: **{top[0][1]:.1%}**")

            st.subheader("Informacion de la familia")
            pfam_info = get_pfam_info(familia)
            if pfam_info:
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**Nombre:** {pfam_info.get('nombre', '')}")
                    st.write(f"**Tipo:** {pfam_info.get('tipo', '')}")
                    st.write(f"**Organismos:** {pfam_info.get('organismos', '')}")
                with col_info2:
                    st.write(f"**Funcion:** {pfam_info.get('funcion', '')}")
                st.info(pfam_info.get('descripcion', ''))
            else:
                st.info(f"Informacion detallada de {familia} no disponible en la base de datos local.")

            st.subheader("Top 5 predicciones")
            df_probs = pd.DataFrame({
                "Familia": [t[0] for t in top],
                "Probabilidad": [t[1] for t in top],
            })

            col_p1, col_p2 = st.columns([1, 1])
            with col_p1:
                st.bar_chart(df_probs.set_index("Familia"), use_container_width=True)

            with col_p2:
                for i, (fam, prob) in enumerate(top):
                    pfam_info_top = get_pfam_info(fam)
                    fam_name = pfam_info_top.get('nombre', '') if pfam_info_top else ''
                    label = f"**{fam}**" + (f" - {fam_name}" if fam_name else "")
                    st.write(label)
                    st.progress(min(prob, 1.0), text=f"{prob:.1%}")

            st.dataframe(
                df_probs.style.format({"Probabilidad": "{:.2%}"}),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("Ingresa una secuencia para predecir")

if secuencia_input.strip():
    secuencia_limpia = ''.join(
        c for c in secuencia_input.upper()
        if c in 'ACDEFGHIKLMNPQRSTVWY'
    )

    if secuencia_limpia:
        st.divider()
        st.subheader("Analisis de Composicion")

        composicion = {
            aa: secuencia_limpia.count(aa) / len(secuencia_limpia) * 100
            for aa in AMINOACIDOS
        }

        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            st.write("**Composicion por aminoacido**")
            df_comp = pd.DataFrame({
                "Aminoacido": [f"{aa} ({AMINOACIDOS_INFO[aa]['abreviatura']})" for aa in AMINOACIDOS],
                "Composicion (%)": [composicion[aa] for aa in AMINOACIDOS]
            })
            st.bar_chart(df_comp.set_index("Aminoacido"), use_container_width=True)

        with col_c2:
            st.write("**Distribucion por tipo**")
            tipos_count = {}
            for aa in AMINOACIDOS:
                tipo = get_aa_tipo(aa)
                tipos_count[tipo] = tipos_count.get(tipo, 0) + secuencia_limpia.count(aa)

            tipos_pct = {k: v / len(secuencia_limpia) * 100 for k, v in tipos_count.items()}
            df_tipos = pd.DataFrame({
                "Tipo": [f"{TIPO_EMOJI.get(t, '')} {t}" for t in tipos_pct.keys()],
                "Porcentaje (%)": list(tipos_pct.values())
            })
            st.bar_chart(df_tipos.set_index("Tipo"), use_container_width=True)

        st.subheader("Top 5 aminoacidos presentes")
        top_aa = sorted(composicion.items(), key=lambda x: x[1], reverse=True)
        significant_aa = [(aa, pct) for aa, pct in top_aa if pct > 0]

        aa_cols = st.columns(min(len(significant_aa), 5))
        for idx, (aa, pct) in enumerate(significant_aa[:5]):
            with aa_cols[idx]:
                info = AMINOACIDOS_INFO.get(aa, {})
                tipo = get_aa_tipo(aa)
                emoji = TIPO_EMOJI.get(tipo, '')
                esencial = "Esencial" if info.get('esencial') else "No esencial"
                st.metric(
                    label=f"{emoji} {aa} - {info.get('nombre', aa)}",
                    value=f"{pct:.1f}%",
                    delta=f"{tipo} | {esencial}"
                )

        with st.expander("Ver detalles completos de todos los aminoacidos presentes"):
            for aa, pct in significant_aa:
                info = AMINOACIDOS_INFO.get(aa, {})
                if not info:
                    continue
                tipo = get_aa_tipo(aa)
                emoji = TIPO_EMOJI.get(tipo, '')
                esencial = "Si" if info.get('esencial') else "No"

                st.write(f"### {emoji} {aa} - {info['nombre']} ({info['abreviatura']})  —  **{pct:.1f}%**")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write(f"- **Tipo:** {info['tipo']}")
                    st.write(f"- **Peso molecular:** {info['peso_molecular']} Da")
                    st.write(f"- **Carga:** {info['carga']}")
                    st.write(f"- **Hidrofobicidad:** {info['hidrofobicidad']}")
                    st.write(f"- **Esencial:** {esencial}")
                with col_d2:
                    st.write(f"- **Descripcion:** {info.get('descripcion', '')}")
                    st.write(f"- **Donde se encuentra:** {info.get('donde_se_encuentra', '')}")
                    st.write(f"- **Funcion biologica:** {info.get('funcion_biologica', '')}")
                    st.write(f"- **Propiedades quimicas:** {info.get('propiedades_quimicas', '')}")
                    st.write(f"- **Enfermedades:** {info.get('enfermedades_relacionadas', '')}")
                st.divider()

st.divider()
st.caption("Clasificador de Proteinas v3.0 | PyTorch MLP | 5,513 familias | 92% precision | Dataset: Pfam | Hugging Face Spaces")
