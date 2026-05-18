"""
Interfaz visual para el Clasificador de Familias de Proteínas.
Soporta embeddings ESM-2 + features manuales.
Incluye información biológica detallada de aminoácidos y familias Pfam.
Ejecutar con: streamlit run streamlit_app.py
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

GRUPOS_COLORES = {
    'nonpolar': '#FF6B6B',
    'polar_uncharged': '#4ECDC4',
    'positively_charged': '#45B7D1',
    'negatively_charged': '#F7DC6F',
    'aromatic': '#BB8FCE',
    'aliphatic': '#E59866',
    'tiny': '#82E0AA',
    'special': '#F1948A',
}

TIPO_COLORES = {
    'No polar': '#FF6B6B',
    'Polar': '#4ECDC4',
    'Cargado positivamente': '#45B7D1',
    'Cargado negativamente': '#F7DC6F',
    'Aromático': '#BB8FCE',
    'Especial': '#F1948A',
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


def color_seq_bar(secuencia):
    colored = []
    for aa in secuencia:
        tipo = get_aa_tipo(aa)
        color = TIPO_COLORES.get(tipo, '#CCCCCC')
        colored.append(
            f'<span style="background-color:{color};color:#1a1a2e;padding:2px 4px;'
            f'border-radius:3px;font-family:monospace;font-size:13px;font-weight:bold;'
            f'margin:1px;display:inline-block;" title="{AMINOACIDOS_INFO.get(aa, {}).get("nombre", aa)} ({tipo})">{aa}</span>'
        )
    return ''.join(colored)


def show_pfam_info(familia_id):
    info = get_pfam_info(familia_id)
    if info:
        st.markdown(f"#### {info.get('nombre', familia_id)}")
        st.markdown(f"**{info.get('descripcion', '')}**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"🏷️ **Tipo:** {info.get('tipo', 'N/A')}")
            st.markdown(f"🌍 **Organismos:** {info.get('organismos', 'N/A')}")
        with col_b:
            st.markdown(f"⚡ **Función:** {info.get('funcion', 'N/A')}")
    else:
        st.info(f"Información detallada de {familia_id} no disponible en la base de datos local.")


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


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: #e94560;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #a8d8ea;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .result-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .aa-card {
        background: #f8f9fa;
        border-left: 4px solid #4ECDC4;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
    .pfam-card {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .seq-display {
        background: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        overflow-x: auto;
        white-space: normal;
        word-wrap: break-word;
        line-height: 2;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    .legend-item {
        display: inline-block;
        margin: 0.2rem 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧬 Clasificador de Familias de Proteínas</h1>
    <p>IA para clasificación de proteínas basada en 5,513 familias Pfam | Precisión: 92%</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Panel de Control")

    st.markdown("### 🧪 Modelo Actual")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Familias", f"{len(encoder.classes_):,}")
        st.metric("Features", model_info.get('num_features', 'N/A'))
    with col_m2:
        st.metric("Precisión", f"{model_info.get('accuracy', 0.92):.1%}")
        st.metric("Embeddings", "No" if not use_embeddings else "Sí")

    st.markdown("---")
    st.markdown("### 🔍 Buscar Familia")
    familias = encoder.classes_
    search = st.text_input("Nombre de familia:", placeholder="PF00001...")
    filtered = [f for f in familias if search.upper() in f.upper()] if search else familias
    if search and filtered:
        for fam in filtered[:30]:
            pfam_info = get_pfam_info(fam)
            if pfam_info:
                st.markdown(f"**{fam}** — {pfam_info.get('nombre', '')}")
            else:
                st.text(f"  {fam}")
        if len(filtered) > 30:
            st.text(f"  ... y {len(filtered) - 30} más")
    elif search:
        st.warning("No se encontraron familias")
    else:
        st.info("Escribe para buscar entre 5,513 familias")

    st.markdown("---")
    st.markdown("### 🧪 Aminoácidos")
    aa_search = st.text_input("Buscar aminoácido:", placeholder="A, Ala, Alanina...")
    if aa_search:
        matches = []
        for code, info in AMINOACIDOS_INFO.items():
            if aa_search.upper() in code.upper() or aa_search.lower() in info['nombre'].lower() or aa_search.lower() in info['abreviatura'].lower():
                matches.append((code, info))
        if matches:
            for code, info in matches:
                tipo = get_aa_tipo(code)
                color = TIPO_COLORES.get(tipo, '#CCC')
                st.markdown(f"""
                <div class="aa-card" style="border-left-color:{color};">
                    <strong style="font-size:1.2em;color:{color};">{code}</strong> — {info['nombre']} ({info['abreviatura']})<br>
                    <small>{info['tipo']} | PM: {info['peso_molecular']} Da | {info['carga']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No encontrado")
    else:
        st.markdown("**Leyenda de colores:**")
        for tipo, color in TIPO_COLORES.items():
            st.markdown(f'<span style="background:{color};color:#1a1a2e;padding:2px 8px;border-radius:4px;font-weight:bold;font-size:0.85em;">{tipo}</span>', unsafe_allow_html=True)

st.markdown("### 📝 Ingresa tu secuencia proteica")

secuencia_input = st.text_area(
    "",
    height=120,
    placeholder="Ej: MKWVTFISLLFLFSSAYSRGVFRDTHKSEIAHRFKDLGEEHFKGLV...",
    help="Ingresa la secuencia completa de la proteína usando los 20 aminoácidos estándar (A C D E F G H I K L M N P Q R S T V W Y)",
    label_visibility="collapsed"
)

if secuencia_input.strip():
    secuencia_limpia = ''.join(
        c for c in secuencia_input.upper()
        if c in 'ACDEFGHIKLMNPQRSTVWY'
    )

    if secuencia_limpia:
        st.markdown("#### 🔬 Secuencia analizada")
        display_seq = secuencia_limpia if len(secuencia_limpia) <= 200 else secuencia_limpia[:200] + f"... ({len(secuencia_limpia)} aa total)"
        st.markdown(
            f'<div class="seq-display">{color_seq_bar(display_seq[:200])}</div>',
            unsafe_allow_html=True
        )

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
            st.metric("Hidrofóbicos", f"{pct_hydrophobic:.1f}%")

if st.button("🧬 Predecir Familia", type="primary", use_container_width=True):
    if secuencia_input.strip():
        secuencia_limpia = ''.join(
            c for c in secuencia_input.upper()
            if c in 'ACDEFGHIKLMNPQRSTVWY'
        )

        if len(secuencia_limpia) < 10:
            st.error("La secuencia es muy corta (mínimo 10 aminoácidos)")
        else:
            with st.spinner("Analizando secuencia con IA..."):
                familia, top = predecir_familia(secuencia_limpia)

            st.markdown(f"""
            <div class="result-card">
                <h2 style="margin:0;color:white;">🎯 Familia Predicha</h2>
                <h1 style="margin:0.5rem 0;color:white;">{familia}</h1>
                <p style="margin:0;color:#e8f5e9;">Confianza: {top[0][1]:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📋 Información de la familia")
            show_pfam_info(familia)

            st.markdown("#### 📊 Top 5 predicciones")
            df_probs = pd.DataFrame({
                "Familia": [t[0] for t in top],
                "Probabilidad": [t[1] for t in top],
            })
            df_probs["Prob. (%)"] = [f"{t[1]:.2%}" for t in top]

            col_p1, col_p2 = st.columns([1, 1])
            with col_p1:
                st.bar_chart(df_probs[["Familia", "Probabilidad"]].set_index("Familia"), use_container_width=True)
            with col_p2:
                for i, (fam, prob) in enumerate(top):
                    pfam_info = get_pfam_info(fam)
                    fam_name = pfam_info.get('nombre', '') if pfam_info else ''
                    bar_color = "#11998e" if i == 0 else "#a8d8ea"
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;margin-bottom:0.5rem;">
                        <div style="flex:1;">
                            <strong>{fam}</strong>
                            {f'<br><small>{fam_name}</small>' if fam_name else ''}
                        </div>
                        <div style="width:120px;background:#e0e0e0;border-radius:8px;overflow:hidden;height:24px;">
                            <div style="width:{prob*100:.0f}%;background:{bar_color};height:100%;border-radius:8px;"></div>
                        </div>
                        <div style="width:60px;text-align:right;font-weight:bold;">{prob:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("Ingresa una secuencia para predecir")

if secuencia_input.strip():
    secuencia_limpia = ''.join(
        c for c in secuencia_input.upper()
        if c in 'ACDEFGHIKLMNPQRSTVWY'
    )

    if secuencia_limpia:
        st.markdown("---")
        st.markdown("### 🧪 Análisis de Composición")

        composicion = {
            aa: secuencia_limpia.count(aa) / len(secuencia_limpia) * 100
            for aa in AMINOACIDOS
        }

        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            st.markdown("#### Composición por aminoácido")
            df_comp = pd.DataFrame({
                "Aminoácido": [f"{aa} ({AMINOACIDOS_INFO[aa]['abreviatura']})" for aa in AMINOACIDOS],
                "Composición (%)": [composicion[aa] for aa in AMINOACIDOS]
            })
            st.bar_chart(df_comp.set_index("Aminoácido"), use_container_width=True)

        with col_c2:
            st.markdown("#### Distribución por tipo")
            tipos_count = {}
            for aa in AMINOACIDOS:
                tipo = get_aa_tipo(aa)
                tipos_count[tipo] = tipos_count.get(tipo, 0) + secuencia_limpia.count(aa)

            tipos_pct = {k: v / len(secuencia_limpia) * 100 for k, v in tipos_count.items()}
            df_tipos = pd.DataFrame({
                "Tipo": list(tipos_pct.keys()),
                "Porcentaje (%)": list(tipos_pct.values())
            })
            st.bar_chart(df_tipos.set_index("Tipo"), use_container_width=True)

        st.markdown("### 📖 Detalle de aminoácidos presentes")
        top_aa = sorted(composicion.items(), key=lambda x: x[1], reverse=True)
        significant_aa = [(aa, pct) for aa, pct in top_aa if pct > 0]

        aa_cols = st.columns(min(len(significant_aa), 5))
        for idx, (aa, pct) in enumerate(significant_aa[:5]):
            with aa_cols[idx]:
                info = AMINOACIDOS_INFO.get(aa, {})
                tipo = get_aa_tipo(aa)
                color = TIPO_COLORES.get(tipo, '#CCC')
                st.markdown(f"""
                <div style="background:{color}15;border:2px solid {color};border-radius:10px;padding:0.8rem;text-align:center;">
                    <div style="font-size:2rem;font-weight:bold;color:{color};">{aa}</div>
                    <div style="font-size:0.85rem;font-weight:bold;">{info.get('nombre', aa)}</div>
                    <div style="font-size:1.2rem;font-weight:bold;color:{color};">{pct:.1f}%</div>
                    <div style="font-size:0.7rem;color:#666;">{info.get('tipo', '')}</div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("Ver detalles completos de todos los aminoácidos presentes"):
            for aa, pct in significant_aa:
                info = AMINOACIDOS_INFO.get(aa, {})
                if not info:
                    continue
                tipo = get_aa_tipo(aa)
                color = TIPO_COLORES.get(tipo, '#CCC')
                esencial = "Sí" if info.get('esencial') else "No"
                st.markdown(f"""
                <div class="aa-card" style="border-left-color:{color};">
                    <strong style="font-size:1.3em;color:{color};">{aa}</strong> — {info['nombre']} ({info['abreviatura']})
                    <span style="float:right;font-weight:bold;color:{color};">{pct:.1f}%</span><br>
                    <small>
                    📦 Tipo: {info['tipo']} | 
                    ⚖️ PM: {info['peso_molecular']} Da | 
                    ⚡ Carga: {info['carga']} | 
                    💧 Hidrofobicidad: {info['hidrofobicidad']} | 
                    ✅ Esencial: {esencial}
                    </small><br>
                    <small>🔬 {info.get('descripcion', '')}</small><br>
                    <small>📍 {info.get('donde_se_encuentra', '')}</small><br>
                    <small>🧪 {info.get('propiedades_quimicas', '')}</small>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <div style='display: inline-flex; gap: 2rem; color: #888; font-size: 0.85rem;'>
        <span>🧬 Clasificador de Proteínas v3.0</span>
        <span>🤖 PyTorch MLP | 5,513 familias | 92% precisión</span>
        <span>📊 Dataset: Pfam</span>
        <span>☁️ Hugging Face Spaces</span>
    </div>
</div>
""", unsafe_allow_html=True)
