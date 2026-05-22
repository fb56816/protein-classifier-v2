"""
Interfaz visual para el Clasificador de Familias de Proteinas.
Incluye informacion biologica detallada de aminoacidos y familias Pfam.
Soporte multilingual (es/en/pt/fr/de).
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
from traducciones import t, IDIOMAS_OPCIONES
from huggingface_hub import hf_hub_download

HF_MODEL_REPO = "Fb56816/protein-classifier-model"
MODEL_FILES = ["pfam_model_5513.pt", "pfam_scaler.pkl", "pfam_encoder.pkl"]


def ensure_model_files():
    modelos_dir = os.path.join(SCRIPT_DIR, "modelos")
    os.makedirs(modelos_dir, exist_ok=True)
    for fname in MODEL_FILES:
        local_path = os.path.join(modelos_dir, fname)
        if not os.path.exists(local_path):
            st.info(f"Descargando {fname} desde Hugging Face...")
            downloaded = hf_hub_download(repo_id=HF_MODEL_REPO, filename=fname, local_dir=modelos_dir)
            if downloaded != local_path:
                import shutil
                shutil.move(downloaded, local_path)
            st.success(f"{fname} descargado correctamente")
    return modelos_dir

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


TIPO_KEYS = {
    'No polar': 'nonpolar',
    'Polar': 'polar',
    'Cargado positivamente': 'positively_charged',
    'Cargado negativamente': 'negatively_charged',
    'Aromatico': 'aromatic',
    'Arom\u00e1tico': 'aromatic',
    'Especial': 'special',
}

TIPO_EMOJI = {
    'nonpolar': '\U0001f534',
    'polar': '\U0001f535',
    'positively_charged': '\u26a1',
    'negatively_charged': '\U0001f53b',
    'aromatic': '\U0001f7e3',
    'special': '\U0001f7e1',
}


def get_aa_tipo_key(code):
    info = AMINOACIDOS_INFO.get(code, {})
    tipo = info.get('tipo', '')
    tipo_lower = tipo.lower()
    if 'cargado negativamente' in tipo_lower or '\u00e1cido' in tipo_lower:
        return 'negatively_charged'
    elif 'cargado positivamente' in tipo_lower or 'b\u00e1sico' in tipo_lower:
        return 'positively_charged'
    elif 'arom\u00e1tico' in tipo_lower:
        return 'aromatic'
    elif 'especial' in tipo_lower or 'imino' in tipo_lower:
        return 'special'
    elif 'polar' in tipo_lower:
        return 'polar'
    else:
        return 'nonpolar'


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
    modelos_dir = ensure_model_files()
    model_path = os.path.join(modelos_dir, "pfam_model_5513.pt")
    use_pytorch = os.path.exists(model_path)

    if use_pytorch:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        input_size = checkpoint['input_size']
        num_classes = checkpoint['num_classes']
        modelo = ProteinMLP(input_size, num_classes)
        modelo.load_state_dict(checkpoint['model_state_dict'])
        modelo.eval()

        with open(os.path.join(modelos_dir, "pfam_encoder.pkl"), "rb") as f:
            encoder = pickle.load(f)
        with open(os.path.join(modelos_dir, "pfam_scaler.pkl"), "rb") as f:
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


# ── LANGUAGE SETUP ──

st.set_page_config(
    page_title="Protein Classifier v3",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="expanded",
)

lang_display = st.sidebar.selectbox(
    "\U0001f310",
    options=list(IDIOMAS_OPCIONES.keys()),
    index=0,
    key="lang_selector_global",
)
lang = IDIOMAS_OPCIONES[lang_display]

# ── SIDEBAR ──

with st.sidebar:
    st.header(t('sidebar_header', lang))

    st.subheader(t('model_subtitle', lang))
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(t('families', lang), f"{len(encoder.classes_):,}")
        st.metric(t('features', lang), model_info.get('num_features', 'N/A'))
    with col_m2:
        st.metric(t('precision', lang), f"{model_info.get('accuracy', 0.92):.1%}")
        st.metric(t('embeddings', lang), t('no', lang))

    st.divider()
    st.subheader(t('search_family', lang))
    familias = encoder.classes_
    search_fam = st.text_input(t('family_name', lang), placeholder=t('family_placeholder', lang))
    filtered = [f for f in familias if search_fam.upper() in f.upper()] if search_fam else []
    if search_fam:
        if filtered:
            for fam in filtered[:30]:
                pfam_info = get_pfam_info(fam)
                if pfam_info:
                    st.write(f"**{fam}** - {pfam_info.get('nombre', '')}")
                else:
                    st.write(f" {fam}")
            if len(filtered) > 30:
                st.write(t('and_more', lang, count=len(filtered) - 30))
        else:
            st.warning(t('no_families_found', lang))

    st.divider()
    st.subheader(t('search_aa', lang))
    aa_search = st.text_input(t('aa_code_name', lang), placeholder=t('aa_placeholder', lang))
    if aa_search:
        matches = []
        for code, info in AMINOACIDOS_INFO.items():
            if (aa_search.upper() in code.upper()
                    or aa_search.lower() in info['nombre'].lower()
                    or aa_search.lower() in info['abreviatura'].lower()):
                matches.append((code, info))
        if matches:
            for code, info in matches:
                tipo_key = get_aa_tipo_key(code)
                emoji = TIPO_EMOJI.get(tipo_key, '\u2b1c')
                esencial = t('yes', lang) if info.get('esencial') else t('no', lang)
                st.write(f"{emoji} **{code}** - {info['nombre']} ({info['abreviatura']})")
                st.caption(f"{t(tipo_key, lang)} | PM: {info['peso_molecular']} Da | {t('charge', lang)} {info['carga']} | {t('essential_label', lang)} {esencial}")
                with st.expander(t('more_info_about', lang, name=info['nombre'])):
                    st.write(f"**{t('description', lang)}** {info.get('descripcion', '')}")
                    st.write(f"**{t('where_found', lang)}** {info.get('donde_se_encuentra', '')}")
                    st.write(f"**{t('biological_function', lang)}** {info.get('funcion_biologica', '')}")
                    st.write(f"**{t('chemical_properties', lang)}** {info.get('propiedades_quimicas', '')}")
                    st.write(f"**{t('hydrophobicity', lang)}** {info.get('hidrofobicidad', '')}")
                    st.write(f"**{t('related_diseases', lang)}** {info.get('enfermedades_relacionadas', '')}")
        else:
            st.warning(t('aa_not_found', lang))
    else:
        st.caption(t('type_legend', lang))
        for tipo_key, emoji in TIPO_EMOJI.items():
            st.write(f"{emoji} {t(tipo_key, lang)}")

# ── MAIN AREA ──

st.title(t('main_title', lang))
st.caption(t('main_caption', lang))

st.divider()

st.subheader(t('enter_sequence', lang))

secuencia_input = st.text_area(
    t('sequence_label', lang),
    height=120,
    placeholder=t('sequence_placeholder', lang),
    help=t('sequence_help', lang)
)

if secuencia_input.strip():
    secuencia_limpia = ''.join(
        c for c in secuencia_input.upper()
        if c in 'ACDEFGHIKLMNPQRSTVWY'
    )

    if secuencia_limpia:
        st.write(f"**{t('analyzed_sequence', lang)}**")
        st.code(format_seq_colored(secuencia_limpia), language=None)

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric(t('length', lang), f"{len(secuencia_limpia)} aa")
        with col_s2:
            peso = len(secuencia_limpia) * 110
            st.metric(t('approx_weight', lang), f"{peso:,} Da")
        with col_s3:
            pct_charged = ((secuencia_limpia.count('K') + secuencia_limpia.count('R') + secuencia_limpia.count('H') + secuencia_limpia.count('D') + secuencia_limpia.count('E')) / len(secuencia_limpia)) * 100
            st.metric(t('charged', lang), f"{pct_charged:.1f}%")
        with col_s4:
            pct_hydrophobic = sum(secuencia_limpia.count(aa) for aa in 'AILMFVW') / len(secuencia_limpia) * 100
            st.metric(t('hydrophobic', lang), f"{pct_hydrophobic:.1f}%")

        if st.button(t('predict_button', lang), type="primary", use_container_width=True):
            if secuencia_input.strip():
                secuencia_limpia = ''.join(
                    c for c in secuencia_input.upper()
                    if c in 'ACDEFGHIKLMNPQRSTVWY'
                )

                if len(secuencia_limpia) < 10:
                    st.error(t('sequence_too_short', lang))
                else:
                    with st.spinner(t('analyzing', lang)):
                        familia, top = predecir_familia(secuencia_limpia)

                    st.success(f"{t('predicted_family', lang)} **{familia}** | {t('confidence', lang)} **{top[0][1]:.1%}**")

                    st.subheader(t('family_info', lang))
                    pfam_info = get_pfam_info(familia)
                    if pfam_info:
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.write(f"**{t('name', lang)}** {pfam_info.get('nombre', '')}")
                            st.write(f"**{t('type', lang)}** {pfam_info.get('tipo', '')}")
                            st.write(f"**{t('organisms', lang)}** {pfam_info.get('organismos', '')}")
                        with col_info2:
                            st.write(f"**{t('function', lang)}** {pfam_info.get('funcion', '')}")
                            st.info(pfam_info.get('descripcion', ''))
                    else:
                        st.info(t('family_not_available', lang, family=familia))

                    st.subheader(t('top5_predictions', lang))
                    df_probs = pd.DataFrame({
                        t('family_col', lang): [fam_t[0] for fam_t in top],
                        t('probability_col', lang): [fam_t[1] for fam_t in top],
                    })

                    col_p1, col_p2 = st.columns([1, 1])
                    with col_p1:
                        st.bar_chart(df_probs.set_index(t('family_col', lang)), use_container_width=True)

                    with col_p2:
                        for i, (fam, prob) in enumerate(top):
                            pfam_info_top = get_pfam_info(fam)
                            fam_name = pfam_info_top.get('nombre', '') if pfam_info_top else ''
                            label = f"**{fam}**" + (f" - {fam_name}" if fam_name else "")
                            st.write(label)
                            st.progress(min(prob, 1.0), text=f"{prob:.1%}")

                    st.dataframe(
                        df_probs.style.format({t('probability_col', lang): "{:.2%}"}),
                        use_container_width=True,
                        hide_index=True
                    )
    else:
        st.warning(t('enter_to_predict', lang))

if secuencia_input.strip():
    secuencia_limpia = ''.join(
        c for c in secuencia_input.upper()
        if c in 'ACDEFGHIKLMNPQRSTVWY'
    )

    if secuencia_limpia:
        st.divider()
        st.subheader(t('composition_analysis', lang))

        composicion = {
            aa: secuencia_limpia.count(aa) / len(secuencia_limpia) * 100
            for aa in AMINOACIDOS
        }

        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            st.write(f"**{t('composition_by_aa', lang)}**")
            df_comp = pd.DataFrame({
                t('family_col', lang): [f"{aa} ({AMINOACIDOS_INFO[aa]['abreviatura']})" for aa in AMINOACIDOS],
                t('composition_pct', lang): [composicion[aa] for aa in AMINOACIDOS]
            })
            st.bar_chart(df_comp.set_index(t('family_col', lang)), use_container_width=True)

        with col_c2:
            st.write(f"**{t('distribution_by_type', lang)}**")
            tipos_count = {}
            for aa in AMINOACIDOS:
                tipo_key = get_aa_tipo_key(aa)
                tipos_count[tipo_key] = tipos_count.get(tipo_key, 0) + secuencia_limpia.count(aa)

            tipos_pct = {k: v / len(secuencia_limpia) * 100 for k, v in tipos_count.items()}
            df_tipos = pd.DataFrame({
                t('type', lang): [f"{TIPO_EMOJI.get(tk, '')} {t(tk, lang)}" for tk in tipos_pct.keys()],
                t('percentage', lang): list(tipos_pct.values())
            })
            st.bar_chart(df_tipos.set_index(t('type', lang)), use_container_width=True)

        st.subheader(t('top5_aa', lang))
        top_aa = sorted(composicion.items(), key=lambda x: x[1], reverse=True)
        significant_aa = [(aa, pct) for aa, pct in top_aa if pct > 0]

        aa_cols = st.columns(min(len(significant_aa), 5))
        for idx, (aa, pct) in enumerate(significant_aa[:5]):
            with aa_cols[idx]:
                info = AMINOACIDOS_INFO.get(aa, {})
                tipo_key = get_aa_tipo_key(aa)
                emoji = TIPO_EMOJI.get(tipo_key, '')
                esencial = t('essential', lang) if info.get('esencial') else t('not_essential', lang)
                st.metric(
                    label=f"{emoji} {aa} - {info.get('nombre', aa)}",
                    value=f"{pct:.1f}%",
                    delta=f"{t(tipo_key, lang)} | {esencial}"
                )

        with st.expander(t('view_full_details', lang)):
            for aa, pct in significant_aa:
                info = AMINOACIDOS_INFO.get(aa, {})
                if not info:
                    continue
                tipo_key = get_aa_tipo_key(aa)
                emoji = TIPO_EMOJI.get(tipo_key, '')
                esencial = t('yes', lang) if info.get('esencial') else t('no', lang)

                st.write(f"### {emoji} {aa} - {info['nombre']} ({info['abreviatura']}) — **{pct:.1f}%**")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write(f"- **{t('type', lang)}** {info['tipo']}")
                    st.write(f"- **{t('molecular_weight', lang)}** {info['peso_molecular']} Da")
                    st.write(f"- **{t('charge', lang)}** {info['carga']}")
                    st.write(f"- **{t('hydrophobicity', lang)}** {info.get('hidrofobicidad', '')}")
                    st.write(f"- **{t('essential_label', lang)}** {esencial}")
                with col_d2:
                    st.write(f"- **{t('description', lang)}** {info.get('descripcion', '')}")
                    st.write(f"- **{t('where_found', lang)}** {info.get('donde_se_encuentra', '')}")
                    st.write(f"- **{t('biological_function', lang)}** {info.get('funcion_biologica', '')}")
                    st.write(f"- **{t('chemical_properties', lang)}** {info.get('propiedades_quimicas', '')}")
                    st.write(f"- **{t('related_diseases', lang)}** {info.get('enfermedades_relacionadas', '')}")
                st.divider()

st.divider()
st.caption(t('footer', lang))
