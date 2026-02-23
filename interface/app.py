import streamlit as st
import pandas as pd
import subprocess
import os
import json
import re
import plotly.express as px

st.set_page_config(
    page_title="Marina-Socioling",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DADOS_PATH  = os.environ.get("DADOS_PATH", "/app/dados")
RBRUL_SCRIPT = "/app/scripts/run_rbrul.R"

METRICAS_PMI     = ["pmi", "n_pmi", "p_pmi", "np_pmi", "w_pmi", "nw_pmi",
                    "pw_pmi", "npw_pmi", "np_relevance", "nw_relevance", "npw_relevance"]
METRICAS_LEXICAIS = ["ttr", "root_ttr", "maas", "log_ttr"]
METRICAS_CORPUS   = ["freq", "stats"]
TODAS_METRICAS    = METRICAS_PMI + METRICAS_LEXICAIS + METRICAS_CORPUS

CANDIDATOS_SOCIAIS = [
    "genero", "gênero", "faixa_etaria", "faixa etária", "escolaridade",
    "regiao", "região", "estilo", "classe", "sexo", "grupo"
]
CANDIDATOS_RAND = [
    "falante", "speaker", "informante", "participante", "sujeito",
    "palavra", "item", "item_lexical", "lexema", "vocábulo",
    "entrevista", "sessao", "sessão", "gravacao", "gravação",
    "comunidade", "cidade", "bairro", "localidade"
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def detectar_binárias(df):
    return [c for c in df.columns if df[c].nunique() == 2]

def detectar_textos(df):
    cols = []
    for col in df.columns:
        if df[col].dtype == object:
            media = df[col].dropna().str.split().str.len().mean()
            if media and media > 3:
                cols.append(col)
    return cols

def detectar_efeitos_aleatorios(df):
    return [col for col in df.columns if col.lower() in CANDIDATOS_RAND]

def detectar_sociais(df, excluir):
    """Retorna todas as colunas que parecem variáveis sociais — incluindo binárias."""
    sociais = []
    for col in df.columns:
        if col in excluir:
            continue
        if col.lower() in CANDIDATOS_SOCIAIS:
            sociais.append(col)
    if not sociais:
        for col in df.columns:
            if col not in excluir and df[col].dtype == object and 2 <= df[col].nunique() <= 10:
                sociais.append(col)
    return sociais

def corrigir_texto(df, col):
    df[col] = df[col].astype(str).apply(
        lambda x: re.sub(r'([a-záàãâéêíóôõúüçA-Z])([A-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ])', r'\1 \2', x)
    )
    return df

def detectar_idioma(df, cols_texto):
    amostra = ""
    for col in cols_texto:
        amostra += " " + " ".join(df[col].dropna().head(50).tolist()).lower()
    amostra_padded = f" {amostra} "

    # Cirílico → russo imediato
    if sum(1 for c in amostra if '\u0400' <= c <= '\u04FF') > 10:
        return "ru"
    # Umlauts/ß → alemão imediato
    if sum(1 for c in amostra if c in "äöüßÄÖÜ") > 10:
        return "de"

    stopwords = {
        "pt": [
            "que", "não", "uma", "com", "para", "mais", "por", "ele", "ela", "mas",
            "como", "seu", "sua", "dos", "das", "nos", "nas", "isso", "esse", "essa",
            "este", "esta", "aqui", "então", "também", "já", "muito", "bem", "quando",
            "onde", "quem", "qual", "porque", "assim", "ainda", "depois", "antes",
            "sempre", "nunca", "havia", "seria", "estava", "foram", "tinha", "tudo",
            "nada", "cada", "outro", "outra", "gente", "hoje", "agora", "aquele",
            "aquela", "nosso", "nossa", "vocês", "eles", "elas", "sobre", "entre", "até"
        ],
        "en": [
            "the", "and", "is", "of", "to", "in", "that", "it", "was", "for",
            "on", "are", "with", "his", "they", "at", "be", "this", "from", "or",
            "had", "by", "not", "but", "what", "all", "were", "when", "we", "there",
            "can", "an", "your", "which", "their", "said", "do", "into", "has",
            "more", "her", "him", "time", "could", "make", "than", "been", "would",
            "who", "will", "my", "one", "about", "up", "out", "so", "them"
        ],
        "de": [
            "und", "ist", "ich", "nicht", "mit", "ein", "eine", "auch", "auf",
            "sich", "als", "dem", "des", "zu", "bei", "war", "von", "aber", "noch",
            "wird", "sie", "nach", "wie", "für", "haben", "kann", "doch", "hier",
            "wenn", "dann", "wir", "man", "aus", "durch", "mehr", "oder", "hat",
            "ihm", "ihr", "uns", "ihn", "diese", "dieser", "dieses", "werden", "sein"
        ],
        "ru": [
            "и", "в", "не", "на", "что", "он", "она", "это", "как", "но",
            "его", "её", "они", "мы", "вы", "же", "от", "за", "по", "из",
            "уже", "так", "был", "была", "было", "были", "есть", "нет", "да", "бы",
            "если", "то", "все", "или", "когда", "их", "там", "где", "кто", "чем"
        ],
    }
    bigramas = {
        "pt": ["de que", "para o", "para a", "não é", "que não", "é que",
               "a gente", "vai ser", "que eu", "eu não", "eu tô", "tá tudo"],
        "en": ["of the", "in the", "to the", "is a", "it is", "there is",
               "do not", "i am", "you are", "he was", "she was"],
        "de": ["in der", "auf der", "ist ein", "ist die", "nicht die",
               "ich bin", "es gibt", "haben wir", "das ist", "ich habe"],
        "ru": ["не было", "что это", "он был", "она была", "в том",
               "это был", "как это", "на это", "из них"],
    }

    scores = {lang: 0 for lang in stopwords}
    for lang, words in stopwords.items():
        for w in words:
            if f" {w} " in amostra_padded:
                scores[lang] += 1
    for lang, bigrams in bigramas.items():
        for bg in bigrams:
            if f" {bg} " in amostra_padded:
                scores[lang] += 2

    melhor = max(scores, key=scores.get)
    return melhor if scores[melhor] > 0 else "pt"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Marina-Socioling")
st.caption("Rbrul — Johnson (2009) · Variationist — ACL 2024")
st.markdown("---")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload de arquivos CSV ou TXT",
    type=["csv", "txt"],
    accept_multiple_files=True,
    help="CSV ou TXT com cabeçalho. Separador: vírgula (CSV) ou tabulação (TXT).",
)

if not uploaded_files:
    st.info("Faça o upload de um ou mais arquivos CSV/TXT para começar.")
    st.stop()

dfs = []
for f in uploaded_files:
    _df = pd.read_csv(f, sep="\t" if f.name.endswith(".txt") else ",")
    _df["_arquivo"] = f.name
    dfs.append(_df)

df = pd.concat(dfs, ignore_index=True)
colunas = [c for c in df.columns if c != "_arquivo"]

with st.expander(f"Pré-visualização — {len(df)} linhas · {len(colunas)} colunas · {len(uploaded_files)} arquivo(s)"):
    st.dataframe(df.head(20), use_container_width=True)

st.markdown("---")

# ── Detecção automática ───────────────────────────────────────────────────────
binárias       = detectar_binárias(df)
cols_texto     = detectar_textos(df)
rand_detectados = detectar_efeitos_aleatorios(df)
excluir_sociais = set(cols_texto)
sociais        = detectar_sociais(df, excluir_sociais)
idioma         = detectar_idioma(df, cols_texto) if cols_texto else "pt"

tem_rbrul        = len(binárias) > 0
tem_variationist = len(cols_texto) > 0 and len(sociais) > 0

with st.expander("Detecção automática", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Rbrul**")
        if tem_rbrul:
            st.success(f"Variáveis dependentes: {len(binárias)}")
            for b in binárias:
                st.info(f"`{b}` — valores: {df[b].unique().tolist()}")

            # Checkboxes para efeitos aleatórios
            st.markdown("**Efeitos aleatórios**")
            rand_selecionados = []
            if rand_detectados:
                for col in rand_detectados:
                    if st.checkbox(f"`{col}`", value=True, key=f"rand_{col}"):
                        rand_selecionados.append(col)
            else:
                st.info("Nenhum efeito aleatório detectado.")

            excluir_fatores = set(binárias) | set(rand_detectados) | set(cols_texto)
            fatores_globais = [c for c in colunas if c not in excluir_fatores]
            st.info(f"Fatores: `{', '.join(fatores_globais)}`")
        else:
            st.warning("Nenhuma coluna binária detectada.")

    with col2:
        st.markdown("**Variationist**")
        if tem_variationist:
            st.success(f"Colunas de texto: {cols_texto}")
            st.success(f"Variáveis sociais: {sociais}")
            st.info(f"Idioma detectado: `{idioma}`")
        else:
            st.warning("Nenhuma coluna de texto ou variável social detectada.")

st.markdown("---")

# ── Botão único ───────────────────────────────────────────────────────────────
if st.button("▶ Rodar análise completa", type="primary", use_container_width=True):

    # ═══════════════════════════════════════════════════════════════════════════
    # RBRUL — um modelo por variável dependente binária
    # ═══════════════════════════════════════════════════════════════════════════
    if tem_rbrul:
        st.subheader("Rbrul — Regressão Logística Variacionista")

        for dep_var in binárias:
            st.markdown(f"#### Modelo: `{dep_var}`")
            fatores = [c for c in colunas
                       if c != dep_var
                       and c not in rand_detectados
                       and c not in cols_texto]

            with st.spinner(f"Rodando modelo para `{dep_var}`..."):
                tmp_csv = os.path.join(DADOS_PATH, f"rbrul_{dep_var}.csv")
                df[colunas].to_csv(tmp_csv, index=False)

                rand_arg = ",".join(rand_selecionados) if rand_selecionados else ""
                cmd = ["Rscript", RBRUL_SCRIPT, tmp_csv, dep_var, ",".join(fatores)]
                if rand_arg:
                    cmd.append(rand_arg)

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                st.success(f"Modelo `{dep_var}` concluído!")
                st.code(result.stdout, language="r")
                st.download_button(
                    f"Baixar output — {dep_var} (.txt)",
                    data=result.stdout,
                    file_name=f"rbrul_{dep_var}.txt",
                    mime="text/plain",
                    key=f"dl_rbrul_{dep_var}",
                )
            else:
                st.error(f"❌ Erro no modelo `{dep_var}`.")
                st.code(result.stderr, language="bash")

    # ═══════════════════════════════════════════════════════════════════════════
    # VARIATIONIST — uma análise por variável social, todas as métricas
    # ═══════════════════════════════════════════════════════════════════════════
    if tem_variationist:
        st.subheader("Variationist — Métricas de Associação")

        for col_texto in cols_texto:
            df = corrigir_texto(df, col_texto)

        for col_variavel in sociais:
            st.markdown(f"#### Variável social: `{col_variavel}`")

            with st.spinner(f"Calculando métricas para `{col_variavel}`..."):
                try:
                    from variationist import Inspector, InspectorArgs, Visualizer, VisualizerArgs

                    col_texto_atual = cols_texto[0]
                    tmp_tsv = os.path.join(DADOS_PATH, f"variationist_{col_variavel}.tsv")
                    df[[col_texto_atual, col_variavel]].to_csv(tmp_tsv, sep="\t", index=False)

                    ins_args = InspectorArgs(
                        text_names=[col_texto_atual],
                        var_names=[col_variavel],
                        metrics=TODAS_METRICAS,
                        n_tokens=1,
                        language=idioma,
                        stopwords=True,
                        lowercase=True,
                    )

                    res = Inspector(dataset=tmp_tsv, args=ins_args).inspect()
                    st.success(f"Análise `{col_variavel}` concluída!")

                    tabs = st.tabs(TODAS_METRICAS)
                    for tab, metrica in zip(tabs, TODAS_METRICAS):
                        with tab:
                            try:
                                dados_metrica = res.get(metrica, {})
                                rows = []
                                for token, variantes in list(dados_metrica.items())[:20]:
                                    for variante, score in variantes.items():
                                        rows.append({"token": token, "variante": variante, "score": score})
                                if rows:
                                    df_tab = pd.DataFrame(rows).sort_values("score", ascending=False)
                                    st.dataframe(df_tab, use_container_width=True)
                                    fig = px.bar(df_tab.head(20), x="token", y="score",
                                                 color="variante",
                                                 title=f"{metrica} · {col_variavel} — Top 20")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Sem dados para esta métrica.")
                            except Exception:
                                st.json(res.get(metrica, {}))

                    charts_path = os.path.join(DADOS_PATH, "charts")
                    os.makedirs(charts_path, exist_ok=True)
                    vis_args = VisualizerArgs(output_folder=charts_path, output_formats=["html"])
                    Visualizer(input_json=res, args=vis_args).create()

                    st.download_button(
                        f"Baixar resultados — {col_variavel} (.json)",
                        data=json.dumps(res, indent=2, ensure_ascii=False),
                        file_name=f"variationist_{col_variavel}.json",
                        mime="application/json",
                        key=f"dl_var_{col_variavel}",
                    )

                except Exception as e:
                    st.error(f"❌ Erro em `{col_variavel}`: {e}")
                    st.exception(e)
