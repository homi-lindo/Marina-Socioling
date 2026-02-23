import streamlit as st
import pandas as pd
import subprocess
import os
import json
import plotly.express as px

st.set_page_config(
    page_title="Marina-Socioling",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DADOS_PATH = os.environ.get("DADOS_PATH", "/app/dados")
RBRUL_SCRIPT = "/app/scripts/run_rbrul.R"
TODAS_METRICAS = ["npw_pmi", "npw_pmi2", "npw_pmi3", "npw_npmi", "npw_llr"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def detectar_binaria(df):
    for col in df.columns:
        if df[col].nunique() == 2:
            return col
    return None

def detectar_texto(df):
    for col in df.columns:
        if df[col].dtype == object:
            media_palavras = df[col].dropna().str.split().str.len().mean()
            if media_palavras and media_palavras > 3:
                return col
    return None

def detectar_falante(df):
    candidatos = ["falante", "speaker", "informante", "participante", "sujeito", "id"]
    for col in df.columns:
        if col.lower() in candidatos:
            return col
    return None

def detectar_variavel_social(df, excluir):
    candidatos = ["genero", "gênero", "faixa_etaria", "faixa etária", "escolaridade",
                  "regiao", "região", "estilo", "classe", "sexo", "grupo"]
    for col in df.columns:
        if col.lower() in candidatos and col not in excluir:
            return col
    # fallback: primeira coluna categórica não excluída
    for col in df.columns:
        if col not in excluir and df[col].dtype == object and df[col].nunique() < 20:
            return col
    return None

def detectar_idioma(df, col_texto):
    amostra = " ".join(df[col_texto].dropna().head(20).tolist()).lower()
    if any(w in amostra for w in ["der", "die", "das", "und", "ist"]):
        return "de"
    if any(w in amostra for w in ["и", "в", "не", "на", "что"]):
        return "ru"
    if any(w in amostra for w in ["the", "and", "is", "of", "to"]):
        return "en"
    return "pt"

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
    st.info("Faça o upload de um ou mais arquivos para começar.")
    st.stop()

# Concatena todos os arquivos
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
dep_var    = detectar_binaria(df)
col_texto  = detectar_texto(df)
falante    = detectar_falante(df)

tem_rbrul       = dep_var is not None
tem_variationist = col_texto is not None

if not tem_rbrul and not tem_variationist:
    st.error("⚠️ Não foi possível detectar colunas compatíveis com Rbrul ou Variationist.")
    st.stop()

# Exibe o que foi detectado
with st.expander("🔍 Detecção automática", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Rbrul**")
        if tem_rbrul:
            excluir_rbrul = {dep_var, falante or ""}
            fatores = [c for c in colunas if c not in excluir_rbrul]
            rand_eff = falante
            st.success(f"✅ Variável dependente: `{dep_var}`")
            st.info(f"Efeito aleatório: `{rand_eff or 'nenhum'}`")
            st.info(f"Fatores: `{', '.join(fatores)}`")
        else:
            st.warning("Nenhuma coluna binária detectada.")
    with col2:
        st.markdown("**Variationist**")
        if tem_variationist:
            excluir_var = {col_texto}
            col_variavel = detectar_variavel_social(df, excluir_var)
            idioma = detectar_idioma(df, col_texto)
            st.success(f"✅ Coluna de texto: `{col_texto}`")
            st.info(f"Variável social: `{col_variavel or 'não detectada'}`")
            st.info(f"Idioma detectado: `{idioma}`")
        else:
            st.warning("Nenhuma coluna de texto detectada.")

st.markdown("---")

# ── Botão único ───────────────────────────────────────────────────────────────
if st.button("▶ Rodar análise completa", type="primary", use_container_width=True):

    # ═══════════════════════════════════════════════════════════════════════════
    # RBRUL
    # ═══════════════════════════════════════════════════════════════════════════
    if tem_rbrul:
        st.subheader("📊 Rbrul — Regressão Logística Variacionista")
        with st.spinner("Executando modelo de regressão logística em R..."):
            tmp_csv = os.path.join(DADOS_PATH, "rbrul_input.csv")
            df[colunas].to_csv(tmp_csv, index=False)

            cmd = ["Rscript", RBRUL_SCRIPT, tmp_csv, dep_var, ",".join(fatores)]
            if rand_eff:
                cmd.append(rand_eff)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            st.success("✅ Rbrul concluído!")
            st.code(result.stdout, language="r")
            st.download_button(
                "⬇️ Baixar output Rbrul (.txt)",
                data=result.stdout,
                file_name="rbrul_output.txt",
                mime="text/plain",
            )
        else:
            st.error("❌ Erro no Rbrul.")
            st.code(result.stderr, language="bash")

    # ═══════════════════════════════════════════════════════════════════════════
    # VARIATIONIST
    # ═══════════════════════════════════════════════════════════════════════════
    if tem_variationist and col_variavel:
        st.subheader("📈 Variationist — Métricas de Associação")
        with st.spinner("Calculando todas as métricas de variação..."):
            try:
                from variationist import Inspector, InspectorArgs, Visualizer, VisualizerArgs

                tmp_tsv = os.path.join(DADOS_PATH, "variationist_input.tsv")
                df[[col_texto, col_variavel]].to_csv(tmp_tsv, sep="\t", index=False)

                ins_args = InspectorArgs(
                    text_names=[col_texto],
                    var_names=[col_variavel],
                    metrics=TODAS_METRICAS,
                    n_tokens=1,
                    language=idioma,
                    stopwords=False,
                    lowercase=True,
                )

                res = Inspector(dataset=tmp_tsv, args=ins_args).inspect()
                st.success("✅ Variationist concluído!")

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
                                             color="variante", title=f"{metrica} — Top 20")
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            st.json(res.get(metrica, {}))

                charts_path = os.path.join(DADOS_PATH, "charts")
                os.makedirs(charts_path, exist_ok=True)
                vis_args = VisualizerArgs(output_folder=charts_path, output_formats=["html"])
                Visualizer(input_json=res, args=vis_args).create()

                st.download_button(
                    "⬇️ Baixar resultados (.json)",
                    data=json.dumps(res, indent=2, ensure_ascii=False),
                    file_name="variationist_output.json",
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"❌ Erro no Variationist: {e}")
                st.exception(e)

    elif tem_variationist and not col_variavel:
        st.warning("⚠️ Variationist: nenhuma variável social detectada no arquivo.")
