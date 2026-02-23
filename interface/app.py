import streamlit as st
import pandas as pd
import subprocess
import os
import json
import plotly.express as px

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marina-Socioling",
    layout="wide",
    initial_sidebar_state="expanded",
)

DADOS_PATH = os.environ.get("DADOS_PATH", "/app/dados")
RBRUL_SCRIPT = "/app/scripts/run_rbrul.R"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Marina-Socioling")
    st.markdown("---")
    ferramenta = st.radio(
        "Selecione a ferramenta",
        ["Rbrul", "Variationist"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Rbrul — Johnson (2009)")
    st.caption("Variationist — ACL 2024")

# ── Upload ────────────────────────────────────────────────────────────────────
st.header("Upload de Dados")
uploaded = st.file_uploader(
    "Upload: CSV ou TXT (cada linha = 1 token)",
    type=["csv", "txt"],
    help="CSV ou TXT com cabeçalho. Separador: vírgula (CSV) ou tabulação (TXT).",
)

if not uploaded:
    st.info("Faça o upload de um CSV ou TXT para começar.")
    st.stop()

if uploaded.name.endswith(".txt"):
    df = pd.read_csv(uploaded, sep="\t")
else:
    df = pd.read_csv(uploaded)

colunas = df.columns.tolist()

with st.expander("Pré-visualização dos dados", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"{len(df)} linhas · {len(colunas)} colunas")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# RBRUL
# ═════════════════════════════════════════════════════════════════════════════
if ferramenta == "Rbrul":
    st.header("Análise com Rbrul")
    st.markdown(
        "Regressão logística variacionista com efeitos mistos, "
    )

    col1, col2 = st.columns(2)
    with col1:
        dep_var = st.selectbox(
            "Variável dependente (binária)",
            colunas,
            help="Coluna com os valores que você quer modelar (ex: presença/ausência).",
        )
    with col2:
        rand_eff = st.selectbox(
            "Efeito aleatório (opcional)",
            ["— nenhum —"] + colunas,
            help="Normalmente o falante/speaker. Ativa modelo de efeitos mistos.",
        )

    fatores = st.multiselect(
        "Grupos de fatores (variáveis independentes)",
        [c for c in colunas if c != dep_var],
        help="Selecione as variáveis sociais e linguísticas.",
    )

    if not fatores:
        st.warning("Selecione ao menos um grupo de fatores.")
        st.stop()

    with st.expander("Distribuição da variável dependente"):
        fig = px.histogram(df, x=dep_var, color=dep_var, title=f"Distribuição: {dep_var}")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("▶ Rodar Rbrul", type="primary", use_container_width=True):
        with st.spinner("Executando modelo de regressão logística em R..."):
            tmp_csv = os.path.join(DADOS_PATH, "rbrul_input.csv")
            df.to_csv(tmp_csv, index=False)

            rand_arg = rand_eff if rand_eff != "— nenhum —" else ""
            cmd = [
                "Rscript", RBRUL_SCRIPT,
                tmp_csv,
                dep_var,
                ",".join(fatores),
            ]
            if rand_arg:
                cmd.append(rand_arg)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            st.success("✅ Análise concluída!")
            st.subheader("Output do Modelo")
            st.code(result.stdout, language="r")
        else:
            st.error("❌ Erro ao executar o Rbrul.")
            st.code(result.stderr, language="bash")

        st.download_button(
            "⬇️ Baixar output (.txt)",
            data=result.stdout,
            file_name="rbrul_output.txt",
            mime="text/plain",
        )

# ═════════════════════════════════════════════════════════════════════════════
# VARIATIONIST
# ═════════════════════════════════════════════════════════════════════════════
elif ferramenta == "Variationist":
    st.header("Análise com Variationist")
    st.markdown(
        "Análise de variação e métricas de associação em corpus textual "
        "([ACL 2024](https://aclanthology.org/2024.acl-demos.33/))."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        col_texto = st.selectbox(
            "Coluna de texto",
            colunas,
            help="Coluna que contém os textos/tokens a serem analisados.",
        )
    with col2:
        col_variavel = st.selectbox(
            "Variável social",
            [c for c in colunas if c != col_texto],
            help="Variável independente para comparação (ex: gênero, faixa etária).",
        )

    metrica = st.selectbox(
        "Métrica de associação",
        ["npw_pmi", "npw_pmi2", "npw_pmi3", "npw_npmi", "npw_llr"],
        help="npw_pmi é a métrica padrão recomendada pelo Variationist.",
    )

    top_n = st.slider("Top N unidades para exibir", 5, 50, 20)

    if st.button("▶ Rodar Variationist", type="primary", use_container_width=True):
        with st.spinner("Calculando métricas de variação..."):
            try:
                from variationist import Inspector, InspectorArgs, Visualizer, VisualizerArgs

                tmp_tsv = os.path.join(DADOS_PATH, "variationist_input.tsv")
                df.to_csv(tmp_tsv, sep="\t", index=False)

                ins_args = InspectorArgs(
                    text_names=[col_texto],
                    var_names=[col_variavel],
                    metrics=[metrica],
                    n_tokens=1,
                    language="pt",
                    stopwords=False,   # True remove stopwords, mas pt pode ter suporte limitado na v0.1.6
                    lowercase=True,
                )


                res = Inspector(dataset=tmp_tsv, args=ins_args).inspect()

                st.success("✅ Análise concluída!")
                st.subheader(f"Top {top_n} resultados")
                st.json(res)

                charts_path = os.path.join(DADOS_PATH, "charts")
                os.makedirs(charts_path, exist_ok=True)
                vis_args = VisualizerArgs(
                    output_folder=charts_path,
                    output_formats=["html"],
                )
                Visualizer(input_json=res, args=vis_args).create()
                st.info(f"📊 Gráficos HTML salvos em {charts_path}")

                st.download_button(
                    "⬇️ Baixar resultados (.json)",
                    data=json.dumps(res, indent=2, ensure_ascii=False),
                    file_name="variationist_output.json",
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"❌ Erro: {e}")
                st.exception(e)
