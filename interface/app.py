import streamlit as st
import pandas as pd
import subprocess
import tempfile
import os
import plotly.express as px

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marina-Socioling",
    page_icon="🔬",
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
        ["📊 Rbrul", "🧩 Variationist"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Rbrul — Johnson (2009)")
    st.caption("Variationist — ACL 2024")

# ── Upload ────────────────────────────────────────────────────────────────────
st.header("Upload de Dados")
uploaded = st.file_uploader(
    "Carregue seu arquivo CSV (cada linha = 1 token)",
    type=["csv"],
    help="O arquivo deve ter cabeçalho. Cada coluna é uma variável.",
)

if not uploaded:
    st.info("⬆️ Faça o upload de um CSV para começar.")
    st.stop()

df = pd.read_csv(uploaded)
colunas = df.columns.tolist()

with st.expander("👁️ Pré-visualização dos dados", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"{len(df)} linhas · {len(colunas)} colunas")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# RBRUL
# ═════════════════════════════════════════════════════════════════════════════
if ferramenta == "📊 Rbrul":
    st.header("📊 Análise com Rbrul")
    st.markdown(
        "Regressão logística variacionista com efeitos mistos, "
        "substituto direto do GoldVarb X."
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
        help="Selecione as variáveis sociais e linguísticas que podem influenciar a variável dependente.",
    )

    if not fatores:
        st.warning("Selecione ao menos um grupo de fatores.")
        st.stop()

    # Distribuição da variável dependente
    with st.expander("📈 Distribuição da variável dependente"):
        fig = px.histogram(df, x=dep_var, color=dep_var, title=f"Distribuição: {dep_var}")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("▶ Rodar Rbrul", type="primary", use_container_width=True):
        with st.spinner("Executando modelo de regressão logística em R..."):
            # Salva CSV temporário
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

        # Salva output
        out_path = os.path.join(DADOS_PATH, "rbrul_output.txt")
        with open(out_path, "w") as f:
            f.write(result.stdout)
        st.download_button(
            "⬇️ Baixar output (.txt)",
            data=result.stdout,
            file_name="rbrul_output.txt",
            mime="text/plain",
        )

# ═════════════════════════════════════════════════════════════════════════════
# VARIATIONIST
# ═════════════════════════════════════════════════════════════════════════════
elif ferramenta == "🧩 Variationist":
    st.header("🧩 Análise com Variationist")
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
        ["frequency", "pmi", "npmi", "tf-idf"],
        help="Como medir a associação entre texto e variável.",
    )

    top_n = st.slider("Top N unidades para exibir", 5, 50, 20)

    if st.button("▶ Rodar Variationist", type="primary", use_container_width=True):
        with st.spinner("Calculando métricas de variação..."):
            try:
                from variationist import VarAnalyzer

                analyzer = VarAnalyzer(
                    df=df,
                    text_cols=[col_texto],
                    var_cols=[col_variavel],
                    metrics=[metrica],
                    n_tokens=1,
                )
                results = analyzer.compute()

                st.success("✅ Análise concluída!")

                # Exibe tabela de resultados
                if hasattr(results, "to_dataframe"):
                    df_res = results.to_dataframe()
                elif isinstance(results, pd.DataFrame):
                    df_res = results
                else:
                    df_res = pd.DataFrame(results)

                st.subheader("Resultados")
                st.dataframe(df_res.head(top_n), use_container_width=True)

                # Gráfico de barras dos top N
                if len(df_res) > 0:
                    cols_num = df_res.select_dtypes("number").columns.tolist()
                    if cols_num:
                        fig2 = px.bar(
                            df_res.head(top_n),
                            x=df_res.columns[0],
                            y=cols_num[0],
                            title=f"Top {top_n} — {metrica.upper()} por {col_variavel}",
                            color=cols_num[0],
                            color_continuous_scale="Blues",
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                # Download
                csv_out = df_res.to_csv(index=False)
                st.download_button(
                    "⬇️ Baixar resultados (.csv)",
                    data=csv_out,
                    file_name="variationist_output.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Erro: {e}")
                st.exception(e)
