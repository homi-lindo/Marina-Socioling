pkgs <- c("lme4", "boot")
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if (length(new_pkgs) > 0) {
  install.packages(new_pkgs, repos="https://cran.r-project.org", quiet=TRUE)
}

#!/usr/bin/env Rscript
# Uso: Rscript run_rbrul.R <csv> <dep_var> <fatores> [efeitos_aleatorios]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Uso: Rscript run_rbrul.R <csv> <dep_var> <fatores> [efeitos_aleatorios]")
}

csv_path  <- args[1]
dep_var   <- args[2]
fatores   <- unlist(strsplit(args[3], ","))
rand_effs <- if (length(args) >= 4 && nchar(args[4]) > 0)
               unlist(strsplit(args[4], ",")) else NULL

cat("=== Carregando dados ===\n")
dados <- read.csv(csv_path, stringsAsFactors = TRUE)

# Garante que fatores e dep_var são factor
for (col in c(fatores, dep_var)) {
  if (col %in% names(dados)) dados[[col]] <- as.factor(dados[[col]])
}

cat(sprintf("Linhas: %d | Colunas: %d\n", nrow(dados), ncol(dados)))

suppressPackageStartupMessages({
  library(lme4)
  library(boot)
})

cat("\n=== Executando Regressão Logística (Regras Variáveis) ===\n")

formula_str <- paste(dep_var, "~", paste(fatores, collapse = " + "))

if (!is.null(rand_effs)) {
  rand_terms  <- paste(sapply(rand_effs, function(r) paste0("(1|", r, ")")), collapse = " + ")
  formula_str <- paste(formula_str, "+", rand_terms)
  cat(sprintf("Modelo: Efeitos Mistos — %s\n", paste(rand_effs, collapse = ", ")))
  modelo <- glmer(as.formula(formula_str), data = dados,
                  family = binomial, control = glmerControl(optimizer = "bobyqa"))
} else {
  cat("Modelo: Regressão Logística Simples (glm)\n")
  # Aplica contr.sum apenas em fatores válidos (is.factor com 2+ níveis)
  fatores_validos <- fatores[sapply(fatores, function(x)
    x %in% names(dados) && is.factor(dados[[x]]) && nlevels(dados[[x]]) >= 2
  )]
  modelo <- glm(as.formula(formula_str), data = dados, family = binomial,
                contrasts = setNames(lapply(fatores_validos, function(x) "contr.sum"), fatores_validos))
}

cat("\n=== Sumário do Modelo ===\n")
print(summary(modelo))

cat("\n=== Pesos de Fator (Factor Weights) ===\n")
coefs   <- fixef(if (!is.null(rand_effs)) modelo else modelo)
weights <- round(boot::inv.logit(coefs), 3)
print(data.frame(Coeficiente = coefs, Peso_de_Fator = weights))

cat("\n=== Log-Likelihood ===\n")
cat(sprintf("LogLik: %.4f\n", as.numeric(logLik(modelo))))
