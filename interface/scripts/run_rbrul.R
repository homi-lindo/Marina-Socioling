pkgs <- c("lme4", "boot")
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if (length(new_pkgs) > 0) {
  install.packages(new_pkgs, repos="https://cran.r-project.org", quiet=TRUE)
}

#!/usr/bin/env Rscript
# Uso: Rscript run_rbrul.R <csv_path> <dep_var> <fator1,fator2,...> [random_effect]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Uso: Rscript run_rbrul.R <csv> <dep_var> <fatores_sep_virgula> [random_effect]")
}

csv_path   <- args[1]
dep_var    <- args[2]
fatores    <- unlist(strsplit(args[3], ","))
rand_eff   <- if (length(args) >= 4) args[4] else NULL

cat("=== Carregando dados ===\n")
dados <- read.csv(csv_path, stringsAsFactors = TRUE)
cat(sprintf("Linhas: %d | Colunas: %d\n", nrow(dados), ncol(dados)))

# Carrega pacotes
suppressPackageStartupMessages({
  library(lme4)
  library(boot)
})

cat("\n=== Executando Regressão Logística (Regras Variáveis) ===\n")

# Monta fórmula
formula_str <- paste(dep_var, "~", paste(fatores, collapse = " + "))
if (!is.null(rand_eff)) {
  formula_str <- paste(formula_str, "+ (1|", rand_eff, ")")
  cat("Modelo: Efeitos Mistos (lme4::glmer)\n")
  modelo <- glmer(as.formula(formula_str), data = dados,
                  family = binomial, control = glmerControl(optimizer = "bobyqa"))
} else {
  cat("Modelo: Regressão Logística Simples (glm)\n")
  modelo <- glm(as.formula(formula_str), data = dados, family = binomial,
                contrasts = setNames(lapply(fatores, function(x) "contr.sum"), fatores))
}

cat("\n=== Sumário do Modelo ===\n")
print(summary(modelo))

cat("\n=== Pesos de Fator (Factor Weights) ===\n")
coefs <- fixef(if (!is.null(rand_eff)) modelo else modelo)
weights <- round(boot::inv.logit(coefs), 3)
print(data.frame(Coeficiente = coefs, Peso_de_Fator = weights))

cat("\n=== Log-Likelihood ===\n")
cat(sprintf("LogLik: %.4f\n", as.numeric(logLik(modelo))))
