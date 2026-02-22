# Socioling Suite — Deploy com Docker

Interface web para análise sociolinguística variacionista, integrando
**Rbrul** (regressão logística / regras variáveis) e **Variationist** (métricas de variação em corpus).

```bash
cd Marina-Socioling
docker compose up -d --build

Acesse: http://localhost:8501

## Estrutura

```
socioling-suite/
├── docker-compose.yml
├── .env.example
├── README.md
└── interface/
    ├── Dockerfile
    ├── requirements.txt
    ├── app.py
    ├── .dockerignore
    └── scripts/
        └── run_rbrul.R
```

## Uso

1. Faça upload de um CSV (cada linha = 1 token)
2. Selecione **Rbrul** para análise de regras variáveis (estilo GoldVarb)
3. Selecione **Variationist** para análise de variação em corpus textual
4. Baixe os resultados em .txt ou .csv

## Notas
- O Rbrul usa `lme4::glmer` (efeitos mistos) quando um efeito aleatório é selecionado
- O Variationist suporta métricas: `frequency`, `pmi`, `npmi`, `tf-idf`

https://github.com/homi-lindo/Marina-Socioling.git
cd Marina-Socioling
