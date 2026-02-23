FROM python:3.11-slim

# Instala dependências do sistema + R
RUN apt-get update && apt-get install -y \
    r-base \
    curl \
    cmake \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala pacotes R necessários pelo Rbrul
RUN Rscript -e "install.packages(c('lme4','boot','Hmisc','lattice'), repos='https://cran.r-project.org', quiet=TRUE)"

# Baixa e salva o script Rbrul localmente (evita dependência de rede em runtime)
RUN curl -o /usr/local/lib/Rbrul.R http://www.danielezrajohnson.com/Rbrul.R

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/dados

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
