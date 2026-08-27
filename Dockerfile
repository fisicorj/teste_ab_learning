FROM python:3.11-slim

WORKDIR /app

# Instala as dependências primeiro (aproveita cache do Docker em rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do código
COPY . .

# O Render (e a maioria dos PaaS) injeta a porta real na env var PORT.
# 5000 é só o valor padrão para rodar localmente com `docker run`.
ENV PORT=5000
EXPOSE 5000

# Usa gunicorn (servidor de produção) em vez do servidor de desenvolvimento do Flask.
# Forma "shell" do CMD para o ${PORT} ser expandido em tempo de execução.
CMD gunicorn experiment_app:app --bind 0.0.0.0:${PORT}
