FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/user-states

EXPOSE $PORT

CMD sh -c "gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120"
