FROM python:3.7-slim

ENV PYTHONUNBUFFERED 1

RUN apt update && rm -rf /var/cache/apk/*

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir

COPY . .

CMD python bot.py
