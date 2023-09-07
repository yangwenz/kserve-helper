FROM python:3.9-alpine
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
COPY ./model.py .
COPY ./dog.jpg .
COPY ./requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python3", "/app/model.py"]