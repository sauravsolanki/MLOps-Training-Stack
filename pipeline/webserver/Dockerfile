FROM python:3.9.7-slim

# Install python packages
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

WORKDIR /app

COPY [ "predict.py", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]