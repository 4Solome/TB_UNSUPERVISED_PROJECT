FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git wget

COPY app/ /app/
COPY models/ /app/models/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
