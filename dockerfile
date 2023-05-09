FROM python:latest

COPY  main.py .

WORKDIR /app

RUN pip install scikit-learn

RUN pip install pandas

CMD ["python", "./main.py"]