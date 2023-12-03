FROM python:3.10.10

WORKDIR /app/ai

COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 9000

CMD ["python", "main.py"]