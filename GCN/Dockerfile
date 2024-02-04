FROM pytorch/pytorch:latest

WORKDIR /app

COPY . .

RUN pip install torch-geometric torch torchvision

CMD ["python", "main.py"]