FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY . .

CMD ["python", "ConV2D.py"]

# Move output files to host directory after program execution
#CMD ["cp", "audioclassification1.h5", "/output/model_training/"]
#CMD ["cp", "audioclassification1.tflite", "/output/model_training/"]