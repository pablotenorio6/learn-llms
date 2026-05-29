# Side-car que expone métricas de nvidia-smi en formato Prometheus.
#
# Usamos una base CUDA para garantizar que nvidia-smi esté disponible cuando
# el contenedor monta la GPU via NVIDIA Container Toolkit. La imagen "base"
# pesa ~150 MB, suficiente para nuestro caso (no necesitamos el toolkit
# completo de CUDA).
FROM nvidia/cuda:12.3.0-base-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir prometheus-client==0.21.1

WORKDIR /app
COPY exporter.py /app/exporter.py

EXPOSE 9835

CMD ["python3", "/app/exporter.py"]
