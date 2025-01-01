FROM python:3.9-slim-buster

WORKDIR /src

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pillow numpy opencv-python
COPY . .

CMD ["/bin/sh", "-c", "bash"]