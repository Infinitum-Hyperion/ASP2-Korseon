FROM python:3.9-slim-buster

WORKDIR /src
COPY ./modules ./modules
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt
RUN pip install --no-cache-dir numpy pillow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
EXPOSE 8080

COPY fast_scnn_citys.pth .
COPY ./data_loader/cityscapes.py ./data_loader/
COPY ./data_loader/__init__.py ./data_loader/__init__.py
COPY visualise.py .
COPY model.py .
COPY main.py .
CMD [ "python", "main.py" ]