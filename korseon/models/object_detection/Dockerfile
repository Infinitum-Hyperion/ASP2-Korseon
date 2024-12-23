FROM hdgigante/python-opencv:4.10.0-debian

RUN mkdir -p /yolo
RUN wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P /yolo
RUN wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg -P /yolo
RUN wget https://github.com/AlexeyAB/darknet/blob/master/data/coco.names -P /yolo

# Add dependencies and set up venv
RUN apt install -y python3.11-venv
WORKDIR /src
COPY ./modules/ ./src/modules/
COPY ./requirements.txt ./src/
RUN python3 -m venv /src/venv --system-site-packages
ENV PATH="/src/venv/bin:$PATH"
RUN pip3 install --no-cache-dir -r ./src/requirements.txt

# Expose the Websocket port
EXPOSE 8080

# Add main script
COPY main.py .
CMD ["/bin/bash", "-c", "source /src/venv/bin/activate && python main.py"]