part of korseon.core;

late final StreamController<List<Vector3>> pointsStreamController =
    StreamController<List<Vector3>>();
late final StreamController<Uint8List> imageByteStreamController =
    StreamController<Uint8List>();
late final StreamController<Uint8List> objDetImageStreamController =
    StreamController<Uint8List>();
late final StreamController<Uint8List> roadSegImageStreamController =
    StreamController<Uint8List>();
late final StreamController<MapRenderData> mapDataStreamController =
    StreamController<MapRenderData>();
late final StreamController<Uint8List> laneSegImageStreamController =
    StreamController<Uint8List>();

final MarkhorConfigs markhorConfigs = MarkhorConfigs(
  keyValueDBProvider: firestoreProvider,
  blobDBProvider: firebaseStorageProvider,
  contextProvider: mk.ContextProvider(artifact: mainArtifact),
  liveTelemetryViewModes: {
    'cam_inputs_1': (BuildContext context) {
      return SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            Row(
              children: [
                PointCloudRendererPane(
                  title: 'Lidar Point Cloud',
                  width: 500,
                  height: 300,
                  pointsStream: pointsStreamController.stream,
                ),
                ImageRendererPane(
                  title: 'Camera Feed (C1)',
                  width: 500,
                  height: 300,
                  imageByteStream: imageByteStreamController.stream.map(
                    (byteList) => Image.memory(
                      byteList,
                      fit: BoxFit.fitWidth,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                ImageRendererPane(
                  title: 'Object Detection',
                  width: 500,
                  height: 300,
                  imageByteStream: objDetImageStreamController.stream.map(
                    (byteList) => Image.memory(
                      byteList,
                      fit: BoxFit.fitWidth,
                    ),
                  ),
                ),
                ImageRendererPane(
                  title: 'Road Segmentation',
                  width: 500,
                  height: 300,
                  imageByteStream: roadSegImageStreamController.stream.map(
                    (byteList) => Image.memory(
                      byteList,
                      fit: BoxFit.fitWidth,
                    ),
                  ),
                ),
                ImageRendererPane(
                  title: 'Lane Segmentation',
                  width: 500,
                  height: 300,
                  imageByteStream: laneSegImageStreamController.stream.map(
                    (byteList) => Image.memory(
                      byteList,
                      fit: BoxFit.fitWidth,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                MapRendererPane(
                  width: 500,
                  height: 300,
                  dataStream: mapDataStreamController.stream,
                ),
              ],
            ),
          ],
        ),
      );
    }
  },
);
