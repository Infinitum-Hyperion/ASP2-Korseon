part of korseon.core;

late final StreamController<List<Vector3>> pointsStreamController =
    StreamController<List<Vector3>>();
late final StreamController<Uint8List> imageByteStreamController =
    StreamController<Uint8List>();
late final StreamController<Uint8List> objDetImageStreamController =
    StreamController<Uint8List>();
late final StreamController<Uint8List> roadSegImageStreamController =
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
                  width: 500,
                  height: 300,
                  pointsStream: pointsStreamController.stream,
                ),
                ImageRendererPane(
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
                  width: 500,
                  height: 300,
                  imageByteStream: roadSegImageStreamController.stream.map(
                    (byteList) => Image.memory(
                      byteList,
                      fit: BoxFit.fitWidth,
                    ),
                  ),
                ),
              ],
            )
          ],
        ),
      );
    }
  },
);
