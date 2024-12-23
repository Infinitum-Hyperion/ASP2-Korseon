part of korseon.core;

late final StreamController<List<Vector3>> pointsStreamController =
    StreamController<List<Vector3>>();
late final StreamController<Uint8List> imageByteStreamController =
    StreamController<Uint8List>();
late final StreamController<Uint8List> objDetImageStreamController =
    StreamController<Uint8List>();

final AutocloudProject project = AutocloudProject(
  keyValueDBProvider: firestoreProvider,
  blobDBProvider: firebaseStorageProvider,
  markhorConfigs: MarkhorConfigs(
    liveTelemetryViewModes: {
      'cam_inputs_1': (BuildContext context) {
        return SingleChildScrollView(
          child: Column(
            children: [
              Row(
                children: [
                  PointCloudRendererPane(
                    width: 500,
                    height: 500,
                    pointsStream: pointsStreamController.stream,
                  ),
                  ImageRendererPane(
                    width: 500,
                    height: 600,
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
              ImageRendererPane(
                width: 500,
                height: 600,
                imageByteStream: objDetImageStreamController.stream.map(
                  (byteList) => Image.memory(
                    byteList,
                    fit: BoxFit.fitWidth,
                  ),
                ),
              )
            ],
          ),
        );
      },
    },
  ),
);
