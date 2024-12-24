part of korseon.core;

Future<void> processMessage(String data) async {
  final Map<String, Object?> msg = jsonDecode(data);
  switch (msg['source']) {
    case 'controller':
      await processStateUpdate(msg);
      break;
    case 'object-detection':
      await processObjectDetectionResults(msg);
      break;
    case 'road-segmentation':
      await processRoadSegmentationResults(msg);
      break;
    default:
      print('Ignored message (no source specified)');
      break;
  }
}

Future<void> processStateUpdate(Map<String, Object?> data) async {
  final mk.ReplayBuffer<Map<String, Object?>> payloadBuffer =
      project.createReplayBuffer(
    'newDataPayload',
    enabled: true,
    call: () => data,
  );
  final Map<String, Object?> payload = await payloadBuffer.get();
  if (payload['msgType'] == 'snapshot') {
    pointsStreamController.add([
      for (final point in (payload['lidarFront'] as List).cast<Map>())
        Vector3(
          point['x'] as double,
          point['y'] as double,
          point['z'] as double,
        )
    ]);
    final res = base64.decode(payload['cameraFront'] as String);
    imageByteStreamController.add(res);
  }
  lcbClient.socket!.sink.add(jsonEncode(
      {'dest': 'object-detection', 'image': payload['cameraFront']}));
  lcbClient.socket!.sink.add(jsonEncode(
      {'dest': 'road-segmentation', 'image': payload['cameraFront']}));
}

Future<void> processObjectDetectionResults(Map<String, Object?> data) async {
  final mk.ReplayBuffer<Map<String, Object?>> payloadBuffer =
      project.createReplayBuffer(
    'objDetResult',
    enabled: true,
    call: () => data,
  );
  final Map<String, Object?> payload = await payloadBuffer.get();

  if (payload['code'] == 'result') {
    final res = base64.decode(payload['image'] as String);
    objDetImageStreamController.add(res);
  }
}

Future<void> processRoadSegmentationResults(Map<String, Object?> data) async {
  final mk.ReplayBuffer<Map<String, Object?>> payloadBuffer =
      project.createReplayBuffer(
    'roadSegResult',
    enabled: true,
    call: () => data,
  );
  final Map<String, Object?> payload = await payloadBuffer.get();

  if (payload['code'] == 'result') {
    final res = base64.decode(payload['image'] as String);
    roadSegImageStreamController.add(res);
  }
}
