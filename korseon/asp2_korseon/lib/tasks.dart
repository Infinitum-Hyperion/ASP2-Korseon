part of korseon.core;

Future<void> processMessage(String data) async {
  AutocloudMethod(
      label: 'processMessage', contextProvider: tasksArtifact.contextProvider);

  final Map<String, Object?> msg = jsonDecode(data);
  if (msg['source'] == vehicleControllerArtifact.id) {
    await processStateUpdate(msg);
  } else if (msg['source'] == objectDetectionArtifact.id) {
    await processObjectDetectionResults(msg);
  } else if (msg['source'] == roadSegmentationArtifact.id) {
    await processRoadSegmentationResults(msg);
  } else {
    print('Ignored message (no source specified)');
    if (msg.containsKey('image')) {
      msg.remove('image');
    }
    print(msg);
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
  lcbClient.send(
      destination: objectDetectionArtifact,
      payload: jsonEncode({'image': payload['cameraFront']}));

  lcbClient.send(
      destination: roadSegmentationArtifact,
      payload: jsonEncode({'image': payload['cameraFront']}));
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
