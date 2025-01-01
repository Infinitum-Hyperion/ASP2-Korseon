part of korseon.core;

Future<void> processMessage(String data) async {
  await AutocloudMethod<void>(
    label: 'processMessage',
    contextProvider: tasksArtifact.contextProvider,
    method: (log) async {
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
    },
  ).run();
}

Future<void> processStateUpdate(Map<String, Object?> data) async {
  if (data['msgType'] == 'snapshot') {
    final mk.ReplayBuffer<Map<String, Object?>> snapshotBuffer =
        project.createReplayBuffer(
      'snapshotBuffer',
      enabled: true,
      call: () => data,
    );
    final payload = await snapshotBuffer.get();
    pointsStreamController.add([
      for (final point in (payload['lidarFront'] as List).cast<Map>())
        Vector3(
          point['x'] as double,
          point['y'] as double,
          point['z'] as double,
        )
    ]);
    final res = base64.decode(payload['cameraTop'] as String);
    imageByteStreamController.add(res);
/*     lcbClient.send(
        destination: objectDetectionArtifact,
        payload: jsonEncode({'image': payload['cameraTop']}));
    lcbClient.send(
        destination: roadSegmentationArtifact,
        payload: jsonEncode({'image': payload['cameraTop']})); */
    await processObjectDetectionResults({});
    await processRoadSegmentationResults({});
  } else if (data['msgType'] == 'nav-update') {
    final mk.ReplayBuffer<Map<String, Object?>> navUpdateBuffer =
        project.createReplayBuffer(
      'navUpdateBuffer',
      enabled: true,
      call: () => data,
    );
    final payload = await navUpdateBuffer.get();
    await performRoadLevelLocalisation(payload);
    await performLaneLevelLocalisation(payload);
  }
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

Future<void> performRoadLevelLocalisation(Map<String, Object?> data) async {
  final List<double> gpsVect = (data['gps'] as List).cast<double>();
  mapDataStreamController.add(
    MapRenderData(
      gpsPoint: LatLng(gpsVect[0], gpsVect[1]),
      localisedSegment:
          null, /**RoadLevelLocalisation().localisedPoint(
        (gpsVect[0], gpsVect[1]),
      ), */
    ),
  );
}

Future<void> performLaneLevelLocalisation(Map<String, Object?> data) async {
  final mk.ReplayBuffer<Map<String, Object?>> payloadBuffer =
      project.createReplayBuffer(
    'laneSegResult',
    enabled: true,
    call: () => data,
  );
  final Map<String, Object?> payload = await payloadBuffer.get();

  if (payload['code'] == 'result') {
    final res = base64.decode(payload['image'] as String);
    laneSegImageStreamController.add(res);
  }
}

Future<void> loadRoadSegments() async {
  const List<String> roadTypeNames = [
    'motorway',
    'trunk',
    'primary',
    'secondary',
    'tertiary',
    'unclassified',
    'residential',
    'motorway_link',
    'trunk_link',
    'primary_link',
    'secondary_link',
    'tertiary_link'
  ];

  // Returns a (way ID, [startNode, endNode]) if the [node] is a road segment
  (String, Iterable<String>)? isNodeRoadType(XmlNode node) {
    for (final XmlNode n
        in node.children.where((XmlNode nd) => nd.nodeType.name == 'tag')) {
      for (final XmlAttribute attr in n.attributes) {
        if (attr.name.local == 'k' && attr.value == 'highway') {
          if (roadTypeNames.contains(n.attributes[1].value)) {
            final ndChildren =
                n.children.where((nd) => nd.nodeType.name == 'nd');
            return (
              n.attributes.first.value,
              [
                ndChildren.first.attributes.first.value,
                ndChildren.last.attributes.first.value,
              ]
            );
          }
        }
      }
    }
    return null;
  }

  GpsPoint getLocation(XmlNode node) {
    final double lat = double.parse(
        node.attributes.firstWhere((a) => a.name.local == 'lat').value);
    final double lon = double.parse(
        node.attributes.firstWhere((a) => a.name.local == 'lon').value);

    return (lat, lon);
  }

  final XmlDocument doc =
      XmlDocument.parse(await rootBundle.loadString('files/mapA5_greater.osm'));
  final Iterable<XmlNode> ways =
      doc.children.where((node) => node.nodeType.name == 'way');
  for (final way in ways) {
    final (String, Iterable<String>)? segInfo = isNodeRoadType(way);
    if (segInfo != null) {
      print('seg: ${segInfo.$1}');
      final XmlNode startNode = doc.children.firstWhere((n) =>
          n.nodeType.name == 'node' &&
          n.attributes.first.value == segInfo.$2.first);
      final XmlNode endNode = doc.children.firstWhere((n) =>
          n.nodeType.name == 'node' &&
          n.attributes.first.value == segInfo.$2.last);
      RoadLevelLocalisation.segmentIds.add(segInfo.$1);
      RoadLevelLocalisation.roadSegments.add(
        RoadSegment(
          getLocation(startNode),
          getLocation(endNode),
        ),
      );
    }
  }
}
