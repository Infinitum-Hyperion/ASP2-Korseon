library korseon.core;

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:autocloud_ui/design_system/design_system.dart';

import 'package:autocloud_sdk/autocloud_sdk.dart';
import 'package:autocloud_ui/main.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:markhor_sdk/markhor_sdk.dart' as mk;
import 'package:markhor_ui/panes/panes.dart';
import 'package:vector_math/vector_math_64.dart';

part './kv_db_provider.dart';
part './blob_db_provider.dart';
part './firestore_configs.dart';

late final StreamController<List<Vector3>> pointsStreamController =
    StreamController<List<Vector3>>();
late final StreamController<Uint8List> imageByteStreamController =
    StreamController<Uint8List>();
final firestoreProvider = FirestoreDBProvider();
final firebaseStorageProvider = FirebaseStorageDBProvider();
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
              )
            ],
          ),
        );
      },
    },
  ),
);

Future<void> onNewData(String data) async {
  final mk.ReplayBuffer<Map<String, Object?>> payloadBuffer =
      project.createReplayBuffer(
    'newDataPayload',
    enabled: true,
    call: () => jsonDecode(data),
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
}

void main(List<String> args) async {
  GlobalState.autocloudProject = project;

  await Firebase.initializeApp(options: webOptions);
  await firestoreProvider.init();
  await firebaseStorageProvider.init();

  final mk.LCBClient lcbClient = mk.LCBClient()
    ..initClient(onMessage: (str) async => await onNewData(str));
  Future.delayed(const Duration(seconds: 5), () => onNewData(''));
  runApp(const AutocloudInterface());
}
