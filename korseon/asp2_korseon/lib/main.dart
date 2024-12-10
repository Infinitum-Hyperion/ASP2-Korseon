library korseon.core;

import 'dart:async';
import 'dart:convert';

import 'package:autocloud_ui/design_system/design_system.dart';

import 'package:autocloud_sdk/autocloud_sdk.dart';
import 'package:autocloud_ui/main.dart';
import 'package:flutter/material.dart';
import 'package:markhor_sdk/markhor_sdk.dart';
import 'package:markhor_ui/panes/panes.dart';
import 'package:vector_math/vector_math_64.dart';

late final StreamController<List<Vector3>> pointsStreamController =
    StreamController<List<Vector3>>();
final AutocloudProject project = AutocloudProject(
  markhorConfigs: MarkhorConfigs(
    liveTelemetryViewModes: {
      'cam_inputs_1': (BuildContext context) {
        return SingleChildScrollView(
          child: Column(
            children: [
              Row(
                children: [
                  PointCloudRendererPane(
                    width: 800,
                    height: 500,
                    pointsStream: pointsStreamController.stream,
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

void main(List<String> args) async {
  final LCBClient lcbClient = LCBClient()
    ..initClient(onMessage: (data) {
      print('got data');
      final Map<String, Object?> payload = jsonDecode(data);
      if (payload['msgType'] == 'snapshot') {
        pointsStreamController.add([
          for (final point in payload['lidarFront'] as List<Map>)
            Vector3(
              point['x'] as double,
              point['y'] as double,
              point['z'] as double,
            )
        ]);
      }
    });
  pointsStreamController.add([Vector3(0, 0, 0)]);

  GlobalState.autocloudProject = project;
  runApp(const AutocloudInterface());
}
