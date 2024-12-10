library korseon.local;

import 'dart:convert';
import 'dart:io';

import 'package:markhor_sdk/standalone/lightweight_communication_bridge_server.dart';

part './secrets.dart';

void main(List<String> args) async {
  final Process autocloudInterface = await Process.start(
    '/usr/local/bin/flutter/bin/flutter',
    ['run', '-d', 'chrome'],
    workingDirectory: asp2KorseonPath,
    mode: ProcessStartMode.inheritStdio,
  );

  final lcb = LightweightCommunicationBridgeForwarder()..initServers();
  await autocloudInterface.exitCode;
  stdin.transform(utf8.decoder).listen((key) async {
    if (key.trim() == 'exit') {
      await lcb.closeServers();
      exit(0);
    }
  });
}
