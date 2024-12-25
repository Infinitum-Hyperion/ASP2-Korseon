library korseon.local;

import 'dart:convert';
import 'dart:io';

import 'package:asp2_korseon/autocloud_project.dart';
import 'package:markhor_sdk/standalone/artifact_discovery.dart';

part './secrets.dart';

void main(List<String> args) async {
  // Launch the Autocloud interface
  final Process autocloudInterface = await Process.start(
    '/usr/local/bin/flutter/bin/flutter',
    ['run', '-d', 'chrome'],
    workingDirectory: asp2KorseonPath,
    mode: ProcessStartMode.inheritStdio,
  );

  // Launch the artifact discovery service
  initProject();
  final ArtifactDiscoveryService discoveryService =
      ArtifactDiscoveryService(project: project)..init();

  await autocloudInterface.exitCode;
  // Shut down artifact discovery service gracefully
  stdin.transform(utf8.decoder).listen((key) async {
    if (key.trim() == 'exit') {
      await discoveryService.closeServers();
      exit(0);
    }
  });
}
