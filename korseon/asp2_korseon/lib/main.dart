library korseon.core;

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:autocloud_ui/design_system/design_system.dart';
import 'package:autocloud_ui/main.dart';
import 'package:markhor_ui/panes/panes.dart';

import 'package:autocloud_sdk/autocloud_sdk.dart';
import 'package:markhor_sdk/markhor_sdk.dart' as mk;
import 'package:vector_math/vector_math_64.dart' hide Colors;
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_storage/firebase_storage.dart';

import 'autocloud_project.dart';
import 'package:asp2_korseon/models/models.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:web/web.dart' as web;
import 'package:xml/xml.dart';

part './kv_db_provider.dart';
part './blob_db_provider.dart';
part './firestore_configs.dart';
part './tasks.dart';
part 'autocloud_markhor_configs.dart';

part './custom_views/map_renderer.dart';

final firestoreProvider = FirestoreDBProvider();
final firebaseStorageProvider = FirebaseStorageDBProvider();
late final mk.LCBClient lcbClient;
void main(List<String> args) async {
  // Set the autocloud project
  GlobalState.autocloudProject = project;
  initProject();
  project.markhorConfigs = markhorConfigs;

  // Initialise providers
  await Firebase.initializeApp(options: webOptions);
  await firestoreProvider.init();
  await firebaseStorageProvider.init();

  // Load road segment data for GPS
  await loadRoadSegments();

  // Initialise listeners for the various services
  lcbClient = mainArtifact.socket!
      .spawnClient((str) async => await processMessage(str));

  // For debug mode
  Future.delayed(const Duration(seconds: 5),
      () => processStateUpdate({'msgType': 'snapshot'}));

  // Start the autocloud UI
  runApp(const AutocloudInterface());
}
