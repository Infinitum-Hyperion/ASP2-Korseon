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
part './tasks.dart';
part './autocloud_project.dart';

final firestoreProvider = FirestoreDBProvider();
final firebaseStorageProvider = FirebaseStorageDBProvider();
late final mk.LCBClient lcbClient;
void main(List<String> args) async {
  // Set the autocloud project
  GlobalState.autocloudProject = project;

  // Initialise providers
  await Firebase.initializeApp(options: webOptions);
  await firestoreProvider.init();
  await firebaseStorageProvider.init();

  // Initialise listeners for the various services
  lcbClient = mk.LCBClient()
    ..initClient(onMessage: (str) async => await processMessage(str));

  // For debug mode
  Future.delayed(const Duration(seconds: 5), () => processStateUpdate({}));

  // Start the autocloud UI
  runApp(const AutocloudInterface());
}
