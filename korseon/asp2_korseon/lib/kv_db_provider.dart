part of korseon.core;

class FirestoreDBProvider extends KeyValueDBProvider {
  late final FirebaseFirestore firestore;
  @override
  Future<void> init() async {
    WidgetsFlutterBinding.ensureInitialized();
    firestore = FirebaseFirestore.instance
      ..useFirestoreEmulator(
        '127.0.0.1',
        8900,
      );
    await super.init();
  }

  @override
  Future<T?> getById<T>(String id, [String? key]) async {
    final snapshot = await firestore.doc(id).get();
    if (snapshot.exists) {
      if (key != null) {
        final data = snapshot.data()!;
        if (data.containsKey(key)) {
          return data[key];
        } else {
          return null;
        }
      } else {
        return snapshot.data()! as T;
      }
    } else {
      return null;
    }
  }

  @override
  Future<void> insertById(String id, Map<String, Object?> value) async {
    await firestore.doc(id).set(value);
  }

  @override
  Future<void> updateById<T>(String id, String key, T value) async {
    await firestore.doc(id).update({key: value});
  }
}
