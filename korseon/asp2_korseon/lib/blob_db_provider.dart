part of korseon.core;

class FirebaseStorageDBProvider<FirebaseStorageReferenceProvider>
    extends BlobDBProvider {
  late final Reference storage;

  @override
  Future<void> init() {
    storage = (FirebaseStorage.instance
          ..useStorageEmulator(
            '127.0.0.1',
            8950,
          ))
        .ref();
    return super.init();
  }

  @override
  Future<Uint8List?> get(String ref) async {
    return await storage.child(ref).getData();
  }

  @override
  Future<void> put(String ref, Uint8List data) async {
    await storage.child(ref).putData(data);
  }

  @override
  Future<void> delete(String ref) async {
    await storage.child(ref).delete();
  }
}
