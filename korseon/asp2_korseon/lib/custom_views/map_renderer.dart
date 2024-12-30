part of korseon.core;

class MapRenderData {
  final LatLng gpsPoint;
  final RoadSegment? localisedSegment;

  const MapRenderData({
    required this.gpsPoint,
    required this.localisedSegment,
  });
}

/// The [dataStream] must be a broadcast [Stream]
class MapRendererPane extends StatefulWidget {
  final double width;
  final double height;
  final Stream<MapRenderData> dataStream;

  const MapRendererPane({
    required this.width,
    required this.height,
    required this.dataStream,
    super.key,
  });
  @override
  State<StatefulWidget> createState() => MapRendererPaneState();
}

class MapRendererPaneState extends State<MapRendererPane> with PaneStyling {
  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return paneStyle(
      title: 'Road Localisation',
      primaryColor: ACPColor.purple,
      width: widget.width,
      child: StreamBuilder(
        stream: widget.dataStream,
        builder: (ctx, snapshot) => snapshot.hasData
            ? SizedBox(
                width: widget.width,
                height: widget.height,
                // Because the `FlutterMap` widget prints useless messages in the console
                child: runZoned(
                  zoneSpecification:
                      ZoneSpecification(print: (self, parent, zone, line) {}),
                  () => FlutterMap(
                    options: MapOptions(
                      initialCenter: snapshot.requireData.gpsPoint,
                      initialZoom: 17,
                    ),
                    children: [
                      TileLayer(
                        urlTemplate:
                            'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                      ),
                      SimpleAttributionWidget(
                        source: const Text('Data from OpenStreetMap'),
                        onTap: () => web.window
                            .open('https://openstreetmap.org/copyright'),
                      ),
                      PolylineLayer(
                        polylines: [
                          if (snapshot.requireData.localisedSegment != null)
                            Polyline(
                              points: [
                                LatLng(
                                  snapshot
                                      .requireData.localisedSegment!.start.$1,
                                  snapshot
                                      .requireData.localisedSegment!.start.$2,
                                ),
                                LatLng(
                                  snapshot.requireData.localisedSegment!.end.$1,
                                  snapshot.requireData.localisedSegment!.end.$2,
                                ),
                              ],
                              color: Colors.blue,
                            ),
                        ],
                      ),
                      MarkerLayer(
                        markers: [
                          Marker(
                            point: snapshot.requireData.gpsPoint,
                            width: 6,
                            height: 6,
                            child: Container(
                              width: 6,
                              height: 6,
                              color: Colors.red,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              )
            : const SizedBox(
                width: 40,
                height: 40,
                child: CircularProgressIndicator(),
              ),
      ),
    );
  }
}
