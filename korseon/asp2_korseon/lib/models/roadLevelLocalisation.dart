part of korseon.core.models;

typedef GpsPoint = (double, double);

class RoadSegment {
  final GpsPoint start;
  final GpsPoint end;

  const RoadSegment(this.start, this.end);

  @override
  bool operator ==(Object other) =>
      other is RoadSegment && other.start == start && other.end == end;

  @override
  String toString() => "GpsPoint($start, $end)";

  @override
  int get hashCode => toString().hashCode;
}

class Candidate {
  final int timestep;
  final GpsPoint point;
  final RoadSegment roadSegment;

  const Candidate(this.timestep, this.point, this.roadSegment);

  @override
  bool operator ==(Object other) =>
      other is Candidate &&
      other.point == point &&
      other.timestep == timestep &&
      other.roadSegment == roadSegment;

  @override
  String toString() => "Candidate($timestep, $point, $roadSegment)";

  @override
  int get hashCode => toString().hashCode;
}

class RoadLevelLocalisation {
  final double gpsStandardDeviation;
  final double inclusionRadius;
  final double adjacencyExclusionRadius;
  final double earlySearchTermination;
  final double speedLimit;

  static final List<GpsPoint> measurements = [];
  static final List<String> segmentIds = [];
  static final List<RoadSegment> roadSegments = [];

  static final List<int> localisedRoads = [];
  static const double earthRad = 6371000;

  /// A 2D array, where `candidates[t][i]` gives the candidate match at
  /// timestep `t` and with road segment at index `i` of `roadSegments`.
  static final List<List<Candidate>> candidates = [];

  RoadLevelLocalisation({
    this.gpsStandardDeviation = 4,
    this.inclusionRadius = 80,
    this.adjacencyExclusionRadius = 0,
    this.earlySearchTermination = 500,
    this.speedLimit = 70,
  });

  RoadSegment localisedPoint(GpsPoint point) {
    final int timestep = measurements.length;
    // Skip for very close measurements
    print('skipping');
    if (measurements.isNotEmpty &&
        distanceGreatCircle(point, measurements.last) <
            adjacencyExclusionRadius) return roadSegments[localisedRoads.last];
    measurements.add(point);
    final List<Candidate> newCandidates = [];
    for (int r = 0; r < roadSegments.length; r++) {
      // Convert coordinates to cartesian
      final (double, double, double) cartPoint = gpsToCartesian(point);
      final (double, double, double) startSeg =
          gpsToCartesian(roadSegments[r].start);
      final (double, double, double) endSeg =
          gpsToCartesian(roadSegments[r].end);
      // Vector representation
      final Vector3 roadSegVect = Vector3(endSeg.$1 - startSeg.$1,
          endSeg.$2 - startSeg.$2, endSeg.$3 - startSeg.$3);
      final Vector3 gpsVect = Vector3(cartPoint.$1 - startSeg.$1,
          cartPoint.$2 - startSeg.$2, cartPoint.$3 - startSeg.$3);
      // Vector projection
      final double projFactor =
          dot3(gpsVect, roadSegVect) / dot3(roadSegVect, roadSegVect);
      // Compute projected point
      final (double, double, double) projectedCoord = (
        startSeg.$1 + projFactor * (endSeg.$1 - startSeg.$1),
        startSeg.$2 + projFactor * (endSeg.$2 - startSeg.$2),
        startSeg.$3 + projFactor * (endSeg.$3 - startSeg.$3)
      );
      newCandidates.add(
          Candidate(timestep, cartesianToGps(projectedCoord), roadSegments[r]));
    }
    candidates.add(newCandidates);

    print('running algo');
    final RoadSegment res = roadSegments[viterbisAlgorithm().last];
    print("RES:${res.start}");
    return res;
  }

  double emissionProbability(int timestep, int roadSegmentIndex) {
    final double greatCircleDist = distanceGreatCircle(
        measurements[timestep], candidates[timestep][roadSegmentIndex].point);
    return greatCircleDist > earlySearchTermination
        ? 0
        : (1 / (sqrt(2) * pi * gpsStandardDeviation)) *
            (pow(
              e,
              -0.5 * (greatCircleDist / gpsStandardDeviation),
            ));
  }

  double transitionProbability(Candidate a, Candidate b, int timestep) {
    final double scalingParam = scalingParamEstimator1();
    final double diff = (distanceGreatCircle(
              measurements[timestep],
              measurements[timestep + 1],
            ) -
            distanceRoute(a, b))
        .abs();
    return (1 / scalingParam) *
        pow(
          e,
          (-diff / scalingParam),
        );
  }

  List<int> viterbisAlgorithm() {
    final List<int> sequence = [];
    final List<List<int>> backpointer = [];
    // Accessed as table[t][state]
    final List<List<double>> viterbiTable = [];

    print('initialising');
    // Initialise the table
    for (int t = 0; t < measurements.length; t++) {
      viterbiTable.add([]);
      for (int i = 0; i < roadSegments.length; i++) {
        viterbiTable[t].add(emissionProbability(t, i));
        backpointer[t].add(0);
      }
    }

    int indexOfMaxItem(List<double> list) {
      int result = 0;
      for (int i = 0; i < list.length; i++) {
        if (list[i] > list[result]) {
          result = i;
        }
      }
      return result;
    }

    print('recursing');
    // Recurse
    for (int t = 1; t < measurements.length; t++) {
      for (int j = 0; j < roadSegments.length; j++) {
        double maxProb = double.negativeInfinity;
        int prevState = 0;

        for (int i = 0; i < roadSegments.length; i++) {
          double prob = viterbiTable[t - 1][i] *
              transitionProbability(
                candidates[t - 1][i],
                candidates[t][j],
                t - 1,
              ) *
              emissionProbability(t, j);

          if (prob > maxProb) {
            maxProb = prob;
            prevState = i;
          }
        }

        viterbiTable[t][j] = maxProb;
        backpointer[t][j] = prevState;
      }
    }

    print('backtracking');
    // Backtracking step
    int lastState = indexOfMaxItem(viterbiTable.last);
    sequence.add(lastState);

    for (int t = measurements.length - 1; t > 0; t--) {
      lastState = backpointer[t][lastState];
      sequence.insert(0, lastState);
    }
    print(sequence);
    return sequence;
  }

  double scalingParamEstimator1() => 5;

  (double, double, double) gpsToCartesian(GpsPoint point) => (
        earthRad * cos(point.$1) * cos(point.$2),
        earthRad * cos(point.$1) * sin(point.$2),
        earthRad * sin(point.$1)
      );

  (double, double) cartesianToGps((double, double, double) point) =>
      ((atan2(point.$2, point.$1) * 180 / pi), asin(point.$3 / earthRad));

  double distanceGreatCircle(GpsPoint a, GpsPoint b) {
    return earthRad *
        acos(
          cos(a.$1) * cos(b.$1) * cos(a.$2 - b.$2) + sin(a.$1) + sin(b.$1),
        );
  }

  /// Calculates the route-distance using A* algo
  double distanceRoute(Candidate a, Candidate b) {
    final List<RoadSegment> open = [a.roadSegment];
    final List<RoadSegment> closed = [];
    final Map<RoadSegment, double> pathCost = {a.roadSegment: 0};

    double heuristicCostEstimate(RoadSegment current) =>
        distanceGreatCircle(current.start, b.roadSegment.end);

    double totalCost(RoadSegment seg) =>
        pathCost[seg]! + heuristicCostEstimate(seg);

    while (open.isNotEmpty) {
      RoadSegment current =
          open.reduce((s1, s2) => pathCost[s1]! > pathCost[s2]! ? s2 : s1);
      if (current == b.roadSegment) {
        return pathCost[current]!;
      }

      open.remove(current);
      closed.add(current);
      for (final neighbour in roadSegments.where(
        (seg) =>
            seg.start == current.start ||
            seg.end == current.end ||
            seg.start == current.end ||
            seg.end == current.start,
      )) {
        if (closed.contains(neighbour)) continue;
        double tentativePathCost = pathCost[current]! +
            distanceGreatCircle(neighbour.start, neighbour.end);
        if (!open.contains(neighbour)) {
          open.add(neighbour);
        } else if (tentativePathCost >= pathCost[neighbour]!) {
          continue;
        }
      }
    }

    throw "Failure";
  }
}
