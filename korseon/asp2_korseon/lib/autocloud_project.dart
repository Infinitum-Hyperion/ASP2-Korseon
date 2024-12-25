import 'package:autocloud_sdk/autocloud_sdk.dart';

final AutocloudProject project = AutocloudProject(
  label: 'asp2-korseon',
);
bool projectHasInit = false;
late final AutocloudArtifact mainArtifact;
late final AutocloudArtifact tasksArtifact;
late final AutocloudArtifact vehicleControllerArtifact;
late final AutocloudArtifact objectDetectionArtifact;
late final AutocloudArtifact roadSegmentationArtifact;
void initProject() {
  if (projectHasInit) return;
  mainArtifact = project.addArtifact(
    label: 'korseon-main',
    socket: const ArtifactSocket(host: '0.0.0.0', port: 8081),
  );

  tasksArtifact = project.addArtifact(label: 'tasks');

  vehicleControllerArtifact = project.addArtifact(
    label: 'vehicle-controller',
    socket: const ArtifactSocket(host: '0.0.0.0', port: 8080),
  );

  objectDetectionArtifact = project.addArtifact(
    label: 'object-detection',
    socket: const ArtifactSocket(host: '0.0.0.0', port: 8079),
  );

  roadSegmentationArtifact = project.addArtifact(
    label: 'road-segmentation',
    socket: const ArtifactSocket(host: '0.0.0.0', port: 8078),
  );

  mainArtifact;
  tasksArtifact;
  vehicleControllerArtifact;
  objectDetectionArtifact;
  roadSegmentationArtifact;

  projectHasInit = true;
}
