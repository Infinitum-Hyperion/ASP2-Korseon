#!/bin/bash

source ./configs.sh

cd /Applications/Webots.app/Contents/Resources/osm_importer
python3 importer.py --input=$OSM_SOURCE --output=$WORLD_DEST/$WORLD_NAME.wbt

mkdir $WORLD_DEST/${WORLD_NAME}_net

echo "WBT world generated and world net created. Open Webots to add SUMO PROTO node"