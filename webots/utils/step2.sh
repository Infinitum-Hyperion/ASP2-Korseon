#!/bin/bash

source ./configs.sh

cd /Applications/Webots.app/Contents/Resources/sumo_exporter
python3 exporter.py --input=$WORLD_DEST/$WORLD_NAME.wbt --output=$WORLD_DEST/${WORLD_NAME}_net

$SUMO_HOME/bin/netconvert --node-files=$WORLD_DEST/${WORLD_NAME}_net/sumo.nod.xml --edge-files=$WORLD_DEST/${WORLD_NAME}_net/sumo.edg.xml --output-file=$WORLD_DEST/${WORLD_NAME}_net/sumo.net.xml

echo "World configurations added and sumo.net.xml created."
echo "Run:"
echo ">> $SUMO_HOME/bin/netedit $WORLD_DEST/${WORLD_NAME}_net/sumo.net.xml"
echo "to manually inspect networks."