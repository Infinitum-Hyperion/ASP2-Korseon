#!/bin/bash

source ./configs.sh

python3 $SUMO_HOME/tools/randomTrips.py -n $WORLD_DEST/${WORLD_NAME}_net/sumo.net.xml -o $WORLD_DEST/${WORLD_NAME}_net/sumo.trip.xml

$SUMO_HOME/bin/duarouter --trip-files $WORLD_DEST/${WORLD_NAME}_net/sumo.trip.xml --net-file $WORLD_DEST/${WORLD_NAME}_net/sumo.net.xml --output-file $WORLD_DEST/${WORLD_NAME}_net/sumo.rou.xml --ignore-errors true
