#!/bin/bash

echo "Stopping node..."
/home/ubuntu/Workspace/hadoop-1.1.0/bin/stop-all.sh

echo "Starting node back up..."
/home/ubuntu/Workspace/hadoop-1.1.0/bin/start-all.sh

echo "Exiting safe mode..."
hadoop dfsadmin -safemode leave

echo "Done."
