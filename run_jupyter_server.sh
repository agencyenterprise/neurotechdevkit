#!/bin/sh
nohup /ndk/venv/bin/jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --notebook-dir=/ndk/notebooks > /dev/null 2>&1 &
sleep 5
echo "Connect to: " & /ndk/venv/bin/jupyter notebook list --jsonlist | jq -r '.[0]["url"] +  "tree?token="  + .[0]["token"]'
tail -f /dev/null