#!/bin/bash 

## use this script to download checkpoints from lab server. 
## remember to use cisco vpn to connect to the lab server first.

# Copies the contents of remote folder directly into local folder
rsync -avz --progress qiming@10.245.80.184:/storage/qiming/pi05_checkpoints/ /home/showlab/Users/qiming/openpi/checkpoints/

# Copies the folder itself inside local folder (creates /checkpoints/pi05_checkpoints/)
# rsync -avz --progress qiming@10.245.80.184:/storage/qiming/pi05_checkpoints /home/showlab/Users/qiming/openpi/checkpoints/
