#! /bin/bash

### Fill the following line with the machine-specific links

# enlish corpus v6.1
en=""
# spanish corpus v6.1
es=""
# german corpus v6.1
de=""
# french corpus v6.1
fr=""

# small set for testing
vot=""

#  fill this array with the links that you want to download
downloads=(en de es fr)

# stop on error
set -e

# change to script dir
cd $(dirname "$0")

for src in "${downloads[@]}"
do
  echo 'Starting download for' $src
  ./download_and_extract.sh $src ${!src} &
done
