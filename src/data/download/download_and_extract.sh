#! /bin/sh

# stop on error
set -e

tarname=temp_$1.tar
tempdir=temp_$1

# download the tar and name it temp.tar
curl -o $tarname -LO $2

# untar the tar to temp_dataset_name
mkdir -p $tempdir
tar -xf $tarname -C $tempdir

# move the content to destination and clean up
mv $tempdir/*/* .
rm -r $tempdir
rm $tarname
