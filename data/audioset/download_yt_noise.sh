AUDIOSET_URL="http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"
AUDIOSET_CSV=$PWD/"unbalanced.csv"
YTDL_DIR=$1

echo "Downloading AudioSet meta info... this may take a while..."
wget -O $AUDIOSET_CSV  $AUDIOSET_URL

echo "Downloading selected AudioSet samples... this may take hours..."
python3 data/audioset/download_youtube_noise.py --input_file $AUDIOSET_CSV --output_dir $YTDL_DIR
