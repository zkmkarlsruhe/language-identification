YTDL_DIR=$1
TEMP_DIR=$PWD/"yt-noise-processed"
CV_DIR=$2

echo "Processing downloaded samples..."
python3 data/other/cut_audio.py --input_dir $YTDL_DIR --output_dir $TEMP_DIR

echo "Splitting and moving the processed samples to the dataset directory..."
python3 data/other/split_to_common_voice.py --input_dir $TEMP_DIR --output_dir $CV_DIR

rm -r $TEMP_DIR
