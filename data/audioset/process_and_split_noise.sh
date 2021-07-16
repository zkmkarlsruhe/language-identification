YTDL_DIR="/data/noise"
TEMP_DIR=$PWD/"yt-noise-processed"
CV_DIR="/data/common_voice_filtered/five_sec_vad/wav/"

echo "Processing downloaded samples..."
python3 data/other/cut_audio.py --input_dir $YTDL_DIR --output_dir $TEMP_DIR

echo "Splitting and moving the processed samples to the dataset directory..."
python3 data/other/split_to_common_voice.py --input_dir $TEMP_DIR --output_dir $CV_DIR

rm -r $TEMP_DIR
