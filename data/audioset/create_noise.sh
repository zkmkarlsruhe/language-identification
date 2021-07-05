
AUDIOSET_CSV="balanced.csv"
YTDL_DIR=$PWD/"yt-noise"
NOISE_DIR=$PWD/"__noise"
CV_DIR="$PWD/data/cv"

python3 data/audioset/download_youtube_noise.py --input_file $AUDIOSET_CSV --output_dir $YTDL_DIR

python3 data/other/cut_audio.py --input_dir $YTDL_DIR --output_dir $NOISE_DIR

python3 data/other/split_to_common_voice --input_dir $NOISE_DIR --output_dir $CV_DIR

rm -r $NOISE_DIR