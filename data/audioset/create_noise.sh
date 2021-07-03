
AUDIOSET_CSV="balanced.csv"
YTDL_DIR=$PWD/"yt-noise"
YTDL_DIR=$PWD/"__noise"

python3 data/audioset/download_youtube_noise.py --input_file $AUDIOSET_CSV --output_dir $YTDL_DIR

python3 data/other/cut_audio.py --input_dir $YTDL_DIR --output_dir 