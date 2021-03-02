# Dataset creation
## Mozilla Common Voice
- download the speech sets that you want to identify
- extract them into a folder i.e. "~/mozilla-data"
- in "mozilla_to_wav.py" modify the list "languages" accordingly
- run the following line:
```
$ python mozilla_to_wav.py --input_dir ~/mozilla-data 
        --output_dir ~/mozilla-processed --number 30000
``` 
#### Split dataset into train validation and test
```
$ python create_train_val_test.py --source ~/mozilla-processed
    --target ~/datset_split
```
## Download noise
- Visit: https://research.google.com/audioset/download.html
- Download the unbalanced_train_segments.csv
- In "download_youtube_noise.py" change the list of valid and non-valid labels
- Download and process youtube videos:
```
$ python download_youtube_noise.py --output_dir ~/youtube-noise \
         --input_file PATH_TO/unbalanced_train_segments.csv 
```


## Working on wav-files
We have created a CLI tool which lets you easily process wav-files.
Please check out the help by running the following command:
```
$ python process_wav_files.py --help
```
Below you will find a few useful scenarios.
#### Convert wav-files to audio features and save images to directory
```
$ python process_wav_files.py --source ~/val_split --target_length_s 10
    --img_dir ~/validation_set --feature_type spectrogram --feature_nu 128
```
#### Augment wav-files and save wavs to directory
```
$ python process_wav_files.py --source ~/train_split --target_length_s 10
    --augment --augment_nu 4 
    --wav_dir ./augmented
```
#### Augment wav-files, compute audio features and save images to directory
```
$ python process_wav_files.py --source ~/train_split --target_length_s 10
    --augment --augment_nu 4
    --img_dir ~/train_set --feature_type logfbank --feature_nu 40
```

## Other Utilities
#### Create a small version of a dataset
```
$ python create_small_set.py --source ~/mozilla-processed/pro
    --target ~/small_dataset --size 100
```