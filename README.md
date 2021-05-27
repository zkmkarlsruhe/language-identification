# Language Identification
A playground to test and experiment with spoken language identification. 

This code base has been developed by Hertz-Lab as part of the project [»The Intelligent Museum«](#the-intelligent-museum). 
Please raise issues, ask questions, throw in ideas or submit code as this repository is intended to be an open platform to collaboratively improve the task of spoken language identification (lid).

Copyright (c) 2021 ZKM | Karlsruhe.
BSD Simplified License.

#### Target Platform:
- Linux (tested with Ubuntu 18.04)
- MacOS (tested with 10.15)
- Windows (not tested, should be easy to fix)



## Installation
To train a neural network we generally recommend to use a GPU, although the network used in this repository may be small enough to be trained on a decent CPU.

### Docker 

##### GPU support
If you haven't installed docker with GPU support yet, please follow [these instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

##### Building the image
Build the docker image using the [Dockerfile](Dockerfile). Make sure not to include any large file in the build process (all files in the build directory are taken into account).
```shell
docker build -t lid .
```

### Local Installation
Otherwise you can install the requirements locally. This process is similar to the one in the Dockerfile.

See <https://www.tensorflow.org/install/gpu> for more information on GPU support for TensorFlow.

#### System Requirements
- ffmpeg
- sox
- portaudio
- youtube-dl (version > 2020)
- python
- pip


#### Ubuntu
Feature extraction
``` shell
sudo apt install ffmpeg sox libasound-dev portaudio19-dev
``` 
youtube-dl (version > 2020)
```shell
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```
Python
``` shell
sudo apt install python3 python3-pip 
```

#### MacOs

#### Python packages
```shell
pip install -r requirements.txt
```


## Dataset
This repository extracts language examples from Mozilla's open speech corpus 
[Common Voice](https://commonvoice.mozilla.org/) and Google's audio scene dataset 
[AudioSet](https://research.google.com/audioset/dataset/index.html) 
for noise data to enhance the robustness of the model. 
However, most code should work for other data sets as well.

### Download
Start by downloading language sets you are interested in. We recommend to languages with at least 1000 speakers and 100 hours validated audio samples. Check [this site](https://commonvoice.mozilla.org/de/languages) for details.

You can use our provided download script, but you have to generate and copy machine-specific download links into it, as the download requires your consent.

```shell
./data/download_common_voice.sh
```
Afterwards collect the data sets in a folder, referred to as `$DATA_DIR`.

Start downloading the Youtube noise dataset. This process parallelized so make sure to use enough threads. Download the __unbalanced__ data set from the [website](https://research.google.com/audioset/download.html) and pass it to the script.
```shell
python download_youtube_noise.py --input_file unbalanced_train_segments.csv --output_dir $NOISE_DIR
```

### Audio Extraction
We use several processing steps to form our data set from the Common Voice downloads. We recommend using the config file to define and document the processing steps. Please take a look at the CLI arguments in the script for more information on the options.
```shell
python data/mozilla_to_wav.py --help
```
Modify the config file accordingly, e.g. replace `cv_dir` with $DATA_DIR and name the languages in the table at the bottom.
```shell
python data/mozilla_to_wav.py --config data/config_moz.yaml
```

If you don't have enough noise data then you may want to randomly scramble the noise data using the following script.
```shell
python data/augment_youtube_noise.py --source $NOISE_DIR
```

Afterwards create another folder in the train, dev and test sub folders which include portions of the Youtube noise.

### Preprocessing
In this version, we use [kapre](https://kapre.readthedocs.io/en/latest/) to extract the features (such as FFT or Mel-filterbanks) within the TensorFlow graph. This is especially useful in terms of portability, since we only need to pass the normalized audio to the model.

If you rather do the preprocessing separately and before training, you may want to utilize the script `data/process_wav.py` and its config file, as well as its dependant source files. In the future, we may create another branch which tackles the problem this way (as we used to do it before using kapre).


## Training
As with the creation of the dataset we use config files to define and document the process. The options we provide should sound familiar. Most importantly, modify the placeholder for the train and validation directories, as well as the languages to be detected (noise is treated as another language).

### Docker 
Run a container of the newly build image
```shell
docker run -it --rm -v $(pwd):/work/src -v $(pwd)/../data:/data lid python train.py --config config_train.yaml
```

### Local installation
```
python train.py --config config_train.yaml
```


## Todo
- test scripts
- results
- training visualization
- speed-up data loader with custom augmenter
- only-testing installation guide
- use a voice (instead of audio) activity detector  


## Further Reading
* [Speech Features](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
* [CRNN-LID](https://github.com/HPI-DeepLearning/crnn-lid)
* [keyword-spotting](https://github.com/douglas125/SpeechCmdRecognition)


## Contribute
Contributions are very welcome.
Create an account, clone or fork the repo, then request a push/merge.
If you find any bugs or suggestions please raise issues.


## The Intelligent Museum
An artistic-curatorial field of experimentation for deep learning and visitor participation

The [ZKM | Center for Art and Media](https://zkm.de/en) and the [Deutsches Museum Nuremberg](https://www.deutsches-museum.de/en/nuernberg/information/) cooperate with the goal of implementing an AI-supported exhibition. Together with researchers and international artists, new AI-based works of art will be realized during the next four years (2020-2023).  They will be embedded in the AI-supported exhibition in both houses. The Project „The Intelligent Museum“ is funded by the Digital Culture Programme of the [Kulturstiftung des Bundes](https://www.kulturstiftung-des-bundes.de/en) (German Federal Cultural Foundation).

As part of the project, digital curating will be critically examined using various approaches of digital art. Experimenting with new digital aesthetics and forms of expression enables new museum experiences and thus new ways of museum communication and visitor participation. The museum is transformed to a place of experience and critical exchange.

![Logo](media/Logo_ZKM_DMN_KSB.png)
