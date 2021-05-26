

# Language Identification
A playground to test and experiment with spoken languages identification. 

This code base has been developed by Hertz-Lab as part of the project [»The Intelligent Museum«](#the-intelligent-museum). 
Please raise issues, ask questions, throw in ideas or submit code as this repository is intended to be an open platform to collaboratively improve the task of spoken language identification (lid).

#### Target Platform:
- Linux (tested with Ubuntu 18.04)
- MacOS (tested with 10.15)
- Windows (not tested)



## Installation
To train the neural network we recommend to use a GPU, although some networks may be small enough to be trained on a decent CPU.

### Docker with GPU support
If you haven't installed docker with GPU support yet, please follow [these instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Build the docker image using the [Dockerfile](Dockerfile). Make sure not to include any large file in the build process (all files in the build directory are taken into account).
```shell
docker build -t lid .
```

### Ubuntu
Otherwise you can install the requirements locally. This process is similiar to the one in the Dockerfile.

See <https://www.tensorflow.org/install/gpu> for more information on GPU support for TensorFlow.

#### Dependencies
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

#### Download
Start by downloading language sets you are interested in. You can use our provided download script, but you have to generate and copy machine-sprcific download links into it, as the donwload requires your consent.

```shell
./data/download_common_voice.sh
```
Afterwards collect the data sets in a folder, referred to as `$DATA_DIR`.

Start downloading the youtube noise dataset. This process is yet not prallelized and may require a day to complete. Download the __unbalanced__ data set from the [website](https://research.google.com/audioset/download.html).
```shell
python download_youtube_noise.py --input_file unbalanced_train_segments.csv --output_dir $NOISE_DIR
```

#### Creation
We use several processing steps to form our data set from the Common Voice downloads. We recommend using the config file to define and document the processing steps. Please take a look at the CLI tool for more information on the options.
Modify the config file accordingly, e.g. replace `cv_dir` with $DATA_DIR and name the languages in the table at the bottom.
```shell
python data/mozilla_to_wav.py --config data/config_moz.yaml
```

If you don't have enough noise data then you may want to randomly scramble the noise data using the following script.
```shell
python data/augment_youtube_noise.py --source $NOISE_DIR
```

Afterwards create another folder in the train, dev and test folders which includes portions of the youtube noise.



## Training

### Docker 
Run a container of the newly build image
```shell
docker run -it --rm -v $(pwd):/work/src -v $(pwd)/../data:/data lid python train.py --config config_train.yaml
```
### local installation
```
python train.py --config config_train.yaml
```

## Todo
- test scripts
- document docker
- README
- results
- parallelize youtube download
- use a voice extractor 
- license

## Further Reading
* [Speech Features](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
* [CRNN-LID](https://github.com/HPI-DeepLearning/crnn-lid)


## License
GPLv3 see `LICENSE` for more information.


## Contribute
Contributions are very welcome!
Please send an email to bethge@zkm.de

## The Intelligent Museum
An artistic-curatorial field of experimentation for deep learning and visitor participation

The [ZKM | Center for Art and Media](https://zkm.de/en) and the [Deutsches Museum Nuremberg](https://www.deutsches-museum.de/en/nuernberg/information/) cooperate with the goal of implementing an AI-supported exhibition. Together with researchers and international artists, new AI-based works of art will be realized during the next four years (2020-2023).  They will be embedded in the AI-supported exhibition in both houses. The Project „The Intelligent Museum“ is funded by the Digital Culture Programme of the [Kulturstiftung des Bundes](https://www.kulturstiftung-des-bundes.de/en) (German Federal Cultural Foundation).

As part of the project, digital curating will be critically examined using various approaches of digital art. Experimenting with new digital aesthetics and forms of expression enables new museum experiences and thus new ways of museum communication and visitor participation. The museum is transformed to a place of experience and critical exchange.

![Logo](media/Logo_ZKM_DMN_KSB.png)
