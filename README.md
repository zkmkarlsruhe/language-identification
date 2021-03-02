# Language Identification System
A playground to test and experiment with spoken languages identification. 

This repository extracts language examples from Mozilla's open speech corpus 
[Common Voice](https://commonvoice.mozilla.org/).
Feel free to contribute your voice and expertise to the corpus. Furthermore, Google's audio scene dataset 
[AudioSet](https://research.google.com/audioset/dataset/index.html) 
can be used to extract noise data to enhance the robustness of the model. 
 
This code base has been developed by Hertz-Lab as part of the project [»The Intelligent Museum«](#the-intelligent-museum). 
Please raise issues, ask questions, throw in ideas or submit code as this repository is intended to be an open platform to collaboratively improve the task of spoken language identification (lid).

##### Target Platform:
* Ubuntu 18.04 Desktop
* MacOS 10.15 (Installation may differ)

##### Features
* Input may be WAV files or microphone
* Parameterizable Acoustic Activation Detection
* Enable neural network to identify language
* Disable neural network for dataset creation
* Pretrained model which detects French, Spanish, English, German and Russian
* Scripts for dataset creation and augmentation
* Possible features: MFCC, Mel-scaled filter banks, spectrogram

##### Structure
- lid_client/: source code for the lid application
- lid_network/: training process and model defenitions
- data/: a collection of scripts to download and process datasets

## Installation

### Ubuntu
Download and install [Anaconda](https://www.anaconda.com/products/individual). Afterwards create a virtual environment
```
$ conda create -n "name" python=3.7
$ conda activate "name"
$ pip3 install -r requirements.txt
```

##### Additional Software 
- ffmpeg, sox and portAudio
```
$ sudo apt install ffmpeg sox libasound-dev portaudio19-dev
```
- youtube-dl (version > 2020)
```
$ sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
$ sudo chmod a+rx /usr/local/bin/youtube-dl
```
## Usage
##### show help
```
$ python lid.py --help
```
##### microphone input
```
$ python lid.py
```
##### WAV-file input
```
$ python lid.py --file_name test/paul_deutsch.wav
```

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