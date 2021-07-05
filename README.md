# Spoken Language Identification
A playground to test and experiment with spoken language identification. 

This code base has been developed by Hertz-Lab as part of the project [»The Intelligent Museum«](#the-intelligent-museum). 
Please raise issues, ask questions, throw in ideas or submit code as this repository is intended to be an open platform to collaboratively improve the task of spoken language identification (LID).

Copyright (c) 2021 ZKM | Karlsruhe.
BSD Simplified License.

#### Target Platform:
- Linux (tested with Ubuntu 18.04)
- MacOS (tested with 10.15)
- Windows (not tested, should be easy to fix)

## Caveats
Please take the following point into account when using our code.
- Spoken Language Identification is not a trivial problem. In order to teach a machine to perfectly recognize every small sentence of a language you need the full spectrum of a language. This requires absurd amounts of data.
- Due to acoustical overlap of languages, the task gets harder the more languages there are to distinguish from another.
- The challenge gets even more difficult when considering different accents, ages, sex and other traits that influence the vocal tract.
- In our first experiments we are able to distinguish 4 languages (and noise) with an overall accuracy of 85% on the Common Voice data set. Common Voice is a very diverse, noisy and community driven collection of spoken language.
- In order to achieve our final goal of a more inclusive museum experience, we need to focus on fairness. However, as of now we haven't evaluated or mitigated bias in our system.


## Trained Models
Our trained models can be downloaded from [this location](https://cloud.zkm.de/index.php/s/83LwnXT9xDsyxGf). The `AttRnn` model expects 5 seconds of normalized audio sampled at 16kHz and outputs probabilities for Noise, English, French, German and Spanish in this order.

## Demonstration
If you are only interested in running the trained model then please check out our notebook in the `demo/` folder. The notebook was developed to be [run in Google Colab](https://colab.research.google.com/github/zkmkarlsruhe/language-identification/blob/main/demo/LID_demo.ipynb). It will guide you through the necessary steps to run the model in your application.

You can also check out our [OpenFrameworks demonstration](https://git.zkm.de/Hertz-Lab/Research/intelligent-museum/LanguageIdentifier) which utilizes our [ofxTensorFlow2](https://github.com/zkmkarlsruhe/ofxTensorFlow2) addon.

## Training Setup
We highly recommend to use a (recent) GPU to train a neural network.

### Docker 

##### GPU support (recommended)
If you haven't installed docker with GPU support yet, please follow [these instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

##### Building the image
Build the docker image using the [Dockerfile](Dockerfile). Make sure not to include large amounts of data into the build process (all files in the build directory are taken into account).
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
- python 3
- pip 3


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

#### Python packages
```shell
pip install -r requirements.txt
```

## Dataset
This repository extracts language examples from Mozilla's open speech corpus 
[Common Voice](https://commonvoice.mozilla.org/). In order to enhance the robustness of the model against noises we use Google's audio scene dataset 
[AudioSet](https://research.google.com/audioset/dataset/index.html). 

### Download
##### Common Voice
Start by downloading language sets you are interested in. We recommend to use languages with at least 1000 speakers and 100 hours validated audio samples. Check [this site](https://commonvoice.mozilla.org/de/languages) for details.

__Note:__ In order to use our provided download script, you have to generate and copy machine-specific download links into it, as the download requires your consent.

```shell
./data/common-voice/download_common_voice.sh
```

In the end, you need a folder, referred to as `$CV_DL_DIR`, which contains subfolders for every language that you want to classify.

##### AudioSet
In this section we will first download the AudioSet meta data file from [this website](https://research.google.com/audioset/download.html). Next, we will search it for specific labels using the provided `data/audioset/download_youtube_noise.py` script. This python script defines the labels that are relevant and those that are not allowed (human voice). With the restrictions for our use case we extracted around 18,000 samples from AudioSet. You can call the shell script like this:
```shell
./data/audioset/download_yt_noise.sh
```
It will create a folder `yt-downloads` which contains the raw audio files (some may yet be flawed).

### Audio Extraction
We use several processing steps to form our data set from the Common Voice downloads. We recommend using the config file to define and document the processing steps. Please take a look at the CLI arguments in the script for more information on the options.
```shell
python data/common-voice/cv_to_wav.py --help
```
__Note:__ Modify the config file accordingly, e.g. replace `cv_input_dir` with `$CV_DL_DIR` and `cv_output_dir` with `$DATA_DIR` (the final dataset directory). Don't forget to name the languages in the table at the bottom.
```shell
python data/common-voice/cv_to_wav.py --config data/common-voice/config_moz.yaml
```
##### Add the Noise
Afterwards we check if the noise data is valid and cut and split it into the previously created `$DATA_DIR`.
Please use the provided shell script and pass it the `$DATA_DIR` path:
```shell
./data/audioset/process_and_split_noise.sh $DATA_DIR
```

### Preprocessing
In this version, we use [kapre](https://kapre.readthedocs.io/en/latest/) to extract the features (such as FFT or Mel-filterbanks) within the TensorFlow graph. This is especially useful in terms of portability, as we only need to pass the normalized audio to the model.

If you rather do the preprocessing separately and before training, you may want to utilize the script `data/process_wav.py` and its config file, as well as its dependant source files. In the future, we may create another branch which tackles the problem this way (as we used to do it before using kapre).


## Training
As with the creation of the dataset we use config files to define and document the process. The options we provide should sound familiar. Most importantly, modify the placeholder for the train and validation directories, as well as the languages to be detected (noise is treated as another language).

### Docker 
The following line runs the training process inside a docker container of the newly build image. The command will grant access to the folder holding the train and test set.
```shell
docker run -it --rm -v $(pwd):/work/src -v $DATA_DIR:/data lid python train.py --config config_train.yaml
```

### Local installation
```
python train.py --config config_train.yaml
```

## TODO
- evaluate the fairness of the model
- use a voice (instead of audio) activity detector  
- rework data loading process (e.g. use TFDataset)
- more automation in the data set creation steps


## Further Reading
* Types of [Speech Features](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
* We used [CRNN-LID](https://github.com/HPI-DeepLearning/crnn-lid) when we first started
* We ported the network from [this keyword-spotting code](https://github.com/douglas125/SpeechCmdRecognition)
* [VoxLingua107](http://bark.phon.ioc.ee/voxlingua107/) another multi-lingual dataset
* [Silero-VAD](https://github.com/snakers4/silero-vad) a free Voice Activity Detector and Language Identifier (en, es, de, ru)


## Contribute
Contributions are very welcome.
Create an account, clone or fork the repo, then request a push/merge.
If you find any bugs or suggestions please raise issues.


## The Intelligent Museum

An artistic-curatorial field of experimentation for deep learning and visitor participation

The [ZKM | Center for Art and Media](https://zkm.de/en) and the [Deutsches Museum Nuremberg](https://www.deutsches-museum.de/en/nuernberg/information/) cooperate with the goal of implementing an AI-supported exhibition. Together with researchers and international artists, new AI-based works of art will be realized during the next four years (2020-2023).  They will be embedded in the AI-supported exhibition in both houses. The Project „The Intelligent Museum” is funded by the Digital Culture Programme of the [Kulturstiftung des Bundes](https://www.kulturstiftung-des-bundes.de/en) (German Federal Cultural Foundation) and funded by the [Beauftragte der Bundesregierung für Kultur und Medien](https://www.bundesregierung.de/breg-de/bundesregierung/staatsministerin-fuer-kultur-und-medien) (Federal Government Commissioner for Culture and the Media).

As part of the project, digital curating will be critically examined using various approaches of digital art. Experimenting with new digital aesthetics and forms of expression enables new museum experiences and thus new ways of museum communication and visitor participation. The museum is transformed to a place of experience and critical exchange.

![Logo](media/Logo_ZKM_DMN_KSB.png)
