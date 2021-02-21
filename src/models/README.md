# Language Identification Using Deep Convolutional Recurrent Neural Networks
- This is a fork of the following repository: 'https://github.com/HPI-DeepLearning/crnn-lid'
- The code was ported to and tested under python 3.7 and tensorflow 2.3
- For GPU support please refer to tensorflow's installation page

### Structure
- models/: contains a variety of tested architectures
- trained_models/: contains a CRNN model trained on spectrogram data of 6 languages 
(English, German, French, Spanish, Mandarin and Russian) of Mozilla's Common Voice speech corpus
- logs/: will contain the training logs and models after the training procedure has been started
- test/: contains spectrogram test data

## Usage
To start training set the desired properties in config.yaml and run:
```
$ python train.py 
```
For evaluation use the same config.yaml and run:
```
$ python evaluate.py
```
To predict on unseen data run:
```
$ python predict.py --input $PATH_TO/image.png
```

#### Labels
```
0 Mandarin,
1 English,
2 French,
3 German,
4 Russian,
5 Spanish
```


