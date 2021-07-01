from src.utils.training_utils import *

langs = ['french', 'english']

labeled_ds = create_dataset_from_set_of_files('test_data/test/', langs)

for audio, x, label in labeled_ds.take(3):
	print(x)
	print(label)
