import torch
torch.set_num_threads(1)
from pprint import pprint
import os

model, lang_dict, lang_group_dict,  utils = torch.hub.load(
							repo_or_dir='snakers4/silero-vad',
							model='silero_lang_detector_116',
							force_reload=True)

get_language_and_group, read_audio = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

wav = read_audio(f'{files_dir}/de.wav')

test_file = f'test_data/train/english/common_voice_en_17245941_0.wav'

import glob

files = glob.glob("test_data/test/*/*.wav")


language_table = {	
		"english": "en",
		"german": "de",
		"french": "fr",
		"spanish":  "es",
		"mandarin": "zh-CN",
		"russian": "ru",
		"farsi": "fa",
		"polish": "pa"
}

tp_count = 0

for i, file in enumerate(files):

	split = file.split(os.sep)

	wav = read_audio(file)

	languages, language_groups = get_language_and_group(wav, model, lang_dict, lang_group_dict, top_n=1)

	for lang in languages:
		# pprint(f'Language: {lang[0]} with prob {lang[-1]}')

		# print(lang[0].split(",")[0])
		# print(language_table[split[2]])

		if lang[0].split(",")[0] == language_table[split[2]]:
			tp_count += 1
	# for i in language_groups:
	# pprint(f'Language group: {i[0]} with prob {i[-1]}')

print (tp_count / i)