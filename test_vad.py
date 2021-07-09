
import glob
import torch
from collections import deque
torch.set_num_threads(1)
import torch.nn.functional as F

from pprint import pprint

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
							model='silero_vad',
							force_reload=True)

(get_speech_ts,
_,
save_audio,
read_audio,
state_generator,
single_audio_stream,
collect_chunks) = utils


def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

def get_speech_ts_adaptive(wav: torch.Tensor,
					model,
					batch_size: int = 200,
					step: int = 500,
					num_samples_per_window: int = 4000, # Number of samples per audio chunk to feed to NN (4000 for 16k SR, 2000 for 8k SR is optimal)
					min_speech_samples: int = 10000,  # samples
					min_silence_samples: int = 4000,
					speech_pad_samples: int = 2000,
					run_function=validate,
					visualize_probs=False,
					device='cpu'):
	"""
	This function is used for splitting long audios into speech chunks using silero VAD
	Attention! All default sample rate values are optimal for 16000 sample rate model, if you are using 8000 sample rate model optimal values are half as much!
	Parameters
	----------
	batch_size: int
		batch size to feed to silero VAD (default - 200)
	step: int
		step size in samples, (default - 500)
	num_samples_per_window: int
		window size in samples (chunk length in samples to feed to NN, default - 4000)
	min_speech_samples: int
		if speech duration is shorter than this value, do not consider it speech (default - 10000)
	min_silence_samples: int
		number of samples to wait before considering as the end of speech (default - 4000)
	speech_pad_samples: int
		widen speech by this amount of samples each side (default - 2000)
	run_function: function
		function to use for the model call
	visualize_probs: bool
		whether draw prob hist or not (default: False)
	device: string
		torch device to use for the model call (default - "cpu")
	Returns
	----------
	speeches: list
		list containing ends and beginnings of speech chunks (in samples)
	"""
	if visualize_probs:
		import pandas as pd    

	num_samples = num_samples_per_window
	num_steps = int(num_samples / step)
	assert min_silence_samples >= step
	outs = []
	to_concat = []
	for i in range(0, len(wav), step):
		chunk = wav[i: i+num_samples]
		if len(chunk) < num_samples:
			chunk = F.pad(chunk, (0, num_samples - len(chunk)))
		to_concat.append(chunk.unsqueeze(0))
		if len(to_concat) >= batch_size:
			chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
			out = run_function(model, chunks)
			outs.append(out)
			to_concat = []

	if to_concat:
		chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
		out = run_function(model, chunks)
		outs.append(out)

	outs = torch.cat(outs, dim=0).cpu()

	buffer = deque(maxlen=num_steps)
	triggered = False
	speeches = []
	smoothed_probs = []
	current_speech = {}
	speech_probs = outs[:, 1]  # 0 index for silence probs, 1 index for speech probs
	median_probs = speech_probs.median()

	trig_sum = 0.89 * median_probs + 0.08 # 0.08 when median is zero, 0.97 when median is 1

	temp_end = 0
	for i, predict in enumerate(speech_probs):
		buffer.append(predict)
		smoothed_prob = max(buffer)
		if visualize_probs:
			smoothed_probs.append(float(smoothed_prob))
		if (smoothed_prob >= trig_sum) and temp_end:
			temp_end = 0
		if (smoothed_prob >= trig_sum) and not triggered:
			triggered = True
			current_speech['start'] = step * max(0, i-num_steps)
			continue
		if (smoothed_prob < trig_sum) and triggered:
			if not temp_end:
				temp_end = step * i
			if step * i - temp_end < min_silence_samples:
				continue
			else:
				current_speech['end'] = temp_end
				if (current_speech['end'] - current_speech['start']) > min_speech_samples:
					speeches.append(current_speech)
				temp_end = 0
				current_speech = {}
				triggered = False
				continue
	if current_speech:
		current_speech['end'] = len(wav)
		speeches.append(current_speech)
	if visualize_probs:
		pd.DataFrame({'probs': smoothed_probs}).plot(figsize=(16, 8))

	logic = [smoothed_prob >= float(trig_sum) for smoothed_prob in smoothed_probs]
	return logic

	for i, ts in enumerate(speeches):
		if i == 0:
			ts['start'] = max(0, ts['start'] - speech_pad_samples)
		if i != len(speeches) - 1:
			silence_duration = speeches[i+1]['start'] - ts['end']
			if silence_duration < 2 * speech_pad_samples:
				ts['end'] += silence_duration // 2
				speeches[i+1]['start'] = max(0, speeches[i+1]['start'] - silence_duration // 2)
			else:
				ts['end'] += speech_pad_samples
		else:
			ts['end'] = min(len(wav), ts['end'] + speech_pad_samples)

	return speeches


files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

test_file = f'{files_dir}/en.wav'
# test_file = f'test_data/train/english/common_voice_en_17245941_0.wav'
# test_file = f'/home/paul/Projects/language-identification/test_data/train/__noise/Groundhog Screaming.wav'

wav = read_audio(test_file)
# get speech timestamps from full audio file
logic = get_speech_ts_adaptive(wav, model, step=500, num_samples_per_window=4000, visualize_probs=True)
# pprint(logic)


from src.audio.utils import LogicDataSource, LogicValidater
from auditok.core import StreamTokenizer
 
dsource = LogicDataSource(logic)
 
tokenizer = StreamTokenizer(validator=LogicValidater(),
                               min_length=2.5,
                               max_length=5,
                               max_continuous_silence=2,
                               mode=StreamTokenizer.STRICT_MIN_LENGTH)
out = tokenizer.tokenize(dsource)
print(out)