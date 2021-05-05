from flask import Flask
from flask import request

from src.utils.language_analyzer import LanguageAnalyzer

app = Flask(__name__)

analyzer = LanguageAnalyzer()

@app.route('/lid', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['the_file']
		filename = 'temp.wav'
		f.save(filename)
		out = analyzer.predict_on_audio_file(filename)
		print(out)
	return {
		'class': 1,
	}