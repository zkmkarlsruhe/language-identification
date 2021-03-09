input_dir = config["common_voice_dir"]
output_dir = config["chopped_wavs"]
max_chops = config["max_chops"]


command = "python mozilla_to_wav.py --input_dir " + input_dir \
            "--output_dir" + output_dir \
            "--total_chops" + max_chops

source = output_dir + 
img_dir = config[""]
feature_type = config[""]
feature_nu = config[""]
augment_nu = config[""]

command = "process_wav_files.py --source " + \
    "--img_dir" + \
    "--run_as_thread" + \
    "--feature_type" + \
    "--feature_nu" + \
if augment:
    command += "--augment" 
    command += "--augment_nu" + augment_nu


os.system(command)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--config', default="config.yaml")
    cli_args = parser.parse_args()

    mozilla data 