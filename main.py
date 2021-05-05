

command = "python mozilla_to_wav.py --input_dir " + input_dir \
            "--output_dir" + output_dir \
            "--total_chops" + max_chops

command = "process_wav_files.py --source " + \
    "--img_dir" + \
    "--run_as_thread" + \
    "--feature_type" + \
    "--feature_nu" + \

os.system(command)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--config', default="config.yaml")
    cli_args = parser.parse_args()
