import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features
import warnings
warnings.filterwarnings("ignore")
import time


def load_gmm_model(model_path):
	gmm_files = [os.path.join(model_path, fname) for fname in os.listdir(model_path) if fname.endswith('.gmm')]
	speakers = [os.path.basename(fname).split(".gmm")[0] for fname in gmm_files]
	# Load the Gaussian gender Models
	gmm_model = []
	for fname in gmm_files:
		with open(fname, 'rb') as f:
			model = pickle.load(f)
			gmm_model.append(model)
	return gmm_model, speakers


def recognize_file(gmm_model, spk_list, file_path):
	sr, audio = read(file_path)
	vector = extract_features(audio, sr)

	log_likelihood = np.zeros(len(gmm_model))

	for i, gmm in enumerate(gmm_model):
		scores = np.array(gmm.score(vector))
		log_likelihood[i] = scores.sum()

	spk_vote = np.argmax(log_likelihood)
	spk_id = spk_list[spk_vote]
	probs = np.exp(log_likelihood[spk_vote]) / (np.exp(log_likelihood)).sum()

	# print("Testing audio {} was detected as - {}".format(file_path, spk_id))
	return spk_id, probs


def recognize_directory(gmm_model, spk_list, data_folder):
	if not os.path.isdir(data_folder):
		print('{} is not a directory.'.format(data_folder))
		return

	err_count = 0
	total_count = 0
	for test_file in os.listdir(data_folder):
		test_filepath = os.path.join(data_folder, test_file)
		rec_id = recognize_file(gmm_model, spk_list, test_filepath)

		checker_name = test_file.split("_")[1]
		if rec_id != checker_name:
			err_count += 1
		total_count += 1

	print("error={}, total_sample={}".format(err_count, total_count))
	accuracy = ((total_count - err_count) / total_count) * 100
	print("Accuracy : ", accuracy, "%")


if __name__ == '__main__':
	my_model, speakers = load_gmm_model('Speakers_models')
	recognize_directory(my_model, speakers, 'SampleData')
