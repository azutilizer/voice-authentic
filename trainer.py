import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
from featureextraction import extract_features
import warnings
warnings.filterwarnings("ignore")


def training(data_folder, model_output_folder):

    # Extracting features for each speaker (10 files per speakers)
    for spk_id in os.listdir(data_folder):
        spk_path = os.path.join(data_folder, spk_id)
        if not os.path.isdir(spk_path):
            continue

        features = np.asarray(())
        for spk_file in os.listdir(spk_path):
            file_path = os.path.join(spk_path, spk_file)

            print(file_path)

            # read the audio
            sr, audio = read(file_path)

            # extract 40 dimensional MFCC & delta MFCC features
            vector = extract_features(audio, sr)

            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))

        # model training
        gmm = GMM(n_components=16, covariance_type='diag', n_init=3)
        gmm.fit(features)

        # dumping the trained gaussian model
        spk_model = "{}.gmm".format(spk_id)
        with open(os.path.join(model_output_folder, spk_model), 'wb') as f:
            pickle.dump(gmm, f)
        print('+ modeling completed for speaker:', spk_model, " with data point = ", features.shape)


if __name__ == '__main__':
    data = 'trainingData'
    model_path = 'Speakers_models'
    training(data, model_path)
