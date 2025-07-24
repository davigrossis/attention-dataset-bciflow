import numpy as np
from scipy.io import loadmat
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.fe.logpower import logpower
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

mat = loadmat("EEG Data/eeg_record14.mat")
o = mat['o'][0][0]



sfreq = o[3][0][0]                  # Frequência de amostragem (128.0)
labels = o[4].flatten()             # Labels por amostra (308868,)
timestamps = o[5]                   # shape (308868, 6)
meta = o[6]                         # shape (308868, 25)

eeg_continuo = meta[:, 2:16].T      # shape (14, 308868)


X = np.expand_dims(np.expand_dims(eeg_continuo, axis=0), axis=0) # shape (1, 1, 14, 308868)
sfreq = int(o[3][0][0])
y = labels.shape[0]
n_amostras = labels.shape[0]

labels = np.zeros(n_amostras, dtype=np.uint8) # até 10 minutos
focus_end = int(10 * 60 * sfreq) # a partir de 10 minutos 
unfocus_end = int(20 * 60 * sfreq) # a partir de 20 minutos

labels[focus_end:unfocus_end] = 1 # de 10 a 20 minutos todos 1
labels[unfocus_end:] = 2 # de 20 minutos em diante todos 2

events = {
  "focused": [0,600],
  "unfocused": [600,1200],
  "drowsy": [1200,2100],
}
ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

y_dict = {"focused": 0, "unfocused": 1, "drowsy": 2}
tmin = 0.0

dataset = {
    'X': X,                     
    'y': y,           
    'sfreq': sfreq,
    'y_dict': y_dict,
    'events': events,
    'ch_names': ch_names,
    'tmin': tmin
}

print("EEG signals shape:", dataset["X"].shape)
print("Labels:", dataset["y"])
print("Class dictionary:", dataset["y_dict"])
print("Events:", dataset["events"])
print("Channel names:", dataset["ch_names"])
print("Sampling frequency (Hz):", dataset["sfreq"])
print("Start time (s):", dataset["tmin"])

#x = (1,1 ,14, 308868) -> (1, 1, 14, 308868)
#y = (308868)
#colocar o 2