from scipy.io import loadmat

mat = loadmat("EEG Data/eeg_record28.mat")
print(mat.keys())
obj = mat['o']
print(type(obj))
print(obj.shape)
print(obj)

o = mat['o'][0][0]
print("ID:", o[0])
print("Campo vazio?", o[1])
print("Nº de amostras:", o[2])
sfreq = o[3][0][0]
print("Frequência de amostragem:", sfreq, "Hz")
labels = o[4]  
print("Labels shape:", labels.shape)
print("Valores únicos:", set(labels.flatten()))
print("Primeiros 20 labels:", labels[:2000].flatten())
timestamps = o[5] 
print("Timestamps shape:", timestamps.shape)
print("Exemplo:", timestamps[:5])
meta = o[6]  # shape (308868, 25)
print("Meta shape:", meta.shape)
eeg = o[7]  # shape (1, 2, 14, 128)
print("EEG shape:", eeg.shape)
eeg = o[7][0, 0]  
print("Canais:", eeg.shape[0])
print("Amostras por canal:", eeg.shape[1])
print("Exemplo de sinal do canal 0:", eeg[0][:10])

print("Labels shape:", labels.shape)
print("Valores únicos:", set(labels.flatten()))
print("Primeiros 20 labels:", labels[:2000].flatten())