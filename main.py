import librosa
import librosa.display
import numpy as np
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, AveragePooling2D

def create_spectrogram(filename, n_mels):
    audio_series, sampling_rate = librosa.load(filename)
    song, _ = librosa.effects.trim(audio_series)
    spectrogram = librosa.feature.melspectrogram(song, sr=sampling_rate, n_mels=n_mels)
    spectrogram_db = librosa.power_to_db(spectrogram)
    return spectrogram_db

def create_data(n_mels):
    genre_dict = {"blues": np.array([1,0,0,0,0,0,0,0,0,0]),
                  "classical": np.array([0,1,0,0,0,0,0,0,0,0]),
                  "country": np.array([0,0,1,0,0,0,0,0,0,0]),
                  "disco": np.array([0,0,0,1,0,0,0,0,0,0]),
                  "hiphop": np.array([0,0,0,0,1,0,0,0,0,0]),
                  "jazz": np.array([0,0,0,0,0,1,0,0,0,0]),
                  "metal": np.array([0,0,0,0,0,0,1,0,0,0]),
                  "pop": np.array([0,0,0,0,0,0,0,1,0,0]),
                  "reggae": np.array([0,0,0,0,0,0,0,0,1,0]),
                  "rock": np.array([0,0,0,0,0,0,0,0,0,1])}
    rootdir = "./genres"
    spectrograms = []
    labels = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            print(file)
            spec = create_spectrogram(os.path.join(subdir, file), n_mels)
            spec = np.delete(spec, slice(30,spec.shape[1]), 1)
            spectrograms.append(spec)
            labels.append(genre_dict[file.split(".")[0]])
    return [np.array(spectrograms), np.array(labels)]

def generate_train_test(n_mels):
    spectrograms, labels = create_data(n_mels)
    return train_test_split(spectrograms, labels, test_size=0.33)


X_train, X_test, y_train, y_test = generate_train_test(30)

def trainModel(model, epochs, optimizer):
    model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = 'accuracy')
    return model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epochs,
                     batch_size = 128)

X_train = X_train.reshape(669,30,30,1)
X_test = X_test.reshape(330,30,30,1)

model= Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(30, 30, 1)))
model.add(Dropout(0.2))
model.add(AveragePooling2D(pool_size = (2,2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(AveragePooling2D(pool_size = (2,2)))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

print(model.summary())

model_history = trainModel(model = model, epochs = 500, optimizer = 'adam')

test_loss, test_acc = model.evaluate(X_test, y_test, batch_size = 128)
print("Test loss: ", test_loss)
print("Test accuracy: ", test_acc)