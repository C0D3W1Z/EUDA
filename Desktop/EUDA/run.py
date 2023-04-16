
# Import necessary libraries
import librosa
import numpy as np
import tensorflow as tf
import sounddevice as sd
import sys

# Load the pre-trained model
model = tf.keras.models.load_model('emotion_detection_model.h5')

# Define a function to preprocess the audio data
def preprocess_audio(signal, sr):
    # Extract the Mel-frequency cepstral coefficients (MFCCs) from the audio signal
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    
    # Pad the MFCCs to a fixed length
    mfccs_padded = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])), mode='constant')
    
    # Reshape the MFCCs to match the input shape of the model
    mfccs_reshaped = mfccs_padded.reshape(1, 40, 100, 1)
    
    return mfccs_reshaped

# Define a callback function to process the microphone input
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    # Preprocess the audio data
    mfccs = preprocess_audio(indata.flatten(), sr=sd.query_devices(0)['default_samplerate'])
    # Use the model to predict the emotion
    emotion = model.predict(mfccs)[0]
    # Print the predicted emotion
    print("Emotion: ", np.argmax(emotion))

# Start the microphone input stream
with sd.InputStream(callback=callback):
    sd.sleep(10000)  # Keep the stream open for 10 seconds