import librosa
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def compute_mfcc(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(mfccs, xticklabels=False, yticklabels=[f"MFCC-{i+1}" for i in range(mfccs.shape[0])], cmap='coolwarm', ax = ax)
    ax.set_title("MFCC Heatmap")
    ax.set_xlabel("Frames")
    ax.set_ylabel("MFCC Coefficients")
    # plt.show()

    return mfccs, fig
  
def compute_zero_crossing_rate(audio_path):
    y, sr = librosa.load(audio_path)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    # Plotting the ZCR
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(zcr, label="Zero Crossing Rate")
    ax.set_title("Zero Crossing Rate over Time")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Rate")
    ax.legend()
    # plt.show()

    return zcr, fig
  
def compute_gunning_index(text):
    sentences = text.split('.')
    words = text.split()
    syllables = sum([len(word) for word in words])  # Approximation
    avg_words_per_sentence = len(words) / len(sentences)
    avg_syllables_per_word = syllables / len(words)
    
    gunning_index = 0.4 * (avg_words_per_sentence + 100 * avg_syllables_per_word)
    return gunning_index
  
def compute_error_density(errors, total_words):
    error_density = errors / total_words
    
    # Pie chart
    labels = ['Errors', 'Correct']
    sizes = [errors, total_words - errors]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=['red', 'yellow'])
    ax.set_title("Error Density")
    # plt.show()

    return error_density, fig
  
def calculate_snr(audio, noise):
    signal_power = np.mean(audio**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Function to classify the audio as Clear (1) or Not Clear (0) based on SNR
def classify_audio(audio_file):
    try:
        # Load the audio file (ensure the file is mono and sampled at 16kHz)
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)

        # Optional: Generate noise sample (if separate noise file is not available)
        # Here, we'll assume noise is the first 0.5 seconds of the audio
        noise_duration = int(0.5 * sr)  # First 0.5 seconds
        noise = audio[:noise_duration]
        signal = audio[noise_duration:]

        # Calculate SNR
        snr = calculate_snr(signal, noise)

        # Define threshold for classification
        if snr > 20:
            return "clear", snr  # Clear
        else:
            return "not clear", snr  # Not Clear
    except Exception as e:
        # Handle any errors that might occur, such as file loading issues
        print(f"Error processing audio file: {e}")
        return None, None

def compute_pause_pattern(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # Define an amplitude threshold for speech vs. silence
    threshold = 0.02  # Adjust based on your file

    # Detect non-silent parts (i.e., speech)
    speech_segments = librosa.effects.split(y, top_db=20)  # top_db is the amplitude threshold for detecting silence

    # Calculate pauses (silent parts) and their durations
    pauses = []
    start_of_last_speech = 0
    for start, end in speech_segments:
        if start > start_of_last_speech:
            pause_duration = (start - start_of_last_speech) / sr  # Calculate pause duration in seconds
            pauses.append(pause_duration)
        start_of_last_speech = end

    # Calculate average pause duration
    avg_pause = np.mean(pauses) if pauses else 0

    # Plot the original and segmented waveform
    fig, ax = plt.subplots(figsize=(6, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    
    # Highlight speech sections in red
    for start, end in speech_segments:
        ax.axvspan(start / sr, end / sr, color='red', alpha=0.3)  # Speech in red

    ax.set_title("Speech Segments Highlighted in Red")
    plt.tight_layout()

    return avg_pause, fig
  
def compute_words_per_second(audio_path, text):
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    words = len(text.split())
    words_per_second = words / duration

    # Simulate a varying plot for demonstration
    time_intervals = np.linspace(0, duration, len(text.split()))
    wps = np.random.uniform(words_per_second - 0.5, words_per_second + 0.5, len(time_intervals))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time_intervals, wps, label="Words Per Second")
    ax.set_title("Words Per Second Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Words Per Second")
    ax.legend()
    # ax.set_show()

    return words_per_second, fig
