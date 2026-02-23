# import os
# import librosa
# import numpy as np
# import pandas as pd


# # -----------------------------
# # Feature Extraction Function
# # -----------------------------
# def extract_features(y, sr):
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     mfcc_mean = np.mean(mfcc, axis=1)

#     energy = np.mean(librosa.feature.rms(y=y))
#     zcr = np.mean(librosa.feature.zero_crossing_rate(y))
#     spectral_centroid = np.mean(
#         librosa.feature.spectral_centroid(y=y, sr=sr)
#     )

#     return np.hstack((
#         mfcc_mean,
#         energy,
#         zcr,
#         spectral_centroid
#     ))


# # -----------------------------
# # Main Processing Logic
# # -----------------------------
# def main():
#     dataset_path = "SpeechDataset"
#     speech_path = os.path.join(dataset_path, "Speech")
#     noise_path = os.path.join(dataset_path, "_background_noise_")

#     data = []
#     labels = []

#     print("Processing Speech Audio Files...")

#     # Process all speech folders
#     for word_folder in os.listdir(speech_path):
#         word_folder_path = os.path.join(speech_path, word_folder)

#         if os.path.isdir(word_folder_path):
#             for file in os.listdir(word_folder_path):
#                 if file.endswith(".wav" or ".mp4"):
#                     file_path = os.path.join(word_folder_path, file)

#                     y, sr = librosa.load(file_path, sr=None)
#                     features = extract_features(y, sr)

#                     data.append(features)
#                     labels.append(1)  # Speech

#     print("Processing Noise Audio Files...")

#     # Process long noise files (split into 1-sec chunks)
#     for file in os.listdir(noise_path):
#         if file.endswith(".wav"):
#             file_path = os.path.join(noise_path, file)

#             y, sr = librosa.load(file_path, sr=None)
#             chunk_length = sr  # 1 second

#             total_chunks = len(y) // chunk_length

#             for i in range(total_chunks):
#                 chunk = y[i * chunk_length:(i + 1) * chunk_length]

#                 features = extract_features(chunk, sr)
#                 data.append(features)
#                 labels.append(0)  # Non-speech

#     # Create DataFrame
#     feature_names = (
#         [f"MFCC_{i+1}" for i in range(13)] +
#         ["Energy", "ZCR", "SpectralCentroid"]
#     )

#     df = pd.DataFrame(data, columns=feature_names)
#     df["Label"] = labels

#     # Save CSV
#     df.to_csv("vad_feature_dataset.csv", index=False)

#     print("✅ Feature extraction complete!")
#     print("📄 CSV saved as vad_feature_dataset.csv")
#     print("Total samples:", len(df))


# if __name__ == "__main__":
#     main()


import librosa
import numpy as np
import pandas as pd

def extract_features(y, sr):
    # Your existing feature extraction logic
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    energy = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return np.hstack((mfcc_mean, energy, zcr, spectral_centroid))

def process_single_file(file_path):
    try:
        # librosa handles mp4 by using soundfile or audioread backends
        y, sr = librosa.load(file_path, sr=None)
        
        features = extract_features(y, sr)
        
        # Define names for clarity
        feature_names = ([f"MFCC_{i+1}" for i in range(13)] + 
                         ["Energy", "ZCR", "SpectralCentroid"])
        
        # Create a clean display
        results = pd.Series(features, index=feature_names)
        return results

    except Exception as e:
        return f"Error processing file: {e}"

# --- EXECUTION ---
file_to_check = "test_Speech.mp4" # Replace with your actual file name
feature_results = process_single_file(file_to_check)

print(f"--- Features for {file_to_check} ---")
print(feature_results)