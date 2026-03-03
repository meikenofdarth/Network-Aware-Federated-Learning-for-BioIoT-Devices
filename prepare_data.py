import mne
import numpy as np
import os

# CHB-MIT Dataset: Patient 01, Record 03 contains a seizure between 2996s and 3036s
DATA_URL = "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf?download"
LOCAL_FILE = "chb01_03.edf"
OUTPUT_FILE = "silo-client/eeg_data.npy"

def download_and_process():
    print(f"⬇️  Downloading CHB-MIT Seizure Recording (approx 50MB)...")
    # We use os.system for a simple wget/curl since authentication isn't required for PhysioNet public
    if not os.path.exists(LOCAL_FILE):
        os.system(f"curl -L -o {LOCAL_FILE} '{DATA_URL}'")
    
    print("🧠 Processing EEG Signal...")
    # Read the EDF file
    raw = mne.io.read_raw_edf(LOCAL_FILE, preload=True, verbose=False)
    
    # Pick 1 Channel (FP1-F7 is often good for seizure detection)
    # We select the first channel index 0 to keep it generic
    raw.pick_channels([raw.ch_names[0]])
    
    # Resample to 100Hz to save bandwidth (Original is 256Hz)
    # This is "Edge Optimization" #1
    raw.resample(100)
    
    # Get numpy array
    data = raw.get_data()[0]
    
    # NORMALIZE (Critical for Neural Networks)
    # Scale between 0 and 1
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Save to the silo-client folder so Docker can pick it up
    np.save(OUTPUT_FILE, data)
    print(f"✅ Success: Processed {len(data)} EEG samples.")
    print(f"📂 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    download_and_process()