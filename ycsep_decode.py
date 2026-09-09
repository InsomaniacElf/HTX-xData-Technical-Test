import os
import tempfile

import pandas as pd
import requests


CSV_PATH = r"C:\Users\Sneha\Downloads\YCSEP_static.csv"
API_URL = "http://localhost:8001/asr"


# Load only the columns we need for this test
df = pd.read_csv(
    CSV_PATH,
    usecols=["channel", "text", "audio"]
)

# Keep only The Daily Ketchup Podcast
tdk = df[df["channel"] == "The_Daily_Ketchup_Podcast"]

# Take just ONE clip for testing
row = tdk.iloc[0]

print("Reference text:")
print(row["text"])
print()
print("Audio URL:")
print(row["audio"])
print()


# Download the audio
response = requests.get(row["audio"], timeout=30)
response.raise_for_status()

with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp3"
) as tmp:
    tmp.write(response.content)
    audio_path = tmp.name


try:
    # Send audio to our ASR API
    with open(audio_path, "rb") as audio_file:
        api_response = requests.post(
            API_URL,
            files={
                "file": (
                    "audio.mp3",
                    audio_file,
                    "audio/mpeg"
                )
            },
            timeout=120
        )

    api_response.raise_for_status()

    result = api_response.json()

    print("Generated transcription:")
    print(result["transcription"])
    print()
    print("ASR duration:")
    print(result["duration"], "seconds")

finally:
    # Delete downloaded audio after processing
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print()
        print("Temporary audio deleted.")
        