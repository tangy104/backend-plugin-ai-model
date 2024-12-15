from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from moviepy.editor import VideoFileClip
from io import BytesIO
import tempfile
from .mod import process_audio

# Initialize the FastAPI application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # List your frontend's origin(s)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Function to convert video to audio
def convert_video_to_audio(video_file: BytesIO, temp_video_path: str, audio_path: str):
    try:
        with open(temp_video_path, "wb") as temp_video:
            temp_video.write(video_file.read())

        video = VideoFileClip(temp_video_path)
        video.audio.write_audiofile(audio_path, codec="libmp3lame")
        video.close()
        print(f"Audio file saved at: {audio_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting video to audio: {e}")

# Function to convert video file to WAV
def video_to_wav(temp_video_path: str, output_audio_path: str):
    try:
        # Load the video file
        video = VideoFileClip(temp_video_path)
        
        # Extract audio from the video and save as .wav
        video.audio.write_audiofile(output_audio_path, codec='pcm_s16le')  # PCM codec ensures .wav format
        video.close()
        print(f"Audio successfully extracted and saved to: {output_audio_path}")
        return output_audio_path
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Error converting video to WAV: {e}")

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    try:
        # Ensure the uploaded file is a video
        if not video.content_type.startswith("video"):
            raise HTTPException(status_code=400, detail="Uploaded file is not a video.")

        # Define temporary paths
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video_path = temp_video.name
            print(f"Temporary Video Path: {temp_video_path}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3:
            mp3_path = temp_mp3.name
            print(f"Temporary MP3 Path: {mp3_path}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            wav_path = temp_wav.name
            print(f"Temporary WAV Path: {wav_path}")

        # Convert video to MP3 audio
        convert_video_to_audio(video.file, temp_video_path, mp3_path)

        # Convert MP3 to WAV
        video_to_wav(temp_video_path, wav_path)

        # Process the audio file
        # b, json_data, json_data2 = process_audio(mp3_path, wav_path)
        json_data, json_data2 = process_audio(mp3_path, wav_path)

        # Clean up temporary files
        os.remove(temp_video_path)
        os.remove(mp3_path)
        os.remove(wav_path)

        # Return results
        return JSONResponse(content={
            # "b": b,
            "json_data": json_data,
            "json_data2": json_data2
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
