import os
import sys
import glob
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_file, wav_file):
    """Convert an MP3 file to WAV format"""
    try:
        # Load the MP3 file
        sound = AudioSegment.from_mp3(mp3_file)
        
        # Export as WAV
        sound.export(wav_file, format="wav")
        print(f"Successfully converted {mp3_file} to {wav_file}")
        return True
    except Exception as e:
        print(f"Error converting {mp3_file}: {e}")
        return False

def convert_mp3_to_ogg(mp3_file, ogg_file):
    """Convert an MP3 file to OGG format"""
    try:
        # Load the MP3 file
        sound = AudioSegment.from_mp3(mp3_file)
        
        # Export as OGG
        sound.export(ogg_file, format="ogg")
        print(f"Successfully converted {mp3_file} to {ogg_file}")
        return True
    except Exception as e:
        print(f"Error converting {mp3_file}: {e}")
        return False

def convert_all_mp3_files(directory="assets/sounds", to_format="wav"):
    """Convert all MP3 files in a directory to WAV or OGG format"""
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Find all MP3 files
    mp3_files = glob.glob(os.path.join(directory, "*.mp3"))
    
    if not mp3_files:
        print(f"No MP3 files found in {directory}")
        return
    
    print(f"Found {len(mp3_files)} MP3 files to convert")
    
    # Convert each file
    for mp3_file in mp3_files:
        base_name = os.path.splitext(mp3_file)[0]
        
        if to_format.lower() == "wav":
            output_file = f"{base_name}.wav"
            convert_mp3_to_wav(mp3_file, output_file)
        elif to_format.lower() == "ogg":
            output_file = f"{base_name}.ogg"
            convert_mp3_to_ogg(mp3_file, output_file)
        else:
            print(f"Unsupported output format: {to_format}")
            return

if __name__ == "__main__":
    # If run directly, convert all MP3 files in the assets/sounds directory
    format_to_use = "wav"  # Default to WAV
    
    # Check if a format was specified
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ["wav", "ogg"]:
            format_to_use = sys.argv[1].lower()
        else:
            print(f"Unsupported format: {sys.argv[1]}")
            print("Supported formats: wav, ogg")
            sys.exit(1)
    
    print(f"Converting all MP3 files to {format_to_use.upper()} format...")
    convert_all_mp3_files(to_format=format_to_use)
    print("Conversion complete!")
    
    print("\nTo use this script:")
    print("1. Install pydub: pip install pydub")
    print("2. Install ffmpeg (required by pydub for conversion)")
    print("3. Run: python convert_audio.py [format]")
    print("   where [format] is 'wav' or 'ogg' (default is 'wav')")