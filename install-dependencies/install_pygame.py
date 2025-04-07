import subprocess
import sys
import os

def install_pygame():
    print("Installing pygame...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
        print("Pygame installed successfully!")
        print("You can now run your game by double-clicking on Portalgame.py or running it from your IDE.")
    except Exception as e:
        print(f"Error installing pygame: {e}")
        print("\nAlternative installation method:")
        print("1. Open Command Prompt as administrator")
        print("2. Type: pip install pygame")
        print("3. Press Enter and wait for installation to complete")

if __name__ == "__main__":
    install_pygame()
    input("Press Enter to exit...")