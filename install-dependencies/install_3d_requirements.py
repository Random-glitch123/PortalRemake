import subprocess
import sys
import os

def install_requirements():
    print("Installing required libraries for 3D Portal Game...")
    # We don't need PyOpenGL_accelerate as it can cause issues on some systems
    requirements = ["pygame", "PyOpenGL", "numpy"]
    
    for package in requirements:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully!")
        except Exception as e:
            print(f"Error installing {package}: {e}")
            print(f"\nPlease try to install {package} manually:")
            print("1. Open Command Prompt as administrator")
            print(f"2. Type: pip install {package}")
            print("3. Press Enter and wait for installation to complete")
    
    print("\nAll required libraries have been installed (or attempted to install).")
    print("You can now run your 3D Portal Game by double-clicking on Portal3D.py or using the run_3d_game.bat file.")

if __name__ == "__main__":
    install_requirements()
    input("Press Enter to exit...")