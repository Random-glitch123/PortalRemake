import subprocess
import sys
import os
import platform

def install_package(package):
    print(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully!")
        return True
    except Exception as e:
        print(f"✗ Error installing {package}: {e}")
        return False

def main():
    print("=" * 50)
    print("OpenGL and Dependencies Installer for Portal 3D Game")
    print("=" * 50)
    print("\nThis script will install all required packages for the 3D Portal game.")
    
    # Check Python version
    python_version = platform.python_version()
    print(f"\nDetected Python version: {python_version}")
    
    # Check if pip is installed
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        print("✓ pip is installed")
    except:
        print("✗ pip is not installed or not working properly")
        print("Please install pip first: https://pip.pypa.io/en/stable/installation/")
        input("\nPress Enter to exit...")
        return
    
    # Check operating system
    system = platform.system()
    print(f"Detected operating system: {system}")
    
    # Install required packages
    print("\nInstalling required packages...")
    
    # Core packages
    packages = [
        "pygame",
        "numpy"
    ]
    
    # OpenGL packages
    opengl_packages = [
        "PyOpenGL",
        "PyOpenGL_accelerate"
    ]
    
    # Install core packages
    success = True
    for package in packages:
        if not install_package(package):
            success = False
    
    # Install OpenGL packages
    for package in opengl_packages:
        if not install_package(package):
            success = False
    
    # Final message
    print("\n" + "=" * 50)
    if success:
        print("All packages installed successfully!")
        print("\nYou can now run the 3D Portal game by:")
        print("1. Double-clicking on run_3d_game.bat, or")
        print("2. Running 'python Portal3D.py' in a command prompt")
    else:
        print("Some packages could not be installed automatically.")
        print("\nTry installing them manually with these commands:")
        print("pip install pygame numpy PyOpenGL PyOpenGL_accelerate")
        
        if system == "Windows":
            print("\nIf you're still having issues with PyOpenGL, try downloading and installing")
            print("the appropriate wheel file from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl")
            print("\nFor example, if you have Python 3.9 64-bit, download:")
            print("PyOpenGL‑3.1.6‑cp39‑cp39‑win_amd64.whl")
            print("PyOpenGL_accelerate‑3.1.6‑cp39‑cp39‑win_amd64.whl")
            print("\nThen install them with:")
            print("pip install C:\\path\\to\\PyOpenGL‑3.1.6‑cp39‑cp39‑win_amd64.whl")
            print("pip install C:\\path\\to\\PyOpenGL_accelerate‑3.1.6‑cp39‑cp39‑win_amd64.whl")
    
    print("=" * 50)
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()