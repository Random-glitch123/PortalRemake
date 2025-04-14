"""
Asset Manager Module

This module provides a centralized system for loading, caching, and managing
game assets such as textures and sounds.
"""

import os
import pygame
from OpenGL.GL import *
import numpy as np

class AssetManager:
    """
    Manages game assets including textures and sounds.
    
    This class handles loading, caching, and accessing game assets,
    with robust error handling and fallback mechanisms.
    """
    
    def __init__(self):
        """Initialize the asset manager and create necessary directories."""
        # Create asset directories if they don't exist
        self.texture_dir = os.path.join("assets", "textures")
        self.sound_dir = os.path.join("assets", "sounds")
        
        os.makedirs(self.texture_dir, exist_ok=True)
        os.makedirs(self.sound_dir, exist_ok=True)
        
        # Cache for loaded assets
        self.textures = {}
        self.sounds = {}
        
        # Print initialization message
        print("\nAsset Manager initialized")
        print(f"Texture directory: {self.texture_dir}")
        print(f"Sound directory: {self.sound_dir}")
    
    def load_texture(self, filename, force_reload=False):
        """
        Load a texture and return its OpenGL ID.
        
        Args:
            filename: Name of the texture file to load
            force_reload: If True, reload the texture even if it's in cache
            
        Returns:
            OpenGL texture ID or a default texture ID if loading fails
        """
        # Return cached texture if available and not forcing reload
        if filename in self.textures and not force_reload:
            return self.textures[filename]
            
        texture_path = os.path.join(self.texture_dir, filename)
        try:
            # Check if file exists
            if not os.path.exists(texture_path):
                print(f"Texture file not found: {texture_path}")
                print(f"Please add {filename} to the assets/textures directory.")
                return self._create_default_texture()

            print(f"Loading texture: {filename}")
            
            # Try to load the texture with error handling
            try:
                texture_surface = pygame.image.load(texture_path)
            except pygame.error as e:
                print(f"Error loading texture image {filename}: {e}")
                # Try alternative image formats if the specified one fails
                base_name, ext = os.path.splitext(filename)
                alt_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
                
                # Try each alternative extension
                for alt_ext in alt_extensions:
                    if alt_ext == ext:  # Skip if it's the same as the original extension
                        continue
                        
                    alt_path = os.path.join(self.texture_dir, base_name + alt_ext)
                    if os.path.exists(alt_path):
                        try:
                            print(f"Trying alternative format: {base_name + alt_ext}")
                            texture_surface = pygame.image.load(alt_path)
                            print(f"Successfully loaded alternative texture: {base_name + alt_ext}")
                            break
                        except pygame.error:
                            continue
                else:  # If no alternative format works
                    print(f"Could not load texture {filename} or any alternative format")
                    return self._create_default_texture()
            
            # Convert surface to RGBA if it's not already
            if texture_surface.get_bitsize() < 32:
                texture_surface = texture_surface.convert_alpha()
                
            # Create texture data
            texture_data = pygame.image.tostring(texture_surface, "RGBA", 1)
            width, height = texture_surface.get_size()

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            print(f"Successfully loaded texture: {filename} ({width}x{height})")
            
            # Cache the texture ID
            self.textures[filename] = texture_id
            return texture_id
            
        except Exception as e:
            print(f"Unexpected error loading texture {filename}: {e}")
            # Return a default texture ID (checkerboard pattern)
            default_texture = self._create_default_texture()
            self.textures[filename] = default_texture
            return default_texture

    def _create_default_texture(self):
        """
        Create a default checkerboard texture when texture files are missing.
        
        Returns:
            OpenGL texture ID for the default texture
        """
        size = 64
        texture_data = []
        for y in range(size):
            for x in range(size):
                if (x // 8 + y // 8) % 2 == 0:
                    texture_data.extend([255, 255, 255, 255])  # White
                else:
                    texture_data.extend([128, 128, 128, 255])  # Gray
        
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, bytes(texture_data))
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        
        return texture_id

    def load_sound(self, filename, force_reload=False):
        """
        Load a sound file and return a pygame Sound object.
        
        Args:
            filename: Name of the sound file to load
            force_reload: If True, reload the sound even if it's in cache
            
        Returns:
            pygame.mixer.Sound object or None if loading fails
        """
        # Return cached sound if available and not forcing reload
        if filename in self.sounds and not force_reload:
            return self.sounds[filename]
            
        sound_path = os.path.join(self.sound_dir, filename)
        try:
            # Check if file exists
            if not os.path.exists(sound_path):
                print(f"Sound file not found: {sound_path}")
                return None

            # Try to load the sound
            try:
                sound = pygame.mixer.Sound(sound_path)
                self.sounds[filename] = sound
                return sound
            except pygame.error as e:
                # If MP3 fails, check if there's a WAV or OGG version with the same name
                base_name = os.path.splitext(filename)[0]
                
                # Try WAV version
                wav_path = os.path.join(self.sound_dir, f"{base_name}.wav")
                if os.path.exists(wav_path):
                    print(f"MP3 format failed, using WAV version instead: {wav_path}")
                    sound = pygame.mixer.Sound(wav_path)
                    self.sounds[filename] = sound
                    return sound
                
                # Try OGG version
                ogg_path = os.path.join(self.sound_dir, f"{base_name}.ogg")
                if os.path.exists(ogg_path):
                    print(f"MP3 format failed, using OGG version instead: {ogg_path}")
                    sound = pygame.mixer.Sound(ogg_path)
                    self.sounds[filename] = sound
                    return sound
                
                # If no alternative found, raise the original error
                print(f"Could not load sound {filename}: {e}")
                print("Try converting your MP3 to WAV or OGG format for better compatibility.")
                return None
                
        except Exception as e:
            print(f"Error loading sound {filename}: {e}")
            return None
    
    def preload_assets(self, texture_list=None, sound_list=None):
        """
        Preload a list of assets to avoid loading delays during gameplay.
        
        Args:
            texture_list: List of texture filenames to preload
            sound_list: List of sound filenames to preload
            
        Returns:
            Tuple of (number of textures loaded, number of sounds loaded)
        """
        textures_loaded = 0
        sounds_loaded = 0
        
        # Preload textures
        if texture_list:
            print(f"Preloading {len(texture_list)} textures...")
            for texture_name in texture_list:
                if self.load_texture(texture_name):
                    textures_loaded += 1
        
        # Preload sounds
        if sound_list:
            print(f"Preloading {len(sound_list)} sounds...")
            for sound_name in sound_list:
                if self.load_sound(sound_name):
                    sounds_loaded += 1
        
        print(f"Preloaded {textures_loaded} textures and {sounds_loaded} sounds")
        return textures_loaded, sounds_loaded
    
    def clear_cache(self):
        """Clear the asset cache to free up memory."""
        # Note: This doesn't delete the OpenGL textures, just our references
        self.textures = {}
        self.sounds = {}
        print("Asset cache cleared")

# Create a global instance for easy importing
asset_manager = AssetManager()