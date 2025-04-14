"""
Game Configuration Module

This module provides a centralized configuration system for the Portal Game,
replacing global variables with a more organized, encapsulated approach.
"""

class GameConfig:
    """
    Centralized configuration class for game settings.
    
    This class manages all configurable aspects of the game including
    display settings, graphics options, physics parameters, and gameplay values.
    """
    
    def __init__(self):
        # Display settings
        self.screen_width = 1024
        self.screen_height = 768
        self.fullscreen = False
        self.max_fps = 60
        
        # Camera settings
        self.fov = 60
        self.near_plane = 0.1
        self.far_plane = 100.0
        
        # Physics constants - defined first since portal dimensions depend on these
        self.gravity = 9.8
        self.jump_force = 5.0
        self.player_height = 1.8
        self.player_radius = 0.5
        self.mouse_sensitivity = 0.2
        self.movement_speed = 5.0
        
        # Portal settings
        self.portal_distance = 100.0
        self.portal_width = 1.2
        # Make portal height proportional to player height (slightly taller)
        self.portal_height = self.player_height * 1.1  # 10% taller than player
        self.portal_segments = 36  # Increased for smoother portal edges
        self.portal_depth_offset = 0.02  # Reduced to minimize z-fighting
        self.portal_recursion_depth = 2  # Maximum recursion depth for portal-in-portal rendering
        
        # Graphics settings
        self.enable_lighting = True
        self.enable_tessellation = True
        self.tessellation_level = 2
        self.reflection_strength = 0.5
        self.dynamic_lod = True
        self.stencil_buffer_size = 8  # Ensure we have enough stencil bits for portal rendering
        self.prioritize_portal_rendering = True  # Prioritize portal rendering over other effects
        
        # Gameplay settings
        self.pickup_distance = 3.0
        self.carry_distance = 2.0
        
        # Performance monitoring
        self.adaptive_settings = True
        self.perf_window_size = 30
        self.min_fps_threshold = 30
        self.high_fps_threshold = 55
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode setting."""
        self.fullscreen = not self.fullscreen
        return self.fullscreen
    
    def toggle_lighting(self):
        """Toggle enhanced lighting setting."""
        self.enable_lighting = not self.enable_lighting
        return self.enable_lighting
    
    def toggle_tessellation(self):
        """Toggle tessellation setting."""
        self.enable_tessellation = not self.enable_tessellation
        return self.enable_tessellation
    
    def increase_tessellation(self):
        """Increase tessellation level up to a maximum of 4."""
        if self.tessellation_level < 4:
            self.tessellation_level += 1
        return self.tessellation_level
    
    def decrease_tessellation(self):
        """Decrease tessellation level down to a minimum of 1."""
        if self.tessellation_level > 1:
            self.tessellation_level -= 1
        return self.tessellation_level
    
    def update_player_height(self, new_height):
        """
        Update the player height and adjust portal height accordingly.
        
        Args:
            new_height: The new player height value
            
        Returns:
            The updated portal height
        """
        self.player_height = new_height
        # Update portal height to maintain proportion
        self.portal_height = self.player_height * 1.1  # 10% taller than player
        return self.portal_height
    
    def save_settings(self):
        """
        Save current settings to a configuration file.
        Not implemented yet - placeholder for future enhancement.
        """
        # TODO: Implement settings persistence
        pass
    
    def load_settings(self):
        """
        Load settings from a configuration file.
        Not implemented yet - placeholder for future enhancement.
        """
        # TODO: Implement settings loading
        pass

# Create a global instance for easy importing
config = GameConfig()