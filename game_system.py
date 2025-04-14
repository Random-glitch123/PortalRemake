"""
Game System Module

This module provides centralized configuration and performance monitoring systems
for the Portal Game, replacing global variables with a more organized approach.
"""

import time
import numpy as np

class GameConfig:
    """
    Centralized configuration class for game settings.
    
    This class manages all configurable aspects of the game including
    display settings, graphics options, physics parameters, and gameplay values.
    """
    
    def __init__(self):
        """Initialize the configuration with default values."""
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


class PerformanceMonitor:
    """
    Monitors and manages game performance.
    
    This class tracks frame times, calculates FPS, and can automatically
    adjust graphics settings to maintain target performance.
    """
    
    def __init__(self, config):
        """
        Initialize the performance monitor.
        
        Args:
            config: GameConfig instance to access and modify settings
        """
        self.config = config
        self.frame_times = []
        self.window_size = config.perf_window_size
        self.last_time = time.time()
        self.fps_history = []
        
        # Detailed performance metrics
        self.physics_times = []
        self.render_times = []
        self.input_times = []
        
        # Performance flags
        self.adaptive_settings = config.adaptive_settings
        self.last_adjustment_time = 0
        self.adjustment_cooldown = 1.0  # seconds between adjustments
        
        print("Performance monitor initialized")
        print(f"Adaptive settings: {'Enabled' if self.adaptive_settings else 'Disabled'}")
        print(f"Target FPS: {config.max_fps}")

    def start_frame(self):
        """Start timing a new frame."""
        self.frame_start_time = time.time()
    
    def start_section(self, section_name):
        """
        Start timing a specific section of the frame processing.
        
        Args:
            section_name: Name of the section to time (e.g., 'physics', 'render', 'input')
        """
        setattr(self, f"{section_name}_start_time", time.time())
    
    def end_section(self, section_name):
        """
        End timing for a specific section and record the elapsed time.
        
        Args:
            section_name: Name of the section that was being timed
            
        Returns:
            Elapsed time for the section in seconds
        """
        start_time = getattr(self, f"{section_name}_start_time", self.frame_start_time)
        elapsed = time.time() - start_time
        
        # Store time in the appropriate list
        if section_name == 'physics':
            self.physics_times.append(elapsed)
            if len(self.physics_times) > self.window_size:
                self.physics_times.pop(0)
        elif section_name == 'render':
            self.render_times.append(elapsed)
            if len(self.render_times) > self.window_size:
                self.render_times.pop(0)
        elif section_name == 'input':
            self.input_times.append(elapsed)
            if len(self.input_times) > self.window_size:
                self.input_times.pop(0)
        
        return elapsed

    def end_frame(self):
        """
        End timing for the current frame and update performance metrics.
        
        Returns:
            Frame time in seconds
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Add frame time to history
        self.frame_times.append(dt)

        # Keep only the most recent frames
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

        # Calculate current FPS
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps_history.append(current_fps)

            # Keep only recent history
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)

        # Adjust settings if adaptive mode is enabled
        if self.adaptive_settings:
            self.adjust_settings()
            
        return dt

    def get_avg_fps(self):
        """
        Calculate the average FPS over the recent history.
        
        Returns:
            Average FPS or 0 if no frames have been recorded
        """
        if not self.frame_times:
            return 0

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        if avg_frame_time > 0:
            return 1.0 / avg_frame_time
        return 0
    
    def get_section_stats(self, section_name):
        """
        Get performance statistics for a specific section.
        
        Args:
            section_name: Name of the section to get stats for
            
        Returns:
            Dictionary with min, max, avg times in milliseconds
        """
        times_list = getattr(self, f"{section_name}_times", [])
        if not times_list:
            return {'min': 0, 'max': 0, 'avg': 0, 'percent': 0}
            
        # Convert to milliseconds for readability
        times_ms = [t * 1000 for t in times_list]
        
        # Calculate percentage of frame time
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            avg_section_time = sum(times_list) / len(times_list)
            percent = (avg_section_time / avg_frame_time) * 100 if avg_frame_time > 0 else 0
        else:
            percent = 0
            
        return {
            'min': min(times_ms) if times_ms else 0,
            'max': max(times_ms) if times_ms else 0,
            'avg': sum(times_ms) / len(times_ms) if times_ms else 0,
            'percent': percent
        }

    def adjust_settings(self):
        """
        Automatically adjust graphics settings based on performance.
        
        This method will reduce graphics quality if FPS is too low,
        and increase it if FPS is consistently high.
        """
        # Only adjust settings periodically to avoid oscillation
        current_time = time.time()
        if current_time - self.last_adjustment_time < self.adjustment_cooldown:
            return
            
        # Only adjust if we have enough history
        if len(self.fps_history) < 10:
            return

        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # If FPS is too low, reduce graphics settings
        if avg_fps < self.config.min_fps_threshold:
            # Reduce tessellation first
            if self.config.enable_tessellation and self.config.tessellation_level > 1:
                self.config.tessellation_level -= 1
                print(f"Performance: Reduced tessellation level to {self.config.tessellation_level}")
                self.last_adjustment_time = current_time
                return

            # Finally, disable lighting effects entirely
            if self.config.enable_lighting:
                self.config.enable_lighting = False
                print("Performance: Disabled enhanced lighting effects")
                self.last_adjustment_time = current_time
                return

        # If FPS is very high, we can increase settings
        elif avg_fps > self.config.high_fps_threshold and len(self.fps_history) >= 30:
            # First enable lighting effects if disabled
            if not self.config.enable_lighting:
                self.config.enable_lighting = True
                print("Performance: Enabled enhanced lighting effects")
                self.last_adjustment_time = current_time
                return

            # Finally increase tessellation
            if self.config.enable_tessellation and self.config.tessellation_level < 4:
                self.config.tessellation_level += 1
                print(f"Performance: Increased tessellation level to {self.config.tessellation_level}")
                self.last_adjustment_time = current_time
                return
    
    def print_performance_report(self):
        """Print a detailed performance report to the console."""
        avg_fps = self.get_avg_fps()
        physics_stats = self.get_section_stats('physics')
        render_stats = self.get_section_stats('render')
        input_stats = self.get_section_stats('input')
        
        print("\n===== PERFORMANCE REPORT =====")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Frame Time: {1000/avg_fps:.1f}ms (target: {1000/self.config.max_fps:.1f}ms)")
        print("\nBreakdown:")
        print(f"  Physics: {physics_stats['avg']:.2f}ms ({physics_stats['percent']:.1f}% of frame)")
        print(f"  Rendering: {render_stats['avg']:.2f}ms ({render_stats['percent']:.1f}% of frame)")
        print(f"  Input: {input_stats['avg']:.2f}ms ({input_stats['percent']:.1f}% of frame)")
        print("\nGraphics Settings:")
        print(f"  Lighting: {'Enabled' if self.config.enable_lighting else 'Disabled'}")
        print(f"  Tessellation: {'Enabled' if self.config.enable_tessellation else 'Disabled'} (Level: {self.config.tessellation_level})")
        print(f"  Dynamic LOD: {'Enabled' if self.config.dynamic_lod else 'Disabled'}")
        print("==============================\n")


# Create global instances for easy importing
config = GameConfig()
perf_monitor = PerformanceMonitor(config)