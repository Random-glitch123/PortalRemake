"""
UI System Module

This module provides a comprehensive UI system for the Portal Game,
including all screens, UI components, and settings management.
"""

import pygame
import sys
import os
import time
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from game_system import config

# Constants
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 20
BUTTON_COLOR = (100, 100, 100)
BUTTON_HOVER_COLOR = (150, 150, 150)
BUTTON_TEXT_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)
LOADING_BAR_COLOR = (0, 150, 255)
LOADING_BAR_BG_COLOR = (50, 50, 50)
SLIDER_WIDTH = 300
SLIDER_HEIGHT = 20
SLIDER_HANDLE_SIZE = 30
SLIDER_COLOR = (80, 80, 80)
SLIDER_HANDLE_COLOR = (0, 150, 255)
SLIDER_ACTIVE_COLOR = (0, 200, 255)

#------------------------------------------------------------------------------
# UI Components
#------------------------------------------------------------------------------

class Button:
    """
    A clickable button UI component.
    
    This class represents a button that can be clicked to trigger an action.
    It supports hover effects and custom actions.
    """
    def __init__(self, x, y, width, height, text, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hover = False
        
    def draw(self, surface, font):
        # Draw button background
        color = BUTTON_HOVER_COLOR if self.hover else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BUTTON_TEXT_COLOR, self.rect, 2)  # Border
        
        # Draw button text
        text_surf = font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
    def check_hover(self, mouse_pos):
        self.hover = self.rect.collidepoint(mouse_pos)
        return self.hover
        
    def check_click(self, mouse_pos, mouse_click):
        if self.rect.collidepoint(mouse_pos) and mouse_click:
            if self.action:
                self.action()
            return True
        return False

class Slider:
    """
    A slider UI component for adjusting numeric values.
    
    This class represents a slider that can be dragged to adjust a value
    within a specified range.
    """
    def __init__(self, x, y, width, height, min_val, max_val, current_val, label, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_rect = pygame.Rect(0, 0, SLIDER_HANDLE_SIZE, SLIDER_HANDLE_SIZE)
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = current_val
        self.label = label
        self.action = action
        self.active = False
        self.update_handle_position()
        
    def update_handle_position(self):
        # Calculate handle position based on current value
        value_range = self.max_val - self.min_val
        if value_range == 0:
            value_range = 1  # Prevent division by zero
        
        position = ((self.current_val - self.min_val) / value_range) * self.rect.width
        self.handle_rect.centerx = self.rect.left + position
        self.handle_rect.centery = self.rect.centery
        
    def draw(self, surface, font):
        # Draw slider background
        pygame.draw.rect(surface, SLIDER_COLOR, self.rect)
        pygame.draw.rect(surface, BUTTON_TEXT_COLOR, self.rect, 2)  # Border
        
        # Draw handle
        handle_color = SLIDER_ACTIVE_COLOR if self.active else SLIDER_HANDLE_COLOR
        pygame.draw.circle(surface, handle_color, self.handle_rect.center, SLIDER_HANDLE_SIZE // 2)
        pygame.draw.circle(surface, BUTTON_TEXT_COLOR, self.handle_rect.center, SLIDER_HANDLE_SIZE // 2, 2)  # Border
        
        # Draw label
        label_text = font.render(f"{self.label}: {int(self.current_val)}", True, BUTTON_TEXT_COLOR)
        label_rect = label_text.get_rect(midright=(self.rect.left - 20, self.rect.centery))
        surface.blit(label_text, label_rect)
        
    def check_hover(self, mouse_pos):
        return self.handle_rect.collidepoint(mouse_pos)
        
    def check_click(self, mouse_pos, mouse_click):
        if self.handle_rect.collidepoint(mouse_pos) and mouse_click:
            self.active = True
            return True
        return False
        
    def update(self, mouse_pos, mouse_down):
        if self.active and mouse_down:
            # Update handle position based on mouse
            new_x = max(self.rect.left, min(mouse_pos[0], self.rect.right))
            self.handle_rect.centerx = new_x
            
            # Calculate new value
            position_ratio = (new_x - self.rect.left) / self.rect.width
            self.current_val = self.min_val + position_ratio * (self.max_val - self.min_val)
            
            # Call action if provided
            if self.action:
                self.action(self.current_val)
                
            return True
        return False
        
    def release(self):
        self.active = False

class Checkbox:
    """
    A checkbox UI component for toggling boolean values.
    
    This class represents a checkbox that can be clicked to toggle between
    checked and unchecked states.
    """
    def __init__(self, x, y, size, label, checked=False, action=None):
        self.rect = pygame.Rect(x, y, size, size)
        self.label = label
        self.checked = checked
        self.action = action
        self.hover = False
        
    def draw(self, surface, font):
        # Draw checkbox background
        color = BUTTON_HOVER_COLOR if self.hover else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BUTTON_TEXT_COLOR, self.rect, 2)  # Border
        
        # Draw check mark if checked
        if self.checked:
            inner_rect = pygame.Rect(
                self.rect.left + 4, 
                self.rect.top + 4, 
                self.rect.width - 8, 
                self.rect.height - 8
            )
            pygame.draw.rect(surface, BUTTON_TEXT_COLOR, inner_rect)
        
        # Draw label
        label_text = font.render(self.label, True, BUTTON_TEXT_COLOR)
        label_rect = label_text.get_rect(midleft=(self.rect.right + 10, self.rect.centery))
        surface.blit(label_text, label_rect)
        
    def check_hover(self, mouse_pos):
        self.hover = self.rect.collidepoint(mouse_pos)
        return self.hover
        
    def check_click(self, mouse_pos, mouse_click):
        if self.rect.collidepoint(mouse_pos) and mouse_click:
            self.checked = not self.checked
            if self.action:
                self.action(self.checked)
            return True
        return False

class Dropdown:
    """
    A dropdown UI component for selecting from a list of options.
    
    This class represents a dropdown menu that can be expanded to show
    a list of options to select from.
    """
    def __init__(self, x, y, width, height, options, current_index, label, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.options = options
        self.current_index = current_index
        self.label = label
        self.action = action
        self.hover = False
        self.expanded = False
        self.option_rects = []
        
    def draw(self, surface, font):
        # Draw dropdown background
        color = BUTTON_HOVER_COLOR if self.hover else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BUTTON_TEXT_COLOR, self.rect, 2)  # Border
        
        # Draw current option
        if 0 <= self.current_index < len(self.options):
            text = self.options[self.current_index]
        else:
            text = "Select..."
            
        text_surf = font.render(text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(midleft=(self.rect.left + 10, self.rect.centery))
        surface.blit(text_surf, text_rect)
        
        # Draw dropdown arrow
        arrow_points = [
            (self.rect.right - 20, self.rect.centery - 5),
            (self.rect.right - 10, self.rect.centery - 5),
            (self.rect.right - 15, self.rect.centery + 5)
        ]
        pygame.draw.polygon(surface, BUTTON_TEXT_COLOR, arrow_points)
        
        # Draw label
        label_text = font.render(f"{self.label}:", True, BUTTON_TEXT_COLOR)
        label_rect = label_text.get_rect(midright=(self.rect.left - 20, self.rect.centery))
        surface.blit(label_text, label_rect)
        
        # Draw expanded options if expanded
        if self.expanded:
            self.option_rects = []
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(
                    self.rect.left,
                    self.rect.bottom + i * self.rect.height,
                    self.rect.width,
                    self.rect.height
                )
                self.option_rects.append(option_rect)
                
                # Highlight current selection
                if i == self.current_index:
                    pygame.draw.rect(surface, BUTTON_HOVER_COLOR, option_rect)
                else:
                    pygame.draw.rect(surface, BUTTON_COLOR, option_rect)
                    
                pygame.draw.rect(surface, BUTTON_TEXT_COLOR, option_rect, 2)  # Border
                
                option_text = font.render(option, True, BUTTON_TEXT_COLOR)
                option_text_rect = option_text.get_rect(midleft=(option_rect.left + 10, option_rect.centery))
                surface.blit(option_text, option_text_rect)
        
    def check_hover(self, mouse_pos):
        self.hover = self.rect.collidepoint(mouse_pos)
        return self.hover
        
    def check_click(self, mouse_pos, mouse_click):
        if self.rect.collidepoint(mouse_pos) and mouse_click:
            self.expanded = not self.expanded
            return True
            
        # Check if an option was clicked
        if self.expanded and mouse_click:
            for i, option_rect in enumerate(self.option_rects):
                if option_rect.collidepoint(mouse_pos):
                    self.current_index = i
                    self.expanded = False
                    if self.action:
                        self.action(i)
                    return True
                    
        return False

#------------------------------------------------------------------------------
# Game Screens
#------------------------------------------------------------------------------

class SplashScreen:
    """
    The main menu/splash screen for the game.
    
    This screen is shown when the game starts and provides options to
    start a new game, load a saved game, access settings, etc.
    """
    def __init__(self, screen):
        self.screen = screen
        self.running = True
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        
        # Create buttons
        center_x = SCREEN_WIDTH // 2
        start_y = SCREEN_HEIGHT // 2
        
        self.buttons = [
            Button(center_x - BUTTON_WIDTH // 2, start_y, 
                   BUTTON_WIDTH, BUTTON_HEIGHT, "New Game", self.new_game),
            Button(center_x - BUTTON_WIDTH // 2, start_y + BUTTON_HEIGHT + BUTTON_MARGIN, 
                   BUTTON_WIDTH, BUTTON_HEIGHT, "Load Game", self.load_game),
            Button(center_x - BUTTON_WIDTH // 2, start_y + 2 * (BUTTON_HEIGHT + BUTTON_MARGIN), 
                   BUTTON_WIDTH, BUTTON_HEIGHT, "Settings", self.settings),
            Button(center_x - BUTTON_WIDTH // 2, start_y + 3 * (BUTTON_HEIGHT + BUTTON_MARGIN), 
                   BUTTON_WIDTH, BUTTON_HEIGHT, "Credits", self.credits),
            Button(center_x - BUTTON_WIDTH // 2, start_y + 4 * (BUTTON_HEIGHT + BUTTON_MARGIN), 
                   BUTTON_WIDTH, BUTTON_HEIGHT, "Exit", self.exit_game)
        ]
        
        # Try to load logo
        self.logo = None
        logo_path = os.path.join("assets", "textures", "portal_logo.png")
        try:
            if os.path.exists(logo_path):
                self.logo = pygame.image.load(logo_path)
                # Scale logo to fit nicely at the top
                logo_width = min(self.logo.get_width(), SCREEN_WIDTH * 0.8)
                logo_height = self.logo.get_height() * (logo_width / self.logo.get_width())
                self.logo = pygame.transform.scale(self.logo, (int(logo_width), int(logo_height)))
        except Exception as e:
            print(f"Could not load logo: {e}")
            self.logo = None
            
        # Result of the splash screen (which button was clicked)
        self.result = None
        
    def new_game(self):
        print("Starting new game...")
        self.result = "new_game"
        self.running = False
        
    def load_game(self):
        print("Loading game...")
        self.result = "load_game"
        self.running = False
        
    def settings(self):
        print("Opening settings...")
        self.result = "settings"
        self.running = False
        
    def credits(self):
        print("Showing credits...")
        self.result = "credits"
        self.running = False
        
    def exit_game(self):
        print("Exiting game...")
        pygame.quit()
        sys.exit()
        
    def run(self):
        """Run the splash screen and return the selected option."""
        # We don't need to change display mode here - the main game will handle that
        
        while self.running:
            mouse_pos = pygame.mouse.get_pos()
            mouse_click = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        mouse_click = True
            
            # Clear screen
            self.screen.fill(BACKGROUND_COLOR)
            
            # Draw logo or title
            if self.logo:
                logo_rect = self.logo.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
                self.screen.blit(self.logo, logo_rect)
            else:
                # Draw title text if no logo
                title_text = self.font_large.render("Portal Game", True, BUTTON_TEXT_COLOR)
                title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
                self.screen.blit(title_text, title_rect)
            
            # Draw buttons
            for button in self.buttons:
                button.check_hover(mouse_pos)
                if mouse_click:
                    button.check_click(mouse_pos, mouse_click)
                button.draw(self.screen, self.font_small)
            
            pygame.display.flip()
            self.clock.tick(60)
            
        return self.result

class LoadingScreen:
    """
    A loading screen that shows progress while assets are being loaded.
    
    This screen displays a progress bar, loading text, and rotating tips
    while the game is loading assets or levels.
    """
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 24)
        self.progress = 0.0
        self.loading_text = "Loading..."
        self.tips = [
            "Tip: Use portals to solve puzzles",
            "Tip: Momentum is conserved through portals",
            "Tip: Press E to pick up objects",
            "Tip: Press F to throw held objects",
            "Tip: Press R to reset the level",
            "Tip: Press ESC to exit the game",
            "Tip: Press L to toggle enhanced lighting",
            "Tip: Press T to toggle tessellation"
        ]
        self.current_tip = self.tips[0]
        self.last_tip_change = time.time()
        self.tip_change_interval = 3.0  # Change tip every 3 seconds
        
    def update(self, progress, text=None):
        """Update loading progress (0.0 to 1.0) and optionally the loading text"""
        self.progress = min(1.0, max(0.0, progress))
        if text:
            self.loading_text = text
            
        # Change tip if needed
        current_time = time.time()
        if current_time - self.last_tip_change > self.tip_change_interval:
            self.last_tip_change = current_time
            current_index = self.tips.index(self.current_tip)
            next_index = (current_index + 1) % len(self.tips)
            self.current_tip = self.tips[next_index]
        
    def draw(self):
        """Draw the loading screen with current progress."""
        # We don't need to change display mode here - the main game will handle that
        # Just draw to the existing screen
        
        # Clear screen
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw loading text
        text_surf = self.font.render(self.loading_text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        self.screen.blit(text_surf, text_rect)
        
        # Draw loading bar background
        bar_width = SCREEN_WIDTH * 0.7
        bar_height = 30
        bar_x = (SCREEN_WIDTH - bar_width) // 2
        bar_y = SCREEN_HEIGHT // 2
        pygame.draw.rect(self.screen, LOADING_BAR_BG_COLOR, (bar_x, bar_y, bar_width, bar_height))
        
        # Draw loading bar progress
        progress_width = bar_width * self.progress
        pygame.draw.rect(self.screen, LOADING_BAR_COLOR, (bar_x, bar_y, progress_width, bar_height))
        
        # Draw progress percentage
        percent_text = self.small_font.render(f"{int(self.progress * 100)}%", True, BUTTON_TEXT_COLOR)
        percent_rect = percent_text.get_rect(center=(SCREEN_WIDTH // 2, bar_y + bar_height // 2))
        self.screen.blit(percent_text, percent_rect)
        
        # Draw tip
        tip_surf = self.small_font.render(self.current_tip, True, BUTTON_TEXT_COLOR)
        tip_rect = tip_surf.get_rect(center=(SCREEN_WIDTH // 2, bar_y + bar_height + 40))
        self.screen.blit(tip_surf, tip_rect)
        
        pygame.display.flip()
        
        # Process events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

class SettingsScreen:
    """
    A settings screen that allows the user to configure game options.
    
    This screen provides tabs for video and audio settings, with various
    controls for adjusting game parameters.
    """
    def __init__(self, screen):
        self.screen = screen
        self.running = True
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Create tabs
        self.tabs = ["Video", "Audio"]
        self.current_tab = 0
        self.tab_buttons = []
        
        tab_width = 150
        tab_height = 40
        tab_start_x = (SCREEN_WIDTH - (len(self.tabs) * tab_width)) // 2
        
        for i, tab in enumerate(self.tabs):
            self.tab_buttons.append(
                Button(
                    tab_start_x + i * tab_width, 
                    100, 
                    tab_width, 
                    tab_height, 
                    tab,
                    lambda idx=i: self.set_tab(idx)
                )
            )
        
        # Create video settings controls
        center_x = SCREEN_WIDTH // 2
        start_y = 200
        
        # Resolution dropdown
        self.resolutions = ["800x600", "1024x768", "1280x720", "1366x768", "1920x1080"]
        current_res = f"{config.screen_width}x{config.screen_height}"
        current_res_index = self.resolutions.index(current_res) if current_res in self.resolutions else 1
        
        self.resolution_dropdown = Dropdown(
            center_x, 
            start_y, 
            300, 
            40, 
            self.resolutions, 
            current_res_index,
            "Resolution",
            self.set_resolution
        )
        
        # Fullscreen checkbox
        self.fullscreen_checkbox = Checkbox(
            center_x, 
            start_y + 60, 
            30, 
            "Fullscreen", 
            config.fullscreen,
            self.set_fullscreen
        )
        
        # FOV slider
        self.fov_slider = Slider(
            center_x, 
            start_y + 120, 
            SLIDER_WIDTH, 
            SLIDER_HEIGHT, 
            30, 
            120, 
            config.fov,
            "FOV",
            self.set_fov
        )
        
        # Max FPS slider
        self.fps_slider = Slider(
            center_x, 
            start_y + 180, 
            SLIDER_WIDTH, 
            SLIDER_HEIGHT, 
            30, 
            240, 
            config.max_fps,
            "Max FPS",
            self.set_max_fps
        )
        
        # Graphics quality settings
        self.lighting_checkbox = Checkbox(
            center_x, 
            start_y + 240, 
            30, 
            "Enhanced Lighting", 
            config.enable_lighting,
            self.set_lighting
        )
        
        self.tessellation_checkbox = Checkbox(
            center_x, 
            start_y + 300, 
            30, 
            "Tessellation", 
            config.enable_tessellation,
            self.set_tessellation
        )
        
        self.tessellation_slider = Slider(
            center_x, 
            start_y + 360, 
            SLIDER_WIDTH, 
            SLIDER_HEIGHT, 
            1, 
            4, 
            config.tessellation_level,
            "Tessellation Level",
            self.set_tessellation_level
        )
        
        self.dynamic_lod_checkbox = Checkbox(
            center_x, 
            start_y + 420, 
            30, 
            "Dynamic LOD", 
            config.dynamic_lod,
            self.set_dynamic_lod
        )
        
        # Create audio settings controls
        self.master_volume_slider = Slider(
            center_x, 
            start_y + 60, 
            SLIDER_WIDTH, 
            SLIDER_HEIGHT, 
            0, 
            100, 
            int(pygame.mixer.music.get_volume() * 100) if pygame.mixer.get_init() else 50,
            "Master Volume",
            self.set_master_volume
        )
        
        self.effects_volume_slider = Slider(
            center_x, 
            start_y + 120, 
            SLIDER_WIDTH, 
            SLIDER_HEIGHT, 
            0, 
            100, 
            50,  # Default value, should be loaded from config
            "Effects Volume",
            self.set_effects_volume
        )
        
        self.music_enabled_checkbox = Checkbox(
            center_x, 
            start_y + 180, 
            30, 
            "Enable Music", 
            True,  # Default value, should be loaded from config
            self.set_music_enabled
        )
        
        self.effects_enabled_checkbox = Checkbox(
            center_x, 
            start_y + 240, 
            30, 
            "Enable Sound Effects", 
            True,  # Default value, should be loaded from config
            self.set_effects_enabled
        )
        
        # Create back button
        self.back_button = Button(
            center_x - BUTTON_WIDTH // 2, 
            SCREEN_HEIGHT - 100, 
            BUTTON_WIDTH, 
            BUTTON_HEIGHT, 
            "Back",
            self.back
        )
        
        # Create apply button
        self.apply_button = Button(
            center_x - BUTTON_WIDTH // 2, 
            SCREEN_HEIGHT - 160, 
            BUTTON_WIDTH, 
            BUTTON_HEIGHT, 
            "Apply Changes",
            self.apply_changes
        )
        
        # Track changes
        self.changes_made = False
        self.original_settings = self.get_current_settings()
        
    def get_current_settings(self):
        """Get a copy of the current settings for comparison"""
        return {
            'screen_width': config.screen_width,
            'screen_height': config.screen_height,
            'fullscreen': config.fullscreen,
            'fov': config.fov,
            'max_fps': config.max_fps,
            'enable_lighting': config.enable_lighting,
            'enable_tessellation': config.enable_tessellation,
            'tessellation_level': config.tessellation_level,
            'dynamic_lod': config.dynamic_lod
        }
        
    def set_tab(self, tab_index):
        """Switch between settings tabs"""
        self.current_tab = tab_index
        
    def set_resolution(self, index):
        """Set the screen resolution from the dropdown selection"""
        if 0 <= index < len(self.resolutions):
            resolution = self.resolutions[index]
            width, height = map(int, resolution.split('x'))
            config.screen_width = width
            config.screen_height = height
            self.changes_made = True
            
    def set_fullscreen(self, value):
        """Set fullscreen mode"""
        config.fullscreen = value
        self.changes_made = True
        
    def set_fov(self, value):
        """Set field of view"""
        config.fov = value
        self.changes_made = True
        
    def set_max_fps(self, value):
        """Set maximum FPS"""
        config.max_fps = int(value)
        self.changes_made = True
        
    def set_lighting(self, value):
        """Toggle enhanced lighting"""
        config.enable_lighting = value
        self.changes_made = True
        
    def set_tessellation(self, value):
        """Toggle tessellation"""
        config.enable_tessellation = value
        self.changes_made = True
        
    def set_tessellation_level(self, value):
        """Set tessellation level"""
        config.tessellation_level = int(value)
        self.changes_made = True
        
    def set_dynamic_lod(self, value):
        """Toggle dynamic level of detail"""
        config.dynamic_lod = value
        self.changes_made = True
        
    def set_master_volume(self, value):
        """Set master volume"""
        volume = value / 100.0
        if pygame.mixer.get_init():
            pygame.mixer.music.set_volume(volume)
        self.changes_made = True
        
    def set_effects_volume(self, value):
        """Set sound effects volume"""
        # This would normally update a config value
        # For now, just mark that changes were made
        self.changes_made = True
        
    def set_music_enabled(self, value):
        """Toggle music"""
        # This would normally update a config value
        # For now, just mark that changes were made
        self.changes_made = True
        
    def set_effects_enabled(self, value):
        """Toggle sound effects"""
        # This would normally update a config value
        # For now, just mark that changes were made
        self.changes_made = True
        
    def back(self):
        """Go back to the previous screen"""
        # Check if there are unsaved changes
        if self.changes_made:
            # In a real implementation, you might show a confirmation dialog here
            # For now, just revert changes
            self.revert_changes()
            
        self.running = False
        
    def apply_changes(self):
        """Apply the changes made to settings"""
        # In a real implementation, this might trigger a screen mode change
        # or save settings to a file
        print("Applying settings changes:")
        current_settings = self.get_current_settings()
        
        for key, value in current_settings.items():
            if self.original_settings.get(key) != value:
                print(f"  {key}: {self.original_settings.get(key)} -> {value}")
                
        # Update original settings to reflect the new state
        self.original_settings = current_settings
        self.changes_made = False
        
        # Save settings to config file (not implemented yet)
        config.save_settings()
        
    def revert_changes(self):
        """Revert any unapplied changes"""
        # Restore original settings
        config.screen_width = self.original_settings['screen_width']
        config.screen_height = self.original_settings['screen_height']
        config.fullscreen = self.original_settings['fullscreen']
        config.fov = self.original_settings['fov']
        config.max_fps = self.original_settings['max_fps']
        config.enable_lighting = self.original_settings['enable_lighting']
        config.enable_tessellation = self.original_settings['enable_tessellation']
        config.tessellation_level = self.original_settings['tessellation_level']
        config.dynamic_lod = self.original_settings['dynamic_lod']
        
        # Update UI controls to match
        self.fullscreen_checkbox.checked = config.fullscreen
        self.fov_slider.current_val = config.fov
        self.fov_slider.update_handle_position()
        self.fps_slider.current_val = config.max_fps
        self.fps_slider.update_handle_position()
        self.lighting_checkbox.checked = config.enable_lighting
        self.tessellation_checkbox.checked = config.enable_tessellation
        self.tessellation_slider.current_val = config.tessellation_level
        self.tessellation_slider.update_handle_position()
        self.dynamic_lod_checkbox.checked = config.dynamic_lod
        
        # Reset resolution dropdown
        current_res = f"{config.screen_width}x{config.screen_height}"
        if current_res in self.resolutions:
            self.resolution_dropdown.current_index = self.resolutions.index(current_res)
        
        self.changes_made = False
        
    def run(self):
        """Run the settings screen"""
        active_slider = None
        
        while self.running:
            mouse_pos = pygame.mouse.get_pos()
            mouse_click = False
            mouse_down = pygame.mouse.get_pressed()[0]
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.back()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        mouse_click = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left mouse button
                        if active_slider:
                            active_slider.release()
                            active_slider = None
            
            # Clear screen
            self.screen.fill(BACKGROUND_COLOR)
            
            # Draw title
            title_text = self.font_large.render("Settings", True, BUTTON_TEXT_COLOR)
            title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 50))
            self.screen.blit(title_text, title_rect)
            
            # Draw tabs
            for button in self.tab_buttons:
                button.check_hover(mouse_pos)
                if mouse_click:
                    button.check_click(mouse_pos, mouse_click)
                button.draw(self.screen, self.font_medium)
            
            # Draw current tab content
            if self.current_tab == 0:  # Video settings
                # Draw video settings
                self.resolution_dropdown.check_hover(mouse_pos)
                if mouse_click:
                    self.resolution_dropdown.check_click(mouse_pos, mouse_click)
                self.resolution_dropdown.draw(self.screen, self.font_small)
                
                self.fullscreen_checkbox.check_hover(mouse_pos)
                if mouse_click:
                    self.fullscreen_checkbox.check_click(mouse_pos, mouse_click)
                self.fullscreen_checkbox.draw(self.screen, self.font_small)
                
                self.fov_slider.check_hover(mouse_pos)
                if mouse_click and self.fov_slider.check_click(mouse_pos, mouse_click):
                    active_slider = self.fov_slider
                self.fov_slider.update(mouse_pos, mouse_down)
                self.fov_slider.draw(self.screen, self.font_small)
                
                self.fps_slider.check_hover(mouse_pos)
                if mouse_click and self.fps_slider.check_click(mouse_pos, mouse_click):
                    active_slider = self.fps_slider
                self.fps_slider.update(mouse_pos, mouse_down)
                self.fps_slider.draw(self.screen, self.font_small)
                
                self.lighting_checkbox.check_hover(mouse_pos)
                if mouse_click:
                    self.lighting_checkbox.check_click(mouse_pos, mouse_click)
                self.lighting_checkbox.draw(self.screen, self.font_small)
                
                self.tessellation_checkbox.check_hover(mouse_pos)
                if mouse_click:
                    self.tessellation_checkbox.check_click(mouse_pos, mouse_click)
                self.tessellation_checkbox.draw(self.screen, self.font_small)
                
                self.tessellation_slider.check_hover(mouse_pos)
                if mouse_click and self.tessellation_slider.check_click(mouse_pos, mouse_click):
                    active_slider = self.tessellation_slider
                self.tessellation_slider.update(mouse_pos, mouse_down)
                self.tessellation_slider.draw(self.screen, self.font_small)
                
                self.dynamic_lod_checkbox.check_hover(mouse_pos)
                if mouse_click:
                    self.dynamic_lod_checkbox.check_click(mouse_pos, mouse_click)
                self.dynamic_lod_checkbox.draw(self.screen, self.font_small)
                
            elif self.current_tab == 1:  # Audio settings
                # Draw audio settings
                self.master_volume_slider.check_hover(mouse_pos)
                if mouse_click and self.master_volume_slider.check_click(mouse_pos, mouse_click):
                    active_slider = self.master_volume_slider
                self.master_volume_slider.update(mouse_pos, mouse_down)
                self.master_volume_slider.draw(self.screen, self.font_small)
                
                self.effects_volume_slider.check_hover(mouse_pos)
                if mouse_click and self.effects_volume_slider.check_click(mouse_pos, mouse_click):
                    active_slider = self.effects_volume_slider
                self.effects_volume_slider.update(mouse_pos, mouse_down)
                self.effects_volume_slider.draw(self.screen, self.font_small)
                
                self.music_enabled_checkbox.check_hover(mouse_pos)
                if mouse_click:
                    self.music_enabled_checkbox.check_click(mouse_pos, mouse_click)
                self.music_enabled_checkbox.draw(self.screen, self.font_small)
                
                self.effects_enabled_checkbox.check_hover(mouse_pos)
                if mouse_click:
                    self.effects_enabled_checkbox.check_click(mouse_pos, mouse_click)
                self.effects_enabled_checkbox.draw(self.screen, self.font_small)
            
            # Draw back button
            self.back_button.check_hover(mouse_pos)
            if mouse_click:
                self.back_button.check_click(mouse_pos, mouse_click)
            self.back_button.draw(self.screen, self.font_medium)
            
            # Draw apply button
            self.apply_button.check_hover(mouse_pos)
            if mouse_click:
                self.apply_button.check_click(mouse_pos, mouse_click)
            self.apply_button.draw(self.screen, self.font_medium)
            
            pygame.display.flip()
            self.clock.tick(60)
            
        return "back"  # Return to previous screen

class CreditsScreen:
    """
    A screen that displays credits for the game.
    
    This screen shows information about the developers, assets, and
    acknowledgments for the game.
    """
    def __init__(self, screen):
        self.screen = screen
        self.running = True
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Credits content
        self.credits = [
            ("Portal Game", self.font_large),
            ("", None),
            ("Development Team", self.font_medium),
            ("Lead Developer", self.font_small),
            ("Game Designer", self.font_small),
            ("Graphics Artist", self.font_small),
            ("Sound Designer", self.font_small),
            ("", None),
            ("Technologies Used", self.font_medium),
            ("Python", self.font_small),
            ("Pygame", self.font_small),
            ("PyOpenGL", self.font_small),
            ("NumPy", self.font_small),
            ("", None),
            ("Special Thanks", self.font_medium),
            ("Valve Corporation for the original Portal concept", self.font_small),
            ("The Python community", self.font_small),
            ("All open-source contributors", self.font_small),
            ("", None),
            ("© 2023 Portal Game Team", self.font_small)
        ]
        
        # Scrolling parameters
        self.scroll_y = SCREEN_HEIGHT
        self.scroll_speed = 1
        
        # Create back button
        center_x = SCREEN_WIDTH // 2
        self.back_button = Button(
            center_x - BUTTON_WIDTH // 2, 
            SCREEN_HEIGHT - 60, 
            BUTTON_WIDTH, 
            BUTTON_HEIGHT, 
            "Back",
            self.back
        )
        
    def back(self):
        """Go back to the previous screen"""
        self.running = False
        
    def run(self):
        """Run the credits screen"""
        while self.running:
            mouse_pos = pygame.mouse.get_pos()
            mouse_click = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.back()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        mouse_click = True
            
            # Clear screen
            self.screen.fill(BACKGROUND_COLOR)
            
            # Draw scrolling credits
            y = self.scroll_y
            for text, font in self.credits:
                if not text:  # Empty line
                    y += 20
                    continue
                    
                if font:
                    text_surf = font.render(text, True, BUTTON_TEXT_COLOR)
                    text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, y))
                    if -50 < y < SCREEN_HEIGHT:  # Only draw if on screen
                        self.screen.blit(text_surf, text_rect)
                    y += text_rect.height + 10
            
            # Update scroll position
            self.scroll_y -= self.scroll_speed
            
            # Reset scroll if all credits have scrolled past
            total_height = sum(30 if text else 20 for text, font in self.credits)
            if self.scroll_y < -total_height:
                self.scroll_y = SCREEN_HEIGHT
            
            # Draw back button (fixed at bottom)
            self.back_button.check_hover(mouse_pos)
            if mouse_click:
                self.back_button.check_click(mouse_pos, mouse_click)
            self.back_button.draw(self.screen, self.font_medium)
            
            pygame.display.flip()
            self.clock.tick(60)
            
        return "back"  # Return to previous screen

# Example usage:
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Portal Game")
    
    # Show splash screen
    splash = SplashScreen(screen)
    result = splash.run()
    print(f"Splash screen result: {result}")
    
    # Show appropriate screen based on selection
    if result == "settings":
        settings = SettingsScreen(screen)
        settings.run()
    elif result == "credits":
        credits = CreditsScreen(screen)
        credits.run()
    elif result in ["new_game", "load_game"]:
        loading = LoadingScreen(screen)
        
        # Simulate loading process
        for i in range(101):
            loading.update(i / 100.0, f"Loading game assets... ({i}%)")
            loading.draw()
            pygame.time.delay(50)  # Simulate loading time
    
    pygame.quit()
    sys.exit()