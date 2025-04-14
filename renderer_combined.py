"""
Portal Game Renderer Module

This module handles all rendering functionality for the Portal game,
including portal rendering, level rendering, and special effects.
It centralizes all OpenGL code to make the main game logic cleaner.
"""

import pygame
import math
import numpy as np
from pygame.locals import *
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError as e:
    print("ERROR: OpenGL libraries could not be imported.")
    print(f"Error details: {e}")
    print("Please run the INSTALL_OPENGL_COMPLETE.bat file to fix this issue.")
    import sys
    sys.exit(1)
import time

from portal_physics import normalize, cross_product, calculate_rotation_matrix
from game_config import config

# For backward compatibility with existing code
SCREEN_WIDTH = config.screen_width
SCREEN_HEIGHT = config.screen_height
FOV = config.fov
NEAR_PLANE = config.near_plane
FAR_PLANE = config.far_plane
PORTAL_DISTANCE = config.portal_distance
PORTAL_WIDTH = config.portal_width
PORTAL_HEIGHT = config.portal_height
PORTAL_SEGMENTS = config.portal_segments
PORTAL_DEPTH_OFFSET = config.portal_depth_offset
ENABLE_LIGHTING = config.enable_lighting

# Global camera position for LOD calculations
# This is used by other modules for distance-based effects
camera_position = np.array([0, 0, 0], dtype=np.float32)

class Renderer:
    """
    Main renderer class that handles all rendering operations.
    
    This class centralizes all OpenGL rendering code, including:
    - Setting up the rendering environment
    - Portal rendering with recursive views
    - Level and object rendering
    - Special effects and post-processing
    """
    
    def __init__(self):
        """Initialize the renderer and set up OpenGL."""
        self.setup_opengl()
        
    def setup_opengl(self):
        """Set up OpenGL with the appropriate settings for portal rendering."""
        try:
            # Ensure we have a valid OpenGL context before making any OpenGL calls
            # Force a buffer swap to ensure the context is active
            pygame.display.flip()
            
            # Set up viewport
            glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
            
            # Set up projection matrix
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(FOV, SCREEN_WIDTH / SCREEN_HEIGHT, NEAR_PLANE, FAR_PLANE)
            
            # Set up modelview matrix
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Enable depth testing
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LEQUAL)
            
            # Enable backface culling
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
            
            # Set up lighting if enabled
            if ENABLE_LIGHTING:
                self.setup_lighting()
                
            print("Renderer OpenGL setup successful")
                
        except Exception as e:
            error_str = str(e)
            if "OpenGL_accelerate.errorchecker" in error_str or "line 59" in error_str:
                print("Caught OpenGL_accelerate.errorchecker error during initialization - continuing with limited functionality")
            elif "__call__" in error_str or "line 487" in error_str:
                print("Caught __call__ error during initialization - continuing with limited functionality")
            else:
                print(f"Warning: OpenGL initialization error: {e}")
                print("The game will try to continue with limited graphics functionality.")
    
    def setup_lighting(self, camera_pos=None):
        """Set up OpenGL lighting for the scene."""
        try:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    
            # Set up main light
            if camera_pos is None:
                light_position = [5.0, 5.0, 5.0, 1.0]
            else:
                # Position light near the camera for better illumination
                light_position = [camera_pos[0], camera_pos[1] + 2.0, camera_pos[2], 1.0]
                
            ambient_light = [0.3, 0.3, 0.3, 1.0]
            diffuse_light = [0.8, 0.8, 0.8, 1.0]
            specular_light = [1.0, 1.0, 1.0, 1.0]
    
            try:
                glLightfv(GL_LIGHT0, GL_POSITION, light_position)
                glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
                glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
                glLightfv(GL_LIGHT0, GL_SPECULAR, specular_light)
            except Exception as e:
                print(f"Warning: Error setting up main light: {e}")
    
            # Add a second light for better illumination
            try:
                glEnable(GL_LIGHT1)
                light_position2 = [0.0, 10.0, 0.0, 1.0]
                ambient_light2 = [0.1, 0.1, 0.1, 1.0]
                diffuse_light2 = [0.4, 0.4, 0.4, 1.0]
                specular_light2 = [0.2, 0.2, 0.2, 1.0]
        
                glLightfv(GL_LIGHT1, GL_POSITION, light_position2)
                glLightfv(GL_LIGHT1, GL_AMBIENT, ambient_light2)
                glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse_light2)
                glLightfv(GL_LIGHT1, GL_SPECULAR, specular_light2)
            except Exception as e:
                print(f"Warning: Error setting up secondary light: {e}")
    
            # Set up light attenuation for more realistic lighting
            try:
                glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0)
                glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
                glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.01)
            except Exception as e:
                print(f"Warning: Error setting up light attenuation: {e}")
                
        except Exception as e:
            error_str = str(e)
            if "OpenGL_accelerate.errorchecker" in error_str or "line 59" in error_str:
                print("Caught OpenGL_accelerate.errorchecker error during lighting setup - continuing with basic lighting")
            elif "__call__" in error_str or "line 487" in error_str:
                print("Caught __call__ error during lighting setup - continuing with basic lighting")
            else:
                print(f"Warning: Lighting setup error: {e}")
                print("The game will continue with reduced lighting effects.")
    
    def render_scene(self, game_state):
        """
        Main rendering function that renders the entire scene.
        
        Args:
            game_state: The current game state containing all objects to render
        """
        try:
            # Clear screen and buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
            glLoadIdentity()

            # Set up camera with error handling
            camera_pos = [0, 0, 0]  # Default position
            look_dir = [0, 0, -1]   # Default look direction (looking forward)
            up_vector = [0, 1, 0]   # Default up vector
            
            # Safely get camera information
            if hasattr(game_state, 'player'):
                try:
                    player_camera_pos = game_state.player.get_camera_position()
                    if player_camera_pos is not None and len(player_camera_pos) >= 3:
                        camera_pos = player_camera_pos
                        
                    player_look_dir = game_state.player.get_look_direction()
                    if player_look_dir is not None and len(player_look_dir) >= 3:
                        look_dir = player_look_dir
                        
                    player_up_vector = game_state.player.get_up_vector()
                    if player_up_vector is not None and len(player_up_vector) >= 3:
                        up_vector = player_up_vector
                except Exception as e:
                    print(f"Error getting camera data from player: {e}")

            # Store camera position globally for LOD calculations
            # This is used by other parts of the renderer and other modules for distance calculations
            global camera_position
            
            # Convert to NumPy array with float32 type for better compatibility
            camera_position = np.array(camera_pos, dtype=np.float32)

            # Set camera position and orientation
            gluLookAt(
                camera_pos[0], camera_pos[1], camera_pos[2],
                camera_pos[0] + look_dir[0], camera_pos[1] + look_dir[1], camera_pos[2] + look_dir[2],
                up_vector[0], up_vector[1], up_vector[2]
            )

            # Reset all OpenGL state for clean rendering
            glDisable(GL_LIGHTING)
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_COLOR_MATERIAL)
            glDisable(GL_BLEND)
            
            # Set up lighting for ray-tracing-like effects
            if ENABLE_LIGHTING:
                self.setup_lighting(camera_pos)

            # Handle portal rendering first - before drawing the level
            try:
                self.render_portals(game_state, camera_pos, look_dir, up_vector)
            except Exception as e:
                print(f"Error rendering portals: {e}")
            
            # Now draw the level
            try:
                self.render_level(game_state)
            except Exception as e:
                print(f"Error rendering level: {e}")
            
            # Draw HUD and UI elements
            try:
                self.render_hud(game_state)
            except Exception as e:
                print(f"Error rendering HUD: {e}")
            
            # Swap buffers to display the rendered frame
            pygame.display.flip()
            
        except Exception as e:
            print(f"Critical error in render_scene: {e}")
            import traceback
            traceback.print_exc()
    
    def render_portals(self, game_state, camera_pos, look_dir, up_vector):
        """
        Render the views through both portals.
        
        Args:
            game_state: The current game state
            camera_pos: Camera position vector
            look_dir: Camera look direction vector
            up_vector: Camera up vector
        """
        try:
            # Safely get portal references
            blue_portal = None
            orange_portal = None
            
            if hasattr(game_state, 'blue_portal'):
                blue_portal = game_state.blue_portal
                
            if hasattr(game_state, 'orange_portal'):
                orange_portal = game_state.orange_portal
            
            # Only proceed if both portals exist
            if blue_portal and orange_portal:
                # Link portals
                blue_portal.linked_portal = orange_portal
                orange_portal.linked_portal = blue_portal

                # Check if portals have the 'active' attribute and are active
                blue_active = hasattr(blue_portal, 'active') and blue_portal.active
                orange_active = hasattr(orange_portal, 'active') and orange_portal.active

                # Only attempt portal view rendering if both portals are active
                if blue_active and orange_active:
                    # Safely get portal positions
                    blue_pos = getattr(blue_portal, 'position', np.array([0, 0, 0]))
                    orange_pos = getattr(orange_portal, 'position', np.array([0, 0, 0]))
                    
                    # Calculate distance to each portal to determine rendering order
                    # Render the farther portal first for proper depth sorting
                    try:
                        blue_dist = np.linalg.norm(np.array(camera_pos) - blue_pos)
                        orange_dist = np.linalg.norm(np.array(camera_pos) - orange_pos)
                    except Exception as e:
                        print(f"Error calculating portal distances: {e}")
                        blue_dist = 0
                        orange_dist = 1  # Default to different values
                    
                    # Clear the stencil buffer before portal rendering
                    glClear(GL_STENCIL_BUFFER_BIT)
                    
                    # Ensure depth testing is properly set up
                    glEnable(GL_DEPTH_TEST)
                    glDepthFunc(GL_LEQUAL)
                    
                    # Render portals in order of distance (farther first)
                    first_portal = blue_portal if blue_dist > orange_dist else orange_portal
                    second_portal = orange_portal if blue_dist > orange_dist else blue_portal
                    
                    # Get current level safely
                    current_level = None
                    if hasattr(game_state, 'current_level'):
                        current_level = game_state.current_level
                    
                    # First render the view through the farther portal
                    if current_level:
                        try:
                            self.render_portal_view(
                                first_portal, camera_pos, look_dir, up_vector,
                                first_portal.linked_portal, current_level,
                                recursion_depth=0  # Start with recursion depth 0
                            )
                        except Exception as e:
                            print(f"Error rendering first portal view: {e}")
                            import traceback
                            traceback.print_exc()

                        # Clear the stencil buffer again for the second portal
                        glClear(GL_STENCIL_BUFFER_BIT)
                        
                        # Then render the view through the closer portal
                        try:
                            self.render_portal_view(
                                second_portal, camera_pos, look_dir, up_vector,
                                second_portal.linked_portal, current_level,
                                recursion_depth=0  # Start with recursion depth 0
                            )
                        except Exception as e:
                            print(f"Error rendering second portal view: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Reset the view after portal rendering
                    glLoadIdentity()
                    gluLookAt(
                        camera_pos[0], camera_pos[1], camera_pos[2],
                        camera_pos[0] + look_dir[0], camera_pos[1] + look_dir[1], camera_pos[2] + look_dir[2],
                        up_vector[0], up_vector[1], up_vector[2]
                    )
                    
                    # Reset OpenGL state again
                    glDisable(GL_LIGHTING)
                    glDisable(GL_TEXTURE_2D)
                    glDisable(GL_COLOR_MATERIAL)
                    glDisable(GL_BLEND)
                    
                    # Re-enable lighting if it was enabled
                    if ENABLE_LIGHTING:
                        self.setup_lighting(camera_pos)
        except Exception as e:
            print(f"Error in render_portals: {e}")
            import traceback
            traceback.print_exc()
    
    def render_portal_view(self, portal, camera_pos, look_dir, up_vector, other_portal, level, recursion_depth=0):
        """
        Render the view through a portal.
        
        Args:
            portal: The portal being looked through
            camera_pos: Camera position vector
            look_dir: Camera look direction vector
            up_vector: Camera up vector
            other_portal: The linked portal to render the view from
            level: The current level
            recursion_depth: Current recursion depth for portal-in-portal rendering
        """
        # Add recursion limit to prevent infinite recursion
        MAX_RECURSION_DEPTH = config.portal_recursion_depth
        
        if recursion_depth > MAX_RECURSION_DEPTH:
            return
            
        if not portal.active or not other_portal or not other_portal.active:
            return

        # Debug info
        print(f"Rendering portal view: {portal.color} -> {other_portal.color}, depth={recursion_depth}")
        print(f"Portal position: {portal.position}, normal: {portal.normal}")
        print(f"Other portal position: {other_portal.position}, normal: {other_portal.normal}")

        # Save current matrices and attributes
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glPushAttrib(GL_ALL_ATTRIB_BITS)  # Save all attributes to ensure proper state restoration

        try:
            # Check if stencil buffer is available
            stencil_bits = pygame.display.gl_get_attribute(pygame.GL_STENCIL_SIZE)
            if stencil_bits < 1:
                print("Warning: Stencil buffer not available for portal rendering")
                return

            # Clear only the stencil buffer (not the depth buffer yet)
            glClearStencil(0)  # Ensure stencil buffer is cleared to 0
            glClear(GL_STENCIL_BUFFER_BIT)

            # Set up stencil buffer to only render within the portal shape
            glEnable(GL_STENCIL_TEST)
            glStencilMask(0xFF)  # Enable writing to all stencil bits
            glStencilFunc(GL_ALWAYS, 1, 0xFF)  # Always pass stencil test, write 1 to stencil buffer
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)  # Replace stencil value on depth pass

            # Disable color and depth writing while creating the stencil mask
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
            glDepthMask(GL_FALSE)

            # Draw the portal shape to the stencil buffer
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # Set up the view for drawing the stencil
            gluLookAt(
                camera_pos[0], camera_pos[1], camera_pos[2],
                camera_pos[0] + look_dir[0], camera_pos[1] + look_dir[1], camera_pos[2] + look_dir[2],
                up_vector[0], up_vector[1], up_vector[2]
            )

            # Draw portal shape into stencil buffer
            glPushMatrix()
            glTranslatef(portal.position[0], portal.position[1], portal.position[2])
            
            # Use the rotation matrix calculation from portal_physics.py
            # This function now returns a 4x4 matrix in column-major order ready for OpenGL
            rotation_matrix = calculate_rotation_matrix(portal.normal)
            
            # Convert to the format expected by OpenGL if needed
            if isinstance(rotation_matrix, np.ndarray):
                # Ensure it's contiguous in memory and in the right format
                rotation_matrix = np.ascontiguousarray(rotation_matrix, dtype=np.float32)
            
            # Pass the matrix directly to OpenGL
            glMultMatrixf(rotation_matrix)

            # Make sure we're drawing front faces for proper stencil creation
            glDisable(GL_CULL_FACE)

            # Draw portal as an oval with a slightly larger size for better visibility
            glBegin(GL_TRIANGLE_FAN)
            # Center point
            glVertex3f(0, 0, PORTAL_DEPTH_OFFSET)

            # Draw portal rim
            for i in range(PORTAL_SEGMENTS + 1):
                angle = 2.0 * math.pi * i / PORTAL_SEGMENTS
                # Use different radii for width and height to create oval
                # Make the stencil mask slightly larger than the visual portal
                x = (portal.width / 2) * 1.05 * math.cos(angle)
                y = (portal.height / 2) * 1.05 * math.sin(angle)
                glVertex3f(x, y, PORTAL_DEPTH_OFFSET)
            glEnd()
            glPopMatrix()

            # Re-enable color and depth writing for the portal view
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
            glDepthMask(GL_TRUE)

            # Only render where the stencil buffer is set to 1 (the portal shape)
            glStencilFunc(GL_EQUAL, 1, 0xFF)
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)  # Don't modify the stencil buffer anymore

            # Set up the view from the other portal
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # Calculate the transformed camera position, look direction, and up vector
            # This is the core of the portal rendering effect
            try:
                transformed_camera_pos, transformed_look_dir, transformed_up = portal.get_transform(
                    other_portal, camera_pos, look_dir, up_vector
                )
                print(f"Transformed camera: pos={transformed_camera_pos}, dir={transformed_look_dir}")
            except Exception as e:
                print(f"Error in portal transform calculation: {e}")
                import traceback
                traceback.print_exc()
                # Use default values if transform fails
                transformed_camera_pos = other_portal.position + other_portal.normal * 0.1
                transformed_look_dir = other_portal.normal * -1
                transformed_up = np.array([0, 1, 0])

            # Set up the transformed view
            gluLookAt(
                transformed_camera_pos[0], transformed_camera_pos[1], transformed_camera_pos[2],
                transformed_camera_pos[0] + transformed_look_dir[0],
                transformed_camera_pos[1] + transformed_look_dir[1],
                transformed_camera_pos[2] + transformed_look_dir[2],
                transformed_up[0], transformed_up[1], transformed_up[2]
            )

            # Set up clipping plane to prevent rendering objects behind the destination portal
            # This prevents objects from "leaking" through the back of the portal
            clip_plane = np.array([
                other_portal.normal[0],
                other_portal.normal[1],
                other_portal.normal[2],
                -np.dot(other_portal.normal, other_portal.position)
            ], dtype=np.float32)
            
            glEnable(GL_CLIP_PLANE0)
            glClipPlane(GL_CLIP_PLANE0, clip_plane)

            # Set up lighting for the portal view if enabled
            if ENABLE_LIGHTING:
                self.setup_lighting(transformed_camera_pos)

            # Render the level from the transformed view
            # This is what creates the illusion of seeing through the portal
            level.draw(exclude_portal=other_portal)
            
            # Disable the clipping plane
            glDisable(GL_CLIP_PLANE0)

            # Recursively render portals within portals
            if recursion_depth < MAX_RECURSION_DEPTH:
                # Render the view through the other portal from this transformed position
                self.render_portal_view(
                    other_portal, transformed_camera_pos, transformed_look_dir, transformed_up,
                    portal, level, recursion_depth + 1
                )

        except Exception as e:
            print(f"Error in portal view rendering: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Restore previous matrices and attributes
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
            glPopAttrib()  # Restore all attributes

    def render_level(self, game_state):
        """
        Render the current level and all objects in it.
        
        Args:
            game_state: The current game state containing the level and objects
        """
        try:
            # Draw the level geometry
            if hasattr(game_state, 'current_level') and game_state.current_level:
                try:
                    game_state.current_level.draw()
                except Exception as e:
                    print(f"Error drawing level: {e}")
            
            # Draw the portals themselves (not the views through them)
            try:
                if hasattr(game_state, 'blue_portal') and game_state.blue_portal:
                    if hasattr(game_state.blue_portal, 'active') and game_state.blue_portal.active:
                        game_state.blue_portal.draw(is_blue=True)
                
                if hasattr(game_state, 'orange_portal') and game_state.orange_portal:
                    if hasattr(game_state.orange_portal, 'active') and game_state.orange_portal.active:
                        game_state.orange_portal.draw(is_blue=False)
            except Exception as e:
                print(f"Error drawing portals: {e}")
            
            # Draw all pickup blocks
            try:
                if hasattr(game_state, 'pickup_blocks'):
                    for obj in game_state.pickup_blocks:
                        if obj:  # Make sure object exists
                            obj.draw()
            except Exception as e:
                print(f"Error drawing pickup blocks: {e}")
                    
            # Draw all buttons
            try:
                if hasattr(game_state, 'buttons'):
                    for button in game_state.buttons:
                        if button:  # Make sure button exists
                            button.draw()
            except Exception as e:
                print(f"Error drawing buttons: {e}")
                    
            # Draw all doors
            try:
                if hasattr(game_state, 'current_level') and game_state.current_level:
                    if hasattr(game_state.current_level, 'doors'):
                        for door in game_state.current_level.doors:
                            if door:  # Make sure door exists
                                door.draw()
            except Exception as e:
                print(f"Error drawing doors: {e}")
                
            # Draw the player's held object if any
            try:
                if hasattr(game_state, 'player') and game_state.player:
                    if hasattr(game_state.player, 'held_object') and game_state.player.held_object:
                        game_state.player.held_object.draw()
            except Exception as e:
                print(f"Error drawing held object: {e}")
                
        except Exception as e:
            print(f"Error in render_level: {e}")
            import traceback
            traceback.print_exc()
    
    def render_hud(self, game_state):
        """
        Render the HUD (Heads-Up Display) and UI elements.
        
        Args:
            game_state: The current game state
        """
        try:
            # Switch to orthographic projection for 2D rendering
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1)
            
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            # Disable depth testing and lighting for HUD elements
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_LIGHTING)
            
            # Draw crosshair
            try:
                self.draw_crosshair()
            except Exception as e:
                print(f"Error drawing crosshair: {e}")
            
            # Draw portal gun indicators
            try:
                self.draw_portal_indicators(game_state)
            except Exception as e:
                print(f"Error drawing portal indicators: {e}")
            
            # Draw FPS counter
            try:
                self.draw_fps_counter(game_state)
            except Exception as e:
                print(f"Error drawing FPS counter: {e}")
            
            # Restore previous projection and modelview matrices
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
            
            # Re-enable depth testing
            glEnable(GL_DEPTH_TEST)
            
            # Re-enable lighting if it was enabled
            if ENABLE_LIGHTING:
                glEnable(GL_LIGHTING)
                
        except Exception as e:
            print(f"Error in render_hud: {e}")
            # Try to restore OpenGL state in case of error
            try:
                glMatrixMode(GL_PROJECTION)
                glPopMatrix()
                glMatrixMode(GL_MODELVIEW)
                glPopMatrix()
                glEnable(GL_DEPTH_TEST)
            except:
                # If we can't restore state, reset everything
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(FOV, SCREEN_WIDTH / SCREEN_HEIGHT, NEAR_PLANE, FAR_PLANE)
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                glEnable(GL_DEPTH_TEST)
    
    def draw_crosshair(self):
        """Draw a simple crosshair in the center of the screen."""
        # Draw using immediate mode for simplicity
        glColor3f(1.0, 1.0, 1.0)  # White
        glLineWidth(2.0)
        
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        size = 10
        
        glBegin(GL_LINES)
        # Horizontal line
        glVertex2f(center_x - size, center_y)
        glVertex2f(center_x + size, center_y)
        # Vertical line
        glVertex2f(center_x, center_y - size)
        glVertex2f(center_x, center_y + size)
        glEnd()
    
    def draw_portal_indicators(self, game_state):
        """
        Draw indicators showing which portal colors are available.
        
        Args:
            game_state: The current game state
        """
        try:
            # Draw blue portal indicator
            glColor3f(0.0, 0.5, 1.0)  # Blue
            glBegin(GL_QUADS)
            glVertex2f(20, 20)
            glVertex2f(50, 20)
            glVertex2f(50, 50)
            glVertex2f(20, 50)
            glEnd()
            
            # Draw orange portal indicator
            glColor3f(1.0, 0.5, 0.0)  # Orange
            glBegin(GL_QUADS)
            glVertex2f(60, 20)
            glVertex2f(90, 20)
            glVertex2f(90, 50)
            glVertex2f(60, 50)
            glEnd()
            
            # Draw X over inactive portals
            glColor3f(1.0, 0.0, 0.0)  # Red
            glLineWidth(3.0)
            
            # Safely check blue portal status
            blue_portal_active = False
            if hasattr(game_state, 'blue_portal') and game_state.blue_portal:
                if hasattr(game_state.blue_portal, 'active'):
                    blue_portal_active = game_state.blue_portal.active
                    
            if not blue_portal_active:
                glBegin(GL_LINES)
                glVertex2f(20, 20)
                glVertex2f(50, 50)
                glVertex2f(50, 20)
                glVertex2f(20, 50)
                glEnd()
                
            # Safely check orange portal status
            orange_portal_active = False
            if hasattr(game_state, 'orange_portal') and game_state.orange_portal:
                if hasattr(game_state.orange_portal, 'active'):
                    orange_portal_active = game_state.orange_portal.active
                    
            if not orange_portal_active:
                glBegin(GL_LINES)
                glVertex2f(60, 20)
                glVertex2f(90, 50)
                glVertex2f(90, 20)
                glVertex2f(60, 50)
                glEnd()
                
        except Exception as e:
            print(f"Error in draw_portal_indicators: {e}")
    
    def draw_fps_counter(self, game_state):
        """
        Draw the FPS counter in the corner of the screen.
        
        Args:
            game_state: The current game state containing FPS information
        """
        try:
            # This would normally use pygame's font rendering
            # But since we're in OpenGL mode, we'll just draw a placeholder
            # In a real implementation, you'd render text to a texture and display it
            
            # Draw FPS bar (length proportional to FPS)
            # Default to 60 FPS if not available
            fps = 60
            
            # Safely get FPS from game_state
            if hasattr(game_state, 'fps'):
                try:
                    fps_value = game_state.fps
                    if isinstance(fps_value, (int, float)) and fps_value > 0:
                        fps = fps_value
                except:
                    pass
                
            bar_length = min(100, fps)
            
            # Color based on FPS (red if low, yellow if medium, green if high)
            if fps < 30:
                glColor3f(1.0, 0.0, 0.0)  # Red
            elif fps < 50:
                glColor3f(1.0, 1.0, 0.0)  # Yellow
            else:
                glColor3f(0.0, 1.0, 0.0)  # Green
                
            glBegin(GL_QUADS)
            glVertex2f(SCREEN_WIDTH - 110, 20)
            glVertex2f(SCREEN_WIDTH - 110 + bar_length, 20)
            glVertex2f(SCREEN_WIDTH - 110 + bar_length, 30)
            glVertex2f(SCREEN_WIDTH - 110, 30)
            glEnd()
            
        except Exception as e:
            print(f"Error in draw_fps_counter: {e}")

# Create a global renderer instance
renderer = Renderer()