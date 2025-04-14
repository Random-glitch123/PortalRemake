import pygame
import sys
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
import os
from portal_physics import *  # Import all physics functions
from portal_physics import calculate_rodrigues_rotation  # Explicitly import for portal transform
from ui_system import SplashScreen, LoadingScreen, SettingsScreen, CreditsScreen  # Import UI system
from game_system import config, perf_monitor  # Import centralized configuration and performance monitoring
from asset_manager import asset_manager  # Import asset management system
# Import our new rendering system
from renderer_combined import camera_position, renderer  # Import camera position and renderer

# For backward compatibility with existing code
# These will be removed as we refactor the codebase
SCREEN_WIDTH = config.screen_width
SCREEN_HEIGHT = config.screen_height
FOV = config.fov
NEAR_PLANE = config.near_plane
FAR_PLANE = config.far_plane
PORTAL_DISTANCE = config.portal_distance
PORTAL_WIDTH = config.portal_width
PORTAL_HEIGHT = config.portal_height
PORTAL_SEGMENTS = config.portal_segments
MAX_FPS = config.max_fps
PORTAL_DEPTH_OFFSET = config.portal_depth_offset
FULLSCREEN = config.fullscreen

# Enhanced Graphics Settings
ENABLE_LIGHTING = config.enable_lighting
ENABLE_TESSELLATION = config.enable_tessellation
TESSELLATION_LEVEL = config.tessellation_level
REFLECTION_STRENGTH = config.reflection_strength
DYNAMIC_LOD = config.dynamic_lod

# Initialize Pygame and OpenGL
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Print audio format support information
def check_audio_support():
    print("\n=== Audio Format Support Information ===")
    print(f"Pygame version: {pygame.version.ver}")
    print(f"SDL version: {'.'.join(str(x) for x in pygame.get_sdl_version())}")
    print(f"Mixer initialized: {pygame.mixer.get_init() is not None}")
    
    if pygame.mixer.get_init():
        freq, fmt, channels = pygame.mixer.get_init()
        print(f"Mixer settings: {freq}Hz, Format: {fmt}, Channels: {channels}")
    
    # Try to determine MP3 support
    try:
        # Create a temporary directory for testing
        test_dir = os.path.join("assets", "test")
        os.makedirs(test_dir, exist_ok=True)
        
        # List of formats to check
        formats = [
            ("WAV", "test.wav"),
            ("OGG", "test.ogg"),
            ("MP3", "test.mp3")
        ]
        
        print("\nFormat support test:")
        for format_name, filename in formats:
            try:
                # Just check if the mixer recognizes the extension
                result = "Supported" if pygame.mixer.Sound._file_extension(filename) else "Unsupported"
            except:
                result = "Unknown"
            print(f"  {format_name}: {result}")
            
    except Exception as e:
        print(f"Error during format check: {e}")
    
    print("=========================================\n")

# Check audio support
check_audio_support()

# Set OpenGL attributes before creating the display
# Ensure we have a stencil buffer for portal rendering
pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, config.stencil_buffer_size)
pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)  # Enable multisampling for smoother edges
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)  # 4x multisampling

# Create display with stencil buffer for portal rendering
display_flags = DOUBLEBUF | OPENGL | pygame.HWSURFACE
if FULLSCREEN:
    display_flags |= pygame.FULLSCREEN
display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display_flags)
pygame.display.set_caption("Portal Remake - Enhanced Graphics")
clock = pygame.time.Clock()

# Verify stencil buffer is available
try:
    stencil_bits = pygame.display.gl_get_attribute(pygame.GL_STENCIL_SIZE)
    print(f"Stencil buffer bits: {stencil_bits} (8 bits needed for portal effects)")

    # If we couldn't get the stencil buffer we need, try again without multisampling
    if stencil_bits < 8:
        print("Warning: Could not create stencil buffer with multisampling. Trying without multisampling...")
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
        pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)  # Force 8 bits
        display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display_flags)
        stencil_bits = pygame.display.gl_get_attribute(pygame.GL_STENCIL_SIZE)
        print(f"Stencil buffer bits (retry): {stencil_bits}")
        
    # If we still don't have enough stencil bits, try a different display mode
    if stencil_bits < 8:
        print("Warning: Still could not get 8-bit stencil buffer. Trying with different display flags...")
        # Try with just the essential flags
        display_flags = DOUBLEBUF | OPENGL
        pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)  # Force 8 bits
        display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display_flags)
        stencil_bits = pygame.display.gl_get_attribute(pygame.GL_STENCIL_SIZE)
        print(f"Stencil buffer bits (final attempt): {stencil_bits}")
        
        if stencil_bits < 8:
            print("Warning: Could not get 8-bit stencil buffer. Portal effects may not work correctly.")
        else:
            print("Successfully created display with 8-bit stencil buffer.")
except Exception as e:
    print(f"Error checking stencil buffer: {e}")
    print("Continuing with default display settings.")

# Set up OpenGL with error handling
try:
    # First, ensure we have a valid OpenGL context
    # Force a buffer swap to ensure the context is active
    pygame.display.flip()
    
    # Now it's safe to call OpenGL functions
    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(FOV, SCREEN_WIDTH / SCREEN_HEIGHT, NEAR_PLANE, FAR_PLANE)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    
    print("OpenGL initialization successful")
except Exception as e:
    error_str = str(e)
    if "OpenGL_accelerate.errorchecker" in error_str or "line 59" in error_str:
        print("Caught OpenGL_accelerate.errorchecker error during initialization - continuing with limited functionality")
    elif "__call__" in error_str or "line 487" in error_str:
        print("Caught __call__ error during initialization - continuing with limited functionality")
    else:
        print(f"Warning: OpenGL initialization error: {e}")
        print("The game will try to continue with limited graphics functionality.")

# Set up lighting (simulates ray-tracing effects)
if ENABLE_LIGHTING:
    try:
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    
        # Set up main light
        light_position = [5.0, 5.0, 5.0, 1.0]
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
    
        print("Enhanced lighting enabled - simulating ray-tracing effects")
    except Exception as e:
        error_str = str(e)
        if "OpenGL_accelerate.errorchecker" in error_str or "line 59" in error_str:
            print("Caught OpenGL_accelerate.errorchecker error during lighting setup - continuing with basic lighting")
        elif "__call__" in error_str or "line 487" in error_str:
            print("Caught __call__ error during lighting setup - continuing with basic lighting")
        else:
            print(f"Warning: Lighting setup error: {e}")
            print("The game will continue with reduced lighting effects.")
else:
    print("Enhanced lighting disabled")

# Texture loading function - now uses asset_manager
def load_texture(filename):
    """
    Load a texture using the asset manager.
    
    This function is maintained for backward compatibility with existing code.
    New code should use asset_manager.load_texture() directly.
    
    Args:
        filename: Name of the texture file to load
        
    Returns:
        OpenGL texture ID
    """
    return asset_manager.load_texture(filename)

# Create a default checkerboard texture when texture files are missing
# This is now handled by the asset manager
def create_default_texture():
    """
    Create a default checkerboard texture.
    
    This function is maintained for backward compatibility with existing code.
    New code should use asset_manager._create_default_texture() directly.
    
    Returns:
        OpenGL texture ID for the default texture
    """
    return asset_manager._create_default_texture()

# Sound loading function - now uses asset_manager
def load_sound(filename):
    """
    Load a sound file using the asset manager.
    
    This function is maintained for backward compatibility with existing code.
    New code should use asset_manager.load_sound() directly.
    
    Args:
        filename: Name of the sound file to load
        
    Returns:
        pygame.mixer.Sound object or None if loading fails
    """
    return asset_manager.load_sound(filename)

# Vector operations are now imported from portal_physics.py

# Wall class
class Wall:
    """
    Represents a wall or surface in the game world.
    
    Walls can be textured, collidable, and may support portal placement.
    They are the primary building blocks of game levels.
    """
    def __init__(self, x, y, z, width, height, depth, texture_id=None, portal_surface=True):
        """
        Initialize a wall with position, dimensions, and properties.
        
        Args:
            x, y, z: Position coordinates of the wall's center
            width, height, depth: Dimensions of the wall
            texture_id: OpenGL texture ID for the wall surface
            portal_surface: Whether portals can be placed on this wall
        """
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth
        self.texture_id = texture_id
        self.portal_surface = portal_surface
        self.reflectivity = config.reflection_strength  # How reflective the surface is (0-1)
        self.shininess = 32.0  # Material shininess for specular highlights

        # Calculate vertices
        half_width = width / 2
        half_height = height / 2
        half_depth = depth / 2

        self.vertices = [
            # Front face
            [x - half_width, y - half_height, z + half_depth],
            [x + half_width, y - half_height, z + half_depth],
            [x + half_width, y + half_height, z + half_depth],
            [x - half_width, y + half_height, z + half_depth],

            # Back face
            [x - half_width, y - half_height, z - half_depth],
            [x + half_width, y - half_height, z - half_depth],
            [x + half_width, y + half_height, z - half_depth],
            [x - half_width, y + half_height, z - half_depth]
        ]

        # Calculate normals for each face
        self.normals = [
            [0, 0, 1],    # Front
            [0, 0, -1],   # Back
            [1, 0, 0],    # Right
            [-1, 0, 0],   # Left
            [0, 1, 0],    # Top
            [0, -1, 0]    # Bottom
        ]

        # Define faces (indices of vertices)
        self.faces = [
            [0, 1, 2, 3],  # Front
            [4, 7, 6, 5],  # Back
            [1, 5, 6, 2],  # Right
            [0, 3, 7, 4],  # Left
            [3, 2, 6, 7],  # Top
            [0, 4, 5, 1]   # Bottom
        ]

        # Texture coordinates
        self.tex_coords = [
            [0, 0], [1, 0], [1, 1], [0, 1]
        ]

    def draw(self):
        """
        Draw the wall with appropriate textures, lighting, and level of detail.
        
        This method handles:
        - Dynamic level of detail based on distance
        - Material properties for lighting
        - Texture mapping
        - Tessellation for more detailed geometry
        """
        # Calculate distance to camera for LOD if dynamic LOD is enabled
        if config.dynamic_lod:
            # Get camera position from the renderer_combined module
            from renderer_combined import camera_position
            
            # Calculate distance using NumPy arrays
            wall_pos = np.array([self.x, self.y, self.z], dtype=np.float32)
            distance = np.linalg.norm(wall_pos - camera_position)
            
            # Adjust tessellation level based on distance
            tess_level = max(1, int(config.tessellation_level * (1.0 - min(1.0, distance / 30.0))))
        else:
            tess_level = config.tessellation_level

        # Set up material properties for lighting/reflection
        if config.enable_lighting:
            # Set material properties
            if self.texture_id:
                # Use bright white material with texture to preserve texture colors
                ambient = [1.0, 1.0, 1.0, 1.0]
                diffuse = [1.0, 1.0, 1.0, 1.0]
                specular = [self.reflectivity, self.reflectivity, self.reflectivity, 1.0]
                
                # Enable color material to work better with textures
                glEnable(GL_COLOR_MATERIAL)
                glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
                glColor4f(1.0, 1.0, 1.0, 1.0)  # Full white to show texture properly
            else:
                # Use colored material without texture
                ambient = [0.2, 0.2, 0.2, 1.0]
                diffuse = [0.8, 0.8, 0.8, 1.0]
                specular = [self.reflectivity, self.reflectivity, self.reflectivity, 1.0]
                
                # Disable color material when not using textures
                glDisable(GL_COLOR_MATERIAL)

            glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
            glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
            glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
            glMaterialf(GL_FRONT, GL_SHININESS, self.shininess)

        # Bind texture if available
        if self.texture_id:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
        else:
            glDisable(GL_TEXTURE_2D)

        # Draw each face
        for i, face in enumerate(self.faces):
            # Apply tessellation by subdividing the face if enabled
            if config.enable_tessellation and tess_level > 1:
                self.draw_tessellated_face(i, face, tess_level)
            else:
                # Draw the face as a simple quad
                glBegin(GL_QUADS)
                glNormal3fv(self.normals[i])

                for j, vertex_idx in enumerate(face):
                    if self.texture_id:
                        glTexCoord2fv(self.tex_coords[j])

                    # Set color based on face if no texture
                    if not self.texture_id:
                        if i == 0:  # Front face
                            glColor3f(0.8, 0.8, 0.8)
                        elif i == 1:  # Back face
                            glColor3f(0.7, 0.7, 0.7)
                        elif i == 2:  # Right face
                            glColor3f(0.75, 0.75, 0.75)
                        elif i == 3:  # Left face
                            glColor3f(0.65, 0.65, 0.65)
                        elif i == 4:  # Top face
                            glColor3f(0.9, 0.9, 0.9)
                        else:  # Bottom face
                            glColor3f(0.6, 0.6, 0.6)
                    else:
                        glColor3f(1.0, 1.0, 1.0)  # White when using texture

                    glVertex3fv(self.vertices[vertex_idx])

                glEnd()

        # Disable textures
        glDisable(GL_TEXTURE_2D)

    def draw_tessellated_face(self, face_idx, face, tess_level):
        """Draw a face with tessellation for more detail"""
        # Get the four corners of the face
        v0 = self.vertices[face[0]]
        v1 = self.vertices[face[1]]
        v2 = self.vertices[face[2]]
        v3 = self.vertices[face[3]]

        # Get texture coordinates
        t0 = self.tex_coords[0]
        t1 = self.tex_coords[1]
        t2 = self.tex_coords[2]
        t3 = self.tex_coords[3]

        # Get normal
        normal = self.normals[face_idx]

        # Create a grid of points for tessellation
        glBegin(GL_QUADS)
        glNormal3fv(normal)

        for i in range(tess_level):
            for j in range(tess_level):
                # Calculate the four corners of this sub-quad
                u0 = i / tess_level
                v0 = j / tess_level
                u1 = (i + 1) / tess_level
                v1 = (j + 1) / tess_level

                # Interpolate positions
                p00 = bilinear_interpolate(v0, v1, v2, v3, u0, v0)
                p10 = bilinear_interpolate(v0, v1, v2, v3, u1, v0)
                p11 = bilinear_interpolate(v0, v1, v2, v3, u1, v1)
                p01 = bilinear_interpolate(v0, v1, v2, v3, u0, v1)

                # Interpolate texture coordinates
                tc00 = bilinear_interpolate_2d(t0, t1, t2, t3, u0, v0)
                tc10 = bilinear_interpolate_2d(t0, t1, t2, t3, u1, v0)
                tc11 = bilinear_interpolate_2d(t0, t1, t2, t3, u1, v1)
                tc01 = bilinear_interpolate_2d(t0, t1, t2, t3, u0, v1)

                # Draw the sub-quad
                if self.texture_id:
                    glTexCoord2fv(tc00)
                glVertex3fv(p00)

                if self.texture_id:
                    glTexCoord2fv(tc10)
                glVertex3fv(p10)

                if self.texture_id:
                    glTexCoord2fv(tc11)
                glVertex3fv(p11)

                if self.texture_id:
                    glTexCoord2fv(tc01)
                glVertex3fv(p01)

        glEnd()

    # Bilinear interpolation methods are now imported from portal_physics.py
    
    def check_collision(self, pos, radius):
        """
        Improved collision detection with better handling of edge cases.
        Uses AABB (Axis-Aligned Bounding Box) collision detection with a safety margin.
        
        Args:
            pos: Position to check for collision
            radius: Radius of the object (player, block, etc.)
            
        Returns:
            True if collision detected, False otherwise
        """
        # Add a small safety margin to prevent clipping through thin walls
        safety_margin = 0.05
        
        # Calculate expanded dimensions for collision detection
        half_width = self.width / 2 + radius + safety_margin
        half_height = self.height / 2 + radius + safety_margin
        half_depth = self.depth / 2 + radius + safety_margin
        
        # Check if position is within the expanded bounds
        collision = (pos[0] >= self.x - half_width and pos[0] <= self.x + half_width and
                    pos[1] >= self.y - half_height and pos[1] <= self.y + half_height and
                    pos[2] >= self.z - half_depth and pos[2] <= self.z + half_depth)
        
        # If we detect a collision, print debug info
        if collision:
            print(f"Collision detected with wall at ({self.x}, {self.y}, {self.z})")
            print(f"Player position: {pos}, Distance: {np.linalg.norm(np.array(pos) - np.array([self.x, self.y, self.z]))}")
        
        return collision
    
    def get_collision_normal(self, pos):
        # Find the closest face and return its normal
        half_width = self.width / 2
        half_height = self.height / 2
        half_depth = self.depth / 2
        
        # Calculate distances to each face
        dist_front = abs(pos[2] - (self.z + half_depth))
        dist_back = abs(pos[2] - (self.z - half_depth))
        dist_right = abs(pos[0] - (self.x + half_width))
        dist_left = abs(pos[0] - (self.x - half_width))
        dist_top = abs(pos[1] - (self.y + half_height))
        dist_bottom = abs(pos[1] - (self.y - half_height))
        
        # Find minimum distance
        min_dist = min(dist_front, dist_back, dist_right, dist_left, dist_top, dist_bottom)
        
        # Return normal of closest face
        if min_dist == dist_front:
            return [0, 0, 1]
        elif min_dist == dist_back:
            return [0, 0, -1]
        elif min_dist == dist_right:
            return [1, 0, 0]
        elif min_dist == dist_left:
            return [-1, 0, 0]
        elif min_dist == dist_top:
            return [0, 1, 0]
        else:  # Bottom
            return [0, -1, 0]
    
    def ray_intersection(self, ray_origin, ray_direction):
        # Check intersection with each face
        closest_intersection = None
        closest_distance = float('inf')
        intersection_normal = None
        
        for i, face in enumerate(self.faces):
            # Get face vertices
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]
            
            # Calculate face normal
            normal = self.normals[i]
            
            # Check if ray is parallel to face
            ndotu = np.dot(normal, ray_direction)
            if abs(ndotu) < 0.0001:  # Ray is parallel to face
                continue

            # Calculate intersection point
            w = np.array(ray_origin) - np.array(v0)
            t = -np.dot(normal, w) / ndotu
            
            if t < 0:  # Intersection is behind ray origin
                continue
            
            # Calculate intersection point
            intersection = np.array(ray_origin) + t * np.array(ray_direction)
            
            # Check if intersection is within face bounds
            # For simplicity, we'll use AABB check
            half_width = self.width / 2
            half_height = self.height / 2
            half_depth = self.depth / 2
            
            if (intersection[0] >= self.x - half_width and intersection[0] <= self.x + half_width and
                intersection[1] >= self.y - half_height and intersection[1] <= self.y + half_height and
                intersection[2] >= self.z - half_depth and intersection[2] <= self.z + half_depth):
                
                # Check if this is the closest intersection
                distance = t
                if distance < closest_distance:
                    closest_distance = distance
                    closest_intersection = intersection
                    intersection_normal = normal
        
        if closest_intersection is not None:
            return closest_intersection, intersection_normal, closest_distance
        
        return None, None, None

# Portal class
class Portal:
    def __init__(self, position, normal, color, texture_id=None):
        self.position = np.array(position)
        self.normal = np.array(normal)
        self.color = color
        self.texture_id = texture_id
        self.active = True
        
        # Use the portal dimensions from config
        # The portal height is now proportional to player height (defined in game_config.py)
        self.width = PORTAL_WIDTH
        self.height = PORTAL_HEIGHT  # This is now 110% of player height
        
        self.linked_portal = None  # Reference to the other portal for rendering through-portal view
        self.last_transport_time = 0  # To prevent immediate re-entry

        # Calculate portal orientation using the physics functions
        self.up = np.array([0, 1, 0])
        if abs(np.dot(self.normal, self.up)) > 0.99:
            # If normal is close to up vector, use a different up vector
            self.up = np.array([0, 0, 1])

        # Use the imported normalize and cross_product functions
        self.right = normalize(cross_product(self.up, self.normal))
        self.up = normalize(cross_product(self.normal, self.right))
        
    def check_collision(self, position, radius):
        """
        Check if a position is colliding with the portal.
        
        This improved version makes it easier to go through portals by:
        1. Increasing the collision detection distance
        2. Making the portal collision area larger
        3. Using a more forgiving elliptical shape
        4. Considering the object's radius in the calculation
        """
        if not self.active:
            return False
            
        # Check if position is close to portal plane
        dist_to_plane = abs(np.dot(position - self.position, self.normal))

        # Increase collision detection distance to make it easier to go through portals
        # This is especially important for jumping through portals
        max_dist = radius * 2.5  # Even more forgiving distance (was 2.0)
        if dist_to_plane > max_dist:
            return False

        # Project position onto portal plane
        projected_pos = position - np.dot(position - self.position, self.normal) * self.normal

        # Calculate distance from portal center to projected position
        offset = projected_pos - self.position

        # Check if within elliptical portal shape
        # Transform to portal's local coordinate system
        local_x = np.dot(offset, self.right)
        local_y = np.dot(offset, self.up)

        # Make the collision area significantly larger than the visual portal
        # This makes it much easier to go through portals, especially when jumping
        width_factor = 1.5  # 50% wider collision area (was 1.25)
        height_factor = 1.5  # 50% taller collision area (was 1.25)
        
        # Add the radius to the portal dimensions to make it even easier to go through
        effective_width = self.width * 0.5 * width_factor + radius * 0.8
        effective_height = self.height * 0.5 * height_factor + radius * 0.8

        # Check if point is inside ellipse: (x/a)² + (y/b)² <= 1
        # Use a more forgiving check by using 1.2 instead of 1.0
        normalized_dist = (local_x / effective_width)**2 + (local_y / effective_height)**2

        return normalized_dist <= 1.2  # More forgiving threshold (was 1.0)

    def draw(self, is_blue=None):
        if not self.active:
            return
            
        # If is_blue is provided, use it to determine color
        # Otherwise use the portal's own color attribute
        portal_color = self.color
        if is_blue is not None:
            portal_color = "blue" if is_blue else "orange"

        # Debug info
        print(f"Drawing {portal_color} portal at {self.position} with normal {self.normal}")

        # Save current state
        glPushAttrib(GL_ALL_ATTRIB_BITS)

        # Save current matrix
        glPushMatrix()

        # Translate to portal position
        glTranslatef(self.position[0], self.position[1], self.position[2])
        
        # Use the rotation matrix calculation from portal_physics.py
        # This function returns a 4x4 matrix in column-major order ready for OpenGL
        rotation_matrix = calculate_rotation_matrix(self.normal)
        
        # Convert to the format expected by OpenGL if needed
        if isinstance(rotation_matrix, np.ndarray):
            # Ensure it's contiguous in memory and in the right format
            rotation_matrix = np.ascontiguousarray(rotation_matrix, dtype=np.float32)
        
        # Pass the matrix directly to OpenGL
        glMultMatrixf(rotation_matrix)
        
        # Ensure portal dimensions are up-to-date with config
        # This allows for dynamic resizing if needed
        self.width = PORTAL_WIDTH
        self.height = PORTAL_HEIGHT  # This is now proportional to player height

        # Make sure portal is always visible on top of other geometry
        glDepthFunc(GL_LEQUAL)  # Less than or equal depth test
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)  # Pull the portal towards the camera
        
        # Disable depth writing temporarily to ensure portal is drawn on top
        glDepthMask(GL_FALSE)
        
        # Reset all rendering state for consistent portal appearance
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_CULL_FACE)  # Ensure we see both sides of the portal

        # Enable blending for glow effect
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Draw the portal interior (the "hole")
        glBegin(GL_TRIANGLE_FAN)
        
        # Set color based on portal type with glow effect
        if portal_color == "blue":
            # Blue portal interior - brighter and more vibrant
            glColor4f(0.0, 0.6, 1.0, 0.9)  # Brighter blue with higher alpha
        else:  # Orange
            # Orange portal interior - brighter and more vibrant
            glColor4f(1.0, 0.5, 0.0, 0.9)  # Brighter orange with higher alpha

        # Center point
        glVertex3f(0, 0, 0.01)  # Slight offset to avoid z-fighting

        # Draw portal as an oval shape
        for i in range(PORTAL_SEGMENTS + 1):
            angle = 2.0 * math.pi * i / PORTAL_SEGMENTS
            # Use different radii for width and height to create oval
            x = (self.width * 0.5 - 0.05) * math.cos(angle)  # Slightly smaller than the rim
            y = (self.height * 0.5 - 0.05) * math.sin(angle)
            glVertex3f(x, y, 0.01)
        glEnd()
        
        # Re-enable depth writing
        glDepthMask(GL_TRUE)

        # Draw the portal rim (the edge)
        glBegin(GL_TRIANGLE_STRIP)
        
        # Set color based on portal type - brighter and more vibrant
        if portal_color == "blue":
            # Blue portal rim
            glColor4f(0.0, 0.9, 1.0, 1.0)  # Brighter blue, fully opaque
        else:  # Orange
            # Orange portal rim
            glColor4f(1.0, 0.7, 0.0, 1.0)  # Brighter orange, fully opaque

        # Draw the rim as a triangle strip between two ovals
        for i in range(PORTAL_SEGMENTS + 1):
            angle = 2.0 * math.pi * i / PORTAL_SEGMENTS
            
            # Inner oval (slightly smaller)
            x_inner = (self.width * 0.5 - 0.05) * math.cos(angle)
            y_inner = (self.height * 0.5 - 0.05) * math.sin(angle)
            
            # Outer oval
            x_outer = self.width * 0.5 * math.cos(angle)
            y_outer = self.height * 0.5 * math.sin(angle)
            
            # Add vertices for triangle strip
            glVertex3f(x_inner, y_inner, 0.02)  # Inner point
            glVertex3f(x_outer, y_outer, 0.02)  # Outer point
        glEnd()

        # Draw a bright outline to make the portal more visible
        glLineWidth(4.0)  # Thicker line for better visibility
        glBegin(GL_LINE_LOOP)

        # Set a brighter color for the outline
        if portal_color == "blue":
            glColor4f(0.7, 1.0, 1.0, 1.0)  # Even brighter blue
        else:
            glColor4f(1.0, 0.9, 0.4, 1.0)  # Even brighter orange

        for i in range(PORTAL_SEGMENTS):
            angle = 2.0 * math.pi * i / PORTAL_SEGMENTS
            x = self.width * 0.5 * math.cos(angle)
            y = self.height * 0.5 * math.sin(angle)
            glVertex3f(x, y, 0.03)  # Slightly in front
        glEnd()
        
        # Add a more pronounced glow effect
        glLineWidth(2.0)  # Thicker lines for better visibility
        for j in range(5):  # Add one more ring for better glow
            glBegin(GL_LINE_LOOP)
            
            # Fade the color for outer glows
            alpha = 0.9 - j * 0.18  # Higher starting alpha, same fade rate
            if portal_color == "blue":
                glColor4f(0.7, 1.0, 1.0, alpha)  # Brighter blue with fading alpha
            else:
                glColor4f(1.0, 0.9, 0.4, alpha)  # Brighter orange with fading alpha
                
            scale = 1.0 + (j+1) * 0.08  # Slightly larger rings for better visibility
            
            for i in range(PORTAL_SEGMENTS):
                angle = 2.0 * math.pi * i / PORTAL_SEGMENTS
                x = self.width * 0.5 * scale * math.cos(angle)
                y = self.height * 0.5 * scale * math.sin(angle)
                glVertex3f(x, y, 0.03)
            glEnd()

        # Restore matrix
        glPopMatrix()

        # Restore previous state
        glPopAttrib()

    # The draw_portal_view method has been moved to the renderer module
    # This comment is kept here to document the change

    def check_collision(self, pos, radius):
        # Use the collision detection function from portal_physics.py
        return check_portal_collision(self, pos, radius)

    def get_transform(self, other_portal, camera_pos, look_dir, up_vector):
        """
        Calculate the transformed camera position, look direction, and up vector
        when looking through this portal to the other portal.
        
        Args:
            other_portal: The destination portal
            camera_pos: Original camera position
            look_dir: Original look direction
            up_vector: Original up vector
            
        Returns:
            Tuple of (transformed_position, transformed_look_dir, transformed_up)
        """
        try:
            # Ensure all inputs are numpy arrays
            camera_pos = np.array(camera_pos, dtype=np.float32)
            look_dir = np.array(look_dir, dtype=np.float32)
            up_vector = np.array(up_vector, dtype=np.float32)
            
            # Calculate offset from entry portal
            offset = camera_pos - self.position
            
            # Calculate distance to portal plane
            dist_to_plane = np.dot(offset, self.normal)
            
            # Project offset onto portal plane
            projected_offset = offset - dist_to_plane * self.normal
            
            # Transform to portal's local coordinate system
            local_x = np.dot(projected_offset, self.right)
            local_y = np.dot(projected_offset, self.up)
            
            # Scale coordinates to account for different portal dimensions
            # Add safety check to prevent division by zero
            width_half = max(0.001, self.width * 0.5)
            height_half = max(0.001, self.height * 0.5)
            
            normalized_x = local_x / width_half
            normalized_y = local_y / height_half
            
            # Apply the normalized coordinates to the exit portal's dimensions
            exit_x = normalized_x * (other_portal.width * 0.5)
            exit_y = normalized_y * (other_portal.height * 0.5)
            
            # Calculate the exit offset in the exit portal's coordinate system
            exit_offset = exit_x * other_portal.right + exit_y * other_portal.up
            
            # Calculate the transformed position
            # Flip the distance to the plane to maintain the same relative distance
            transformed_pos = other_portal.position + exit_offset - dist_to_plane * other_portal.normal
            
            # Calculate the rotation between the two portals
            # This is the key to making the view through portals work correctly
            entry_normal = self.normal
            exit_normal = other_portal.normal
            
            # Calculate the rotation matrix between the two portals
            dot_product_portals = np.dot(entry_normal, exit_normal)
            cross_product_portals = np.cross(entry_normal, exit_normal)
            cross_magnitude = np.linalg.norm(cross_product_portals)
            
            # Transform the look direction and up vector
            if cross_magnitude > 0.001:  # If portals aren't parallel
                # Normalize the cross product
                cross_product_portals = cross_product_portals / cross_magnitude
                
                # Calculate rotation angle
                angle = math.acos(max(-1.0, min(1.0, dot_product_portals)))
                
                # Apply rotation to look direction and up vector using Rodrigues formula
                transformed_look_dir = calculate_rodrigues_rotation(look_dir, cross_product_portals, angle)
                transformed_up = calculate_rodrigues_rotation(up_vector, cross_product_portals, angle)
            else:
                # If portals are parallel, just reflect the vectors
                transformed_look_dir = look_dir - 2 * np.dot(look_dir, entry_normal) * entry_normal
                transformed_up = up_vector - 2 * np.dot(up_vector, entry_normal) * entry_normal
            
            # Ensure all outputs are valid
            if np.isnan(transformed_pos).any() or np.isnan(transformed_look_dir).any() or np.isnan(transformed_up).any():
                print("Warning: NaN values detected in portal transform. Using fallback values.")
                transformed_pos = other_portal.position + other_portal.normal * 0.1
                transformed_look_dir = other_portal.normal * -1
                transformed_up = np.array([0, 1, 0])
            
            return transformed_pos, transformed_look_dir, transformed_up
            
        except Exception as e:
            print(f"Error in portal transform calculation: {e}")
            # Return safe default values
            return (other_portal.position + other_portal.normal * 0.1, 
                    other_portal.normal * -1, 
                    np.array([0, 1, 0]))
        
    def check_overlap(self, other_portal):
        """Check if this portal overlaps with another portal"""
        if other_portal is None:
            return False

        # If portals are on different surfaces (different normals), they can't overlap
        # Use a stricter comparison to ensure portals on slightly different surfaces don't overlap
        if not np.allclose(self.normal, other_portal.normal, rtol=0.05, atol=0.05):
            # Check if portals are on the same wall but with different orientations
            # Calculate distance between portal centers
            center_dist = np.linalg.norm(self.position - other_portal.position)
            
            # If portals are very close but on different planes, still consider it an overlap
            if center_dist < max(self.width, self.height) * 0.75:
                return True
            return False

        # Calculate distance between portal centers
        center_dist = np.linalg.norm(self.position - other_portal.position)

        # Calculate minimum distance needed to avoid overlap
        # Use a more conservative approach to prevent portals from being too close
        min_dist = max(self.width, self.height) * 0.75 + max(other_portal.width, other_portal.height) * 0.75

        # If centers are closer than min_dist, portals overlap
        return center_dist < min_dist

# Door class for level completion
class Door:
    def __init__(self, x, y, z, width, height, depth, texture_id=None):
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth
        self.texture_id = texture_id
        self.is_open = False
        self.open_amount = 0.0  # 0.0 = closed, 1.0 = fully open
        self.open_speed = 3.0  # Increased from 1.0 - Speed at which the door opens
        self.required_buttons = []  # List of buttons that need to be pressed to open the door
        self.player_entered = False  # Track if player has entered the door
        self.force_close = False     # Flag to force door to close after player enters
        self.player_passed_through = False  # Track if player has fully passed through
        self.last_button_press_time = 0  # Track when buttons were last pressed
        self.stay_open_time = 10.0  # Time in seconds to keep door open after button is released
        
        # Calculate vertices
        half_width = width / 2
        half_height = height / 2
        half_depth = depth / 2
        
        self.vertices = [
            # Front face
            [x - half_width, y - half_height, z + half_depth],
            [x + half_width, y - half_height, z + half_depth],
            [x + half_width, y + half_height, z + half_depth],
            [x - half_width, y + half_height, z + half_depth],
            
            # Back face
            [x - half_width, y - half_height, z - half_depth],
            [x + half_width, y - half_height, z - half_depth],
            [x + half_width, y + half_height, z - half_depth],
            [x - half_width, y + half_height, z - half_depth]
        ]
        
        # Original y position for animation
        self.original_y = y
        
        # Door area for detecting player entry
        self.door_area = {
            'min_x': x - width/2,
            'max_x': x + width/2,
            'min_z': z - depth/2,
            'max_z': z + depth/2,
            'min_y': y - height/2,
            'max_y': y + height/2
        }
        
        # Direction vector (normalized) pointing from front to back of door
        # This helps determine when player has passed through
        self.direction = normalize(np.array([0, 0, -1]))  # Default direction is negative Z
    
    def draw(self):
        # Save current state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glPushMatrix()
        
        # Calculate current door position based on open amount
        current_y = self.original_y
        if self.is_open and not self.force_close:
            # Move the door up when open
            current_y = self.original_y + self.height * self.open_amount
        
        # Translate to door position with current y
        glTranslatef(self.x, current_y, self.z)
        
        # Bind texture if available
        if self.texture_id:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # Draw the door
        glBegin(GL_QUADS)
        
        # Set color
        if self.texture_id:
            glColor3f(1.0, 1.0, 1.0)  # White for textured objects
        else:
            if self.is_open and not self.force_close:
                glColor3f(0.0, 0.8, 0.0)  # Green when open
            else:
                glColor3f(0.8, 0.0, 0.0)  # Red when closed
        
        # Define vertices for a cube
        half_width = self.width / 2
        half_height = self.height / 2
        half_depth = self.depth / 2
        
        # Define texture coordinates
        tex_coords = [
            [0, 0], [1, 0], [1, 1], [0, 1]
        ]
        
        # Front face
        if self.texture_id:
            glTexCoord2fv(tex_coords[0])
        glVertex3f(-half_width, -half_height, half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[1])
        glVertex3f(half_width, -half_height, half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[2])
        glVertex3f(half_width, half_height, half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[3])
        glVertex3f(-half_width, half_height, half_depth)
        
        # Back face
        if self.texture_id:
            glTexCoord2fv(tex_coords[0])
        glVertex3f(-half_width, -half_height, -half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[1])
        glVertex3f(half_width, -half_height, -half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[2])
        glVertex3f(half_width, half_height, -half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[3])
        glVertex3f(-half_width, half_height, -half_depth)
        
        # Right face
        if self.texture_id:
            glTexCoord2fv(tex_coords[0])
        glVertex3f(half_width, -half_height, -half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[1])
        glVertex3f(half_width, -half_height, half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[2])
        glVertex3f(half_width, half_height, half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[3])
        glVertex3f(half_width, half_height, -half_depth)
        
        # Left face
        if self.texture_id:
            glTexCoord2fv(tex_coords[0])
        glVertex3f(-half_width, -half_height, -half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[1])
        glVertex3f(-half_width, -half_height, half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[2])
        glVertex3f(-half_width, half_height, half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[3])
        glVertex3f(-half_width, half_height, -half_depth)
        
        # Top face
        if self.texture_id:
            glTexCoord2fv(tex_coords[0])
        glVertex3f(-half_width, half_height, -half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[1])
        glVertex3f(half_width, half_height, -half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[2])
        glVertex3f(half_width, half_height, half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[3])
        glVertex3f(-half_width, half_height, half_depth)
        
        # Bottom face
        if self.texture_id:
            glTexCoord2fv(tex_coords[0])
        glVertex3f(-half_width, -half_height, -half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[1])
        glVertex3f(half_width, -half_height, -half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[2])
        glVertex3f(half_width, -half_height, half_depth)
        if self.texture_id:
            glTexCoord2fv(tex_coords[3])
        glVertex3f(-half_width, -half_height, half_depth)
        
        glEnd()
        
        # Disable textures
        glDisable(GL_TEXTURE_2D)
        
        # Restore state
        glPopMatrix()
        glPopAttrib()
    
    def update(self, dt):
        # Check if any required buttons are pressed (changed from all to any for easier gameplay)
        # If there are no required buttons, the door will always open
        current_time = time.time()
        
        if self.required_buttons:
            any_button_pressed = any(button.pressed for button in self.required_buttons)
            
            # If any button is pressed, update the last press time
            if any_button_pressed:
                self.last_button_press_time = current_time
                
            # Keep the door open for stay_open_time seconds after the last button press
            if not any_button_pressed and (current_time - self.last_button_press_time) < self.stay_open_time:
                any_button_pressed = True  # Pretend buttons are still pressed
        else:
            any_button_pressed = True  # If no buttons are required, door should open
        
        # Update door state based on buttons and force_close flag
        if any_button_pressed and not self.is_open and not self.force_close:
            self.is_open = True
            print("Door is opening!")
            # Play door open sound here if available
        elif (not any_button_pressed or self.force_close) and self.is_open:
            self.is_open = False
            print("Door is closing!")
            # Play door close sound here if available
        
        # Animate door opening/closing - dramatically increased speed for better gameplay
        if self.is_open and not self.force_close:
            # Instantly open the door for better gameplay
            self.open_amount = 1.0  # Instantly fully open
            print("Door is fully open!")
        else:
            # Very slow closing for better gameplay
            self.open_amount = max(0.0, self.open_amount - self.open_speed * dt * 0.5)  # Half speed for closing
            if self.open_amount < 0.01 and self.open_amount > 0:
                print("Door is now fully closed")
                self.open_amount = 0.0
            
        # Debug info
        if self.required_buttons:
            button_states = [f"Button {i}: {'Pressed' if button.pressed else 'Not Pressed'}" 
                            for i, button in enumerate(self.required_buttons)]
            print(f"Door state: {'Open' if self.is_open else 'Closed'}, Open amount: {self.open_amount:.2f}")
            print(f"Button states: {', '.join(button_states)}")
    
    def check_collision(self, pos, radius):
        """
        Improved door collision detection with better handling of partially open doors.
        Makes it easier to pass through doors for better gameplay.
        
        Args:
            pos: Position to check for collision
            radius: Radius of the object (player, block, etc.)
            
        Returns:
            True if collision detected, False otherwise
        """
        # If the door is even partially open, allow passage
        # Lowered threshold from 0.5 to 0.3 to make it much easier to pass through
        if self.open_amount >= 0.3 and not self.force_close:
            return False
        
        # Calculate current door position based on open amount
        current_y = self.original_y + self.height * self.open_amount
        
        # Add a small safety margin to prevent clipping
        safety_margin = 0.05
        
        # Calculate expanded dimensions for collision detection
        half_width = self.width / 2 + radius + safety_margin
        half_height = self.height / 2 + radius + safety_margin
        half_depth = self.depth / 2 + radius + safety_margin
        
        # Check if position is within the expanded bounds
        # For partially open doors, we need to check if the player is below the current door position
        if pos[1] < current_y - half_height * 0.7:  # Further reduced height check to make it easier to pass under
            # Player is below the door, so they can pass through
            return False
        
        # Reduce the collision width and depth significantly to make it easier to pass through
        reduced_half_width = half_width * 0.8  # Reduced from 0.9
        reduced_half_depth = half_depth * 0.8  # Reduced from 0.9
        
        # Otherwise, check standard AABB collision with reduced dimensions
        collision = (pos[0] >= self.x - reduced_half_width and pos[0] <= self.x + reduced_half_width and
                    pos[1] >= current_y - half_height and pos[1] <= current_y + half_height and
                    pos[2] >= self.z - reduced_half_depth and pos[2] <= self.z + reduced_half_depth)
        
        # If we detect a collision, print debug info
        if collision:
            print(f"Door collision detected at ({self.x}, {current_y}, {self.z})")
            print(f"Door open amount: {self.open_amount}, Player position: {pos}")
        
        return collision
    
    def check_player_entry(self, player_pos, player_radius):
        """Check if player has entered the door area"""
        # Only check if door is open and player hasn't already entered
        if not self.is_open or self.player_entered or self.force_close:
            return False
            
        # Check if player is within the door area
        in_door_area = (
            player_pos[0] >= self.door_area['min_x'] - player_radius and 
            player_pos[0] <= self.door_area['max_x'] + player_radius and
            player_pos[1] >= self.door_area['min_y'] - player_radius and 
            player_pos[1] <= self.door_area['max_y'] + player_radius and
            player_pos[2] >= self.door_area['min_z'] - player_radius and 
            player_pos[2] <= self.door_area['max_z'] + player_radius
        )
        
        if in_door_area and not self.player_entered:
            self.player_entered = True
            print("Player entered the door!")
            return True
            
        return False
        
    def check_player_passed_through(self, player_pos, player_prev_pos):
        """Check if player has passed through the door completely"""
        if not self.player_entered or self.player_passed_through:
            return False
            
        # Calculate the vector from door center to player
        door_center = np.array([self.x, self.original_y, self.z])
        to_player = player_pos - door_center
        
        # Calculate the vector from previous position to current position
        movement_vector = player_pos - player_prev_pos
        
        # Check if player has moved past the door in the door's direction
        # This is a simplified check - we're seeing if the player has moved
        # in the same general direction as the door's orientation
        if np.linalg.norm(movement_vector) > 0.01:  # Only if player has moved significantly
            movement_dir = normalize(movement_vector)
            # If player is moving in roughly the same direction as the door faces
            if np.dot(movement_dir, self.direction) > 0.5:
                # Check if player is now on the other side of the door
                if np.dot(to_player, self.direction) > 0:
                    self.player_passed_through = True
                    self.force_close = True
                    print("Player passed through the door! Closing door.")
                    return True
                    
        return False

# Level class
# ExitCube class - a green cube that triggers level transition when touched
class ExitCube:
    def __init__(self, x, y, z, size=1.0):
        self.position = [x, y, z]
        self.size = size
        self.color = (0, 1, 0)  # Bright green
        self.rotation = 0
        self.touched = False
        self.transition_timer = 0
        self.transition_delay = 1.5  # Time before level transition
        self.pulse_speed = 3.0
        
    def draw(self):
        # Save current state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glPushMatrix()
        
        # Position the cube
        glTranslatef(self.position[0], self.position[1], self.position[2])
        
        # Rotate the cube for visual interest
        self.rotation += 0.5  # Rotate slowly
        glRotatef(self.rotation, 0, 1, 0)
        
        # Make the cube pulsate
        pulse = 0.5 + 0.5 * math.sin(time.time() * self.pulse_speed)
        scale_factor = 1.0 + 0.1 * pulse
        glScalef(scale_factor, scale_factor, scale_factor)
        
        # Set color with pulsating glow
        if self.touched:
            # Brighter glow when touched
            glColor4f(0.0, 1.0, 0.5, 0.7 + 0.3 * pulse)
        else:
            glColor4f(0.0, 1.0, 0.0, 0.5 + 0.5 * pulse)
        
        # Enable blending for glow effect
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        # Draw outer glow
        half_size = self.size * 0.55
        glBegin(GL_QUADS)
        
        # Front face
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        
        # Back face
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(-half_size, half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        glVertex3f(half_size, -half_size, -half_size)
        
        # Top face
        glVertex3f(-half_size, half_size, -half_size)
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(half_size, half_size, -half_size)
        
        # Bottom face
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(-half_size, -half_size, half_size)
        
        # Right face
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(half_size, -half_size, half_size)
        
        # Left face
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, -half_size)
        
        glEnd()
        
        # Draw solid cube inside
        if self.touched:
            glColor3f(0.0, 1.0, 0.8)  # Brighter when touched
        else:
            glColor3f(0.0, 0.8, 0.0)  # Normal green
            
        half_size = self.size * 0.5
        glBegin(GL_QUADS)
        
        # Front face
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        
        # Back face
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(-half_size, half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        glVertex3f(half_size, -half_size, -half_size)
        
        # Top face
        glVertex3f(-half_size, half_size, -half_size)
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(half_size, half_size, -half_size)
        
        # Bottom face
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(-half_size, -half_size, half_size)
        
        # Right face
        glVertex3f(half_size, -half_size, -half_size)
        glVertex3f(half_size, half_size, -half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(half_size, -half_size, half_size)
        
        # Left face
        glVertex3f(-half_size, -half_size, -half_size)
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, -half_size)
        
        glEnd()
        
        # Reset blend mode
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)
        
        # Restore state
        glPopMatrix()
        glPopAttrib()
        
    def check_collision(self, pos, radius):
        """Check if player is touching the exit cube"""
        # Calculate distance from player to cube center
        dist = np.linalg.norm(np.array(pos) - np.array(self.position))
        
        # Check if player is touching the cube (with some margin)
        collision = dist <= (radius + self.size * 0.7)
        
        if collision and not self.touched:
            self.touched = True
            print("Exit cube touched! Preparing level transition...")
            
        return collision
        
    def update(self, dt):
        """Update the exit cube state"""
        if self.touched:
            self.transition_timer += dt
            self.pulse_speed = 6.0  # Faster pulsing when touched
            
            # Increase size slightly when touched
            if self.size < 1.2:
                self.size += dt * 0.2
                
        return self.touched and self.transition_timer >= self.transition_delay

# ExitHallway class - a hallway that connects levels with a door that closes behind the player
class ExitHallway:
    def __init__(self, x, y, z, direction=[0, 0, -1], length=8.0, width=3.0, height=3.0):
        # Determine which wall the hallway should be embedded in
        # We'll use the direction to determine which wall to embed in
        self.direction = normalize(np.array(direction))
        
        # Calculate the perpendicular vectors for width
        self.right = normalize(np.cross(np.array([0, 1, 0]), self.direction))
        
        # Adjust the position to be embedded in the wall
        # Move the entry point back so it's flush with the wall
        wall_thickness = 0.5  # Estimated wall thickness
        
        # The position is now the entry point, which is slightly recessed into the wall
        self.position = [
            x - self.direction[0] * wall_thickness,
            y,
            z - self.direction[2] * wall_thickness
        ]
        
        # Store parameters
        self.length = length
        self.width = width
        self.height = height
        self.entered = False
        self.door_closed = False
        self.door_position = 0.0  # 0.0 = open, 1.0 = closed
        self.door_close_speed = 1.5
        
        # Create the exit cube at the end of the hallway
        self.exit_cube = ExitCube(
            self.position[0] + self.direction[0] * (length - 2.0),
            self.position[1] + height/2,
            self.position[2] + self.direction[2] * (length - 2.0),
            size=1.0
        )
        
        # Entry point is at the start of the hallway
        self.entry_point = np.array(self.position)
        
        # Exit point is at the end of the hallway
        self.exit_point = np.array([
            self.position[0] + self.direction[0] * length,
            self.position[1],
            self.position[2] + self.direction[2] * length
        ])
        
    def draw(self):
        # Save current state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        
        # Draw the hallway walls
        half_width = self.width / 2
        half_height = self.height / 2
        
        # Calculate corner points for the hallway
        right_vec = self.right * half_width
        up_vec = np.array([0, half_height, 0])
        dir_vec = self.direction * self.length
        
        # Start position
        start = np.array(self.position)
        
        # Draw the hallway walls
        glColor3f(0.7, 0.7, 0.7)  # Light gray walls
        
        # Floor
        glBegin(GL_QUADS)
        glVertex3f(*(start - right_vec - up_vec))
        glVertex3f(*(start + right_vec - up_vec))
        glVertex3f(*(start + right_vec - up_vec + dir_vec))
        glVertex3f(*(start - right_vec - up_vec + dir_vec))
        glEnd()
        
        # Ceiling
        glBegin(GL_QUADS)
        glVertex3f(*(start - right_vec + up_vec))
        glVertex3f(*(start - right_vec + up_vec + dir_vec))
        glVertex3f(*(start + right_vec + up_vec + dir_vec))
        glVertex3f(*(start + right_vec + up_vec))
        glEnd()
        
        # Left wall
        glBegin(GL_QUADS)
        glVertex3f(*(start - right_vec - up_vec))
        glVertex3f(*(start - right_vec - up_vec + dir_vec))
        glVertex3f(*(start - right_vec + up_vec + dir_vec))
        glVertex3f(*(start - right_vec + up_vec))
        glEnd()
        
        # Right wall
        glBegin(GL_QUADS)
        glVertex3f(*(start + right_vec - up_vec))
        glVertex3f(*(start + right_vec + up_vec))
        glVertex3f(*(start + right_vec + up_vec + dir_vec))
        glVertex3f(*(start + right_vec - up_vec + dir_vec))
        glEnd()
        
        # End wall
        glBegin(GL_QUADS)
        glVertex3f(*(start - right_vec - up_vec + dir_vec))
        glVertex3f(*(start + right_vec - up_vec + dir_vec))
        glVertex3f(*(start + right_vec + up_vec + dir_vec))
        glVertex3f(*(start - right_vec + up_vec + dir_vec))
        glEnd()
        
        # Draw the entry door if it's closing or closed
        if self.door_position > 0.0:
            # Calculate door position
            door_height = self.height * self.door_position
            
            # Draw the door (red)
            glColor3f(0.8, 0.0, 0.0)
            glBegin(GL_QUADS)
            glVertex3f(*(start - right_vec - up_vec + up_vec * (1 - self.door_position)))
            glVertex3f(*(start + right_vec - up_vec + up_vec * (1 - self.door_position)))
            glVertex3f(*(start + right_vec + up_vec))
            glVertex3f(*(start - right_vec + up_vec))
            glEnd()
        
        # Draw the exit cube at the end of the hallway
        self.exit_cube.draw()
        
        # Restore state
        glPopAttrib()
        
    def check_entry(self, pos, radius):
        """Check if player has entered the hallway"""
        # Calculate distance from player to entry point
        entry_dist = np.linalg.norm(np.array(pos) - self.entry_point)
        
        # Check if player is past the entry point (in the direction of the hallway)
        player_to_entry = np.array(pos) - self.entry_point
        past_entry = np.dot(player_to_entry, self.direction) > 0
        
        # Check if player is within the width of the hallway
        within_width = abs(np.dot(player_to_entry, self.right)) < (self.width/2 - radius)
        
        if past_entry and within_width and entry_dist < self.length and not self.entered:
            self.entered = True
            print("Player entered the exit hallway!")
            return True
            
        return False
        
    def update(self, dt, player_pos):
        """Update the hallway state"""
        # Check if player has entered and update door state
        if self.entered and not self.door_closed:
            # Calculate how far the player is into the hallway
            player_to_entry = np.array(player_pos) - self.entry_point
            distance_in = np.dot(player_to_entry, self.direction)
            
            # Start closing the door when player is a bit into the hallway
            if distance_in > self.width:
                self.door_position += dt * self.door_close_speed
                if self.door_position >= 1.0:
                    self.door_position = 1.0
                    self.door_closed = True
                    print("Exit hallway door closed!")
        
        # Update the exit cube
        return self.exit_cube.update(dt)
        
    def check_exit_cube_collision(self, pos, radius):
        """Check if player is touching the exit cube"""
        return self.exit_cube.check_collision(pos, radius)

class Level:
    def __init__(self, walls, goal_pos, player_start, pickup_blocks=None, buttons=None, doors=None):
        self.walls = walls
        self.player_start = player_start
        self.pickup_blocks = pickup_blocks if pickup_blocks is not None else []
        self.buttons = buttons if buttons is not None else []
        self.doors = doors if doors is not None else []
        self.exit_door = None  # The door that leads to the goal
        self.player_entered_exit = False  # Track if player has entered the exit door
        self.level_transition_timer = 0  # Timer for automatic level transition
        self.level_transition_delay = 2.0  # Seconds to wait before transitioning
        self.ready_for_next_level = False  # Flag to indicate level is ready to advance
        self.player_prev_pos = None  # Previous player position for tracking movement
        
        # Determine the best direction for the exit hallway based on goal position
        # This will make the hallway embedded in the nearest wall
        
        # Find the nearest wall to embed the hallway in
        nearest_wall = None
        min_distance = float('inf')
        hallway_direction = [0, 0, -1]  # Default direction
        
        for wall in walls:
            # Check if this is a vertical wall (y-axis aligned)
            if abs(wall.width) > abs(wall.height) and abs(wall.depth) > abs(wall.height):
                # This is a vertical wall, check distance to goal
                wall_pos = np.array([wall.x, wall.y, wall.z])
                goal_pos_array = np.array(goal_pos)
                
                # Calculate distance to wall
                distance = np.linalg.norm(wall_pos - goal_pos_array)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_wall = wall
                    
                    # Calculate direction from wall to goal
                    direction = goal_pos_array - wall_pos
                    if np.linalg.norm(direction) > 0:
                        hallway_direction = -normalize(direction)  # Point into the wall
        
        # If no suitable wall found, use default position and direction
        if nearest_wall is None:
            # Create the exit hallway at the goal position with default direction
            self.exit_hallway = ExitHallway(
                goal_pos[0], goal_pos[1] - 1.5, goal_pos[2],  # Position at ground level
                direction=[0, 0, -1],  # Default direction
                length=8.0,
                width=3.0,
                height=3.0
            )
        else:
            # Create the exit hallway embedded in the nearest wall
            # Position it at the same height as the player start for consistency
            self.exit_hallway = ExitHallway(
                goal_pos[0], player_start[1], goal_pos[2],  # Use player's y-height
                direction=hallway_direction,
                length=8.0,
                width=3.0,
                height=3.0
            )
    
    def draw(self, exclude_portal=None):
        """
        Draw the level and all its components.
        
        Args:
            exclude_portal: Optional portal to exclude from drawing (used when rendering through portals)
        """
        # Draw walls
        for wall in self.walls:
            wall.draw()
        
        # Draw buttons
        for button in self.buttons:
            button.draw()
            
        # Draw doors
        for door in self.doors:
            door.draw()
            
        # Draw pickup blocks
        for block in self.pickup_blocks:
            block.draw()
        
        # Draw the exit hallway with the green exit cube
        self.exit_hallway.draw()
    
    def check_wall_collisions(self, pos, radius):
        """
        Check for collisions with walls, doors, and pickup blocks.
        
        Args:
            pos: Position to check for collision
            radius: Radius of the object (player, block, etc.)
            
        Returns:
            Tuple of (collision_found, normal)
            collision_found: True if collision detected, False otherwise
            normal: Normalized vector pointing away from the collision surface
        """
        try:
            # Check collisions with walls
            collision_found = False
            final_normal = np.array([0.0, 0.0, 0.0])
            
            # Check all walls and accumulate collision normals
            for wall in self.walls:
                if wall.check_collision(pos, radius):
                    normal = wall.get_collision_normal(pos)
                    if normal is not None:
                        collision_found = True
                        # Add this normal to our accumulated normal
                        final_normal += np.array(normal)
            
            # Check collisions with doors
            for door in self.doors:
                if door.check_collision(pos, radius):
                    # Calculate normal pointing away from door center
                    door_pos = np.array([door.x, door.original_y, door.z])
                    to_pos = np.array(pos) - door_pos
                    normal = normalize(to_pos)
                    if normal is not None:
                        collision_found = True
                        # Add this normal to our accumulated normal
                        final_normal += np.array(normal)
            
            # Check collisions with pickup blocks
            # Skip this check if the position is from a pickup block itself
            # to avoid self-collision
            for block in self.pickup_blocks:
                # Skip blocks that are being carried
                if block.being_carried:
                    continue
                    
                # Skip if the position is very close to the block's center
                # (likely checking collision for this block itself)
                if np.linalg.norm(np.array(pos) - block.position) < 0.01:
                    continue
                    
                if block.check_collision(pos, radius):
                    normal = block.get_collision_normal(pos)
                    if normal is not None:
                        collision_found = True
                        # Add this normal to our accumulated normal
                        final_normal += np.array(normal)
            
            # If we found any collisions, normalize the accumulated normal
            if collision_found:
                norm = np.linalg.norm(final_normal)
                if norm > 0.001:
                    final_normal = final_normal / norm  # Normalize the vector
                else:
                    # If the normals canceled out, use a default upward normal
                    final_normal = np.array([0.0, 1.0, 0.0])
                return True, final_normal
            
            return False, None
            
        except Exception as e:
            print(f"Error in check_wall_collisions: {e}")
            import traceback
            traceback.print_exc()
            # Return a safe default
            return False, np.array([0.0, 1.0, 0.0])
    
    def check_goal_collision(self, pos, radius):
        """Check if player has collided with the exit cube in the hallway"""
        # First check if player has entered the hallway
        self.exit_hallway.check_entry(pos, radius)
        
        # Then check if player has touched the exit cube
        collision = self.exit_hallway.check_exit_cube_collision(pos, radius)
        
        if collision:
            print("Player reached the exit cube!")
            
        return collision
    
    def check_door_entry(self, player_pos, player_radius):
        """Check if player has entered any doors, especially the exit door"""
        if self.exit_door:
            if self.exit_door.check_player_entry(player_pos, player_radius):
                self.player_entered_exit = True
                return True
        
        # Check other doors too
        for door in self.doors:
            if door != self.exit_door:  # Skip exit door as we already checked it
                if door.check_player_entry(player_pos, player_radius):
                    return True
                    
        return False
    
    def check_door_passage(self, player_pos):
        """Check if player has passed through any doors completely"""
        if self.player_prev_pos is None:
            self.player_prev_pos = np.array(player_pos)
            return False
            
        player_pos_array = np.array(player_pos)
        
        # Check if player has passed through the exit door
        if self.exit_door and self.player_entered_exit:
            if self.exit_door.check_player_passed_through(player_pos_array, self.player_prev_pos):
                # Start the level transition timer
                self.level_transition_timer = 0
                return True
                
        # Check other doors too
        for door in self.doors:
            if door != self.exit_door:  # Skip exit door as we already checked it
                door.check_player_passed_through(player_pos_array, self.player_prev_pos)
                
        # Update previous position for next frame
        self.player_prev_pos = player_pos_array
        return False
    
    def ray_cast(self, ray_origin, ray_direction, max_distance=PORTAL_DISTANCE):
        """
        Cast a ray from origin in the given direction and find the closest intersection.
        
        Args:
            ray_origin: Starting point of the ray
            ray_direction: Direction of the ray (should be normalized)
            max_distance: Maximum distance to check for intersections
            
        Returns:
            Tuple of (intersection_point, normal, distance, wall)
        """
        # Ensure ray direction is normalized
        ray_direction = normalize(ray_direction)
        
        closest_intersection = None
        closest_distance = max_distance
        intersection_normal = None
        intersected_wall = None

        # Debug info
        print(f"Ray cast from {ray_origin} in direction {ray_direction}")
        print(f"Checking {len(self.walls)} walls for intersection")

        for wall in self.walls:
            intersection, normal, distance = wall.ray_intersection(ray_origin, ray_direction)

            if intersection is not None and distance < closest_distance:
                # Debug info
                print(f"Hit wall at {intersection}, distance={distance}, normal={normal}")
                print(f"Wall portal_surface: {wall.portal_surface}")
                
                # Check if this is a valid portal surface
                if not wall.portal_surface:
                    # Skip this wall if it doesn't allow portals
                    print("Skipping - not a portal surface")
                    continue
                    
                # Check if the intersection is near the goal
                # This prevents placing portals that could bypass the door
                if self.exit_door and not self.exit_door.player_passed_through:
                    # Calculate distance from intersection to goal
                    dist_to_goal = np.linalg.norm(np.array(intersection) - np.array(self.goal_pos))
                    # If intersection is too close to goal and player hasn't passed through door
                    if dist_to_goal < 5.0:  # Adjust this value as needed
                        # Skip this intersection - don't allow portal placement near goal
                        print(f"Skipping - too close to goal ({dist_to_goal})")
                        continue
                
                # Check if the normal is valid (not NaN)
                if np.isnan(normal).any():
                    print("Skipping - normal contains NaN values")
                    continue
                
                # Check if the normal is zero
                if np.linalg.norm(normal) < 0.001:
                    print("Skipping - normal is zero")
                    continue
                
                # This is a valid intersection
                closest_intersection = intersection
                closest_distance = distance
                intersection_normal = normal
                intersected_wall = wall
                print(f"Valid intersection found at {intersection}")

        if closest_intersection is None:
            print("No valid intersection found")
            
        return closest_intersection, intersection_normal, closest_distance, intersected_wall
        
    def update(self, dt, player_pos, player_radius):
        # Update all doors (if any)
        for door in self.doors:
            door.update(dt)
        
        # Update the exit hallway
        hallway_ready = self.exit_hallway.update(dt, player_pos)
        
        # Check if player has touched the exit cube
        if self.check_goal_collision(player_pos, player_radius) or hallway_ready:
            # Mark as ready for next level
            self.ready_for_next_level = True
            print("Exit cube touched! Ready for next level!")
            
            # Short delay for transition
            self.level_transition_delay = 0.5  # Short delay for smooth transition
            self.level_transition_timer += dt
                
        return self.ready_for_next_level

# Player class
class Player:
    def __init__(self, position, height=PLAYER_HEIGHT, radius=PLAYER_RADIUS):
        self.position = np.array(position)
        self.height = height
        self.radius = radius
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.pitch = 0.0
        self.on_ground = False
        self.camera_offset = np.array([0.0, height * 0.8, 0.0])  # Camera is at 80% of player height

        # Physics parameters
        self.terminal_velocity = -20.0  # Maximum falling speed
        self.air_control = 0.2          # How much control player has in air (0-1)
        self.ground_friction = 0.8      # Friction when on ground (0-1)
        self.air_friction = 0.02        # Friction when in air (0-1)

    def get_camera_position(self):
        return self.position + self.camera_offset

    def get_look_direction(self):
        # Calculate look direction based on yaw and pitch
        look_x = math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        look_y = math.sin(math.radians(self.pitch))
        look_z = math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))

        return normalize(np.array([look_x, look_y, look_z]))

    def get_right_vector(self):
        look_dir = self.get_look_direction()
        up = np.array([0.0, 1.0, 0.0])
        return normalize(cross_product(look_dir, up))

    def get_up_vector(self):
        look_dir = self.get_look_direction()
        right = self.get_right_vector()
        return normalize(cross_product(right, look_dir))

    def update(self, dt, level):
        # Apply gravity if not on ground (with terminal velocity)
        if not self.on_ground:
            # Accelerate downward due to gravity
            self.velocity[1] = max(self.velocity[1] - GRAVITY * dt, self.terminal_velocity)

            # Apply air friction to horizontal movement
            self.velocity[0] *= (1.0 - self.air_friction)
            self.velocity[2] *= (1.0 - self.air_friction)
        else:
            # Apply ground friction to horizontal movement
            self.velocity[0] *= (1.0 - self.ground_friction * dt)
            self.velocity[2] *= (1.0 - self.ground_friction * dt)

            # Stop horizontal movement if very slow
            if abs(self.velocity[0]) < 0.01:
                self.velocity[0] = 0
            if abs(self.velocity[2]) < 0.01:
                self.velocity[2] = 0

        # Use continuous collision detection to prevent clipping through walls
        # First, check if our current position is already in a wall (can happen after teleporting)
        collision, normal = level.check_wall_collisions(self.position, self.radius)
        if collision:
            # If we're already in a wall, push ourselves out along the normal
            push_distance = 0.1  # Adjust as needed
            self.position += np.array(normal) * push_distance
            
            # Adjust velocity to prevent moving back into the wall
            dot_product = np.dot(self.velocity, normal)
            if dot_product < 0:  # If moving toward the wall
                # Remove the component of velocity going into the wall
                self.velocity -= dot_product * np.array(normal)
        
        # Now perform the regular movement update with smaller steps to prevent tunneling
        num_steps = 3  # Divide the movement into multiple smaller steps
        step_dt = dt / num_steps
        
        for _ in range(num_steps):
            # Calculate the step movement
            step_movement = self.velocity * step_dt
            new_position = self.position + step_movement
            
            # Check for collisions with walls and blocks
            collision, normal = level.check_wall_collisions(new_position, self.radius)
            
            if collision:
                # Calculate reflection with proper physics
                dot_product = np.dot(self.velocity, normal)
                
                # Only reflect if moving toward the surface
                if dot_product < 0:
                    # Reflect velocity off wall (with damping)
                    # Ensure normal is a numpy array
                    normal_array = np.array(normal)
                    reflection = self.velocity - 2 * dot_product * normal_array
                    
                    # Apply damping based on surface type
                    damping = 0.5  # Default damping factor
                    self.velocity = reflection * damping
                    
                    # Check if we're on the ground - ensure normal is a valid array
                    if normal is not None and isinstance(normal, (list, np.ndarray)) and normal[1] > 0.7:  # Normal pointing mostly up
                        self.on_ground = True
                        self.velocity[1] = 0  # Stop vertical movement
                    
                    # Calculate a safe position that doesn't penetrate the wall
                    # Project the movement along the wall surface
                    # Ensure normal is a numpy array
                    normal_array = np.array(normal)
                    tangent = step_movement - np.dot(step_movement, normal_array) * normal_array
                    safe_position = self.position + tangent * 0.9  # Slightly reduced to avoid edge cases
                    
                    # Check if the safe position is actually safe
                    safe_collision, _ = level.check_wall_collisions(safe_position, self.radius)
                    if not safe_collision:
                        new_position = safe_position
                    else:
                        # If the safe position is still colliding, just don't move
                        new_position = self.position
                else:
                    # If we're moving away from the wall but still colliding,
                    # we might be inside a wall - push out along normal
                    push_distance = 0.1  # Adjust as needed
                    # Ensure normal is a valid numpy array
                    if normal is not None and isinstance(normal, (list, np.ndarray)):
                        normal_array = np.array(normal)
                        new_position = self.position + normal_array * push_distance
                    else:
                        # Fallback if normal is invalid
                        new_position = self.position + np.array([0.0, 0.1, 0.0])  # Push up slightly
            else:
                # We're not colliding with walls, but check if we're standing on a pickup block
                standing_on_block = False
                
                # Check each pickup block
                for block in level.pickup_blocks:
                    # Skip blocks that are being carried
                    if block.being_carried:
                        continue
                        
                    # Check if we're standing on this block
                    if block.check_if_standing_on(new_position, self.radius):
                        standing_on_block = True
                        self.on_ground = True
                        self.velocity[1] = 0  # Stop vertical movement
                        
                        # Adjust position to be exactly on top of the block
                        new_position[1] = block.position[1] + block.size + self.radius
                        break
                
                # If we're not standing on a block, check if we're on the ground
                if not standing_on_block and self.on_ground:
                    # Check if there's ground below us
                    ground_check_pos = new_position.copy()
                    ground_check_pos[1] -= 0.1  # Check slightly below our feet
                    ground_collision, ground_normal = level.check_wall_collisions(ground_check_pos, self.radius)
                    
                    # Fix for numpy array comparison
                    # Check if ground_normal exists and if its y component is less than 0.7
                    if not ground_collision or (ground_normal is not None and isinstance(ground_normal, np.ndarray) and ground_normal[1] < 0.7):
                        self.on_ground = False
            
            # Update position for this step
            self.position = new_position

    def jump(self):
        """
        Make the player jump if they are on the ground.
        This adds an upward velocity to the player.
        """
        try:
            if self.on_ground:
                # Apply jump force
                self.velocity[1] = JUMP_FORCE
                self.on_ground = False
                print("Player jumped!")
            else:
                print("Cannot jump - not on ground")
        except Exception as e:
            print(f"Error in jump method: {e}")
            # Fallback jump implementation
            self.velocity[1] = JUMP_FORCE
            self.on_ground = False

    def move(self, direction, speed):
        # Get movement vectors
        look_dir = self.get_look_direction()
        right = self.get_right_vector()

        # Zero out y component for horizontal movement
        look_dir[1] = 0
        look_dir = normalize(look_dir)

        # Calculate movement direction
        if direction == "forward":
            move_dir = look_dir
        elif direction == "backward":
            move_dir = -look_dir
        elif direction == "left":
            move_dir = -right
        elif direction == "right":
            move_dir = right
        else:
            return

        # Apply movement with reduced control in air
        if self.on_ground:
            # Full control on ground
            self.velocity[0] = move_dir[0] * speed
            self.velocity[2] = move_dir[2] * speed
        else:
            # Reduced control in air
            self.velocity[0] += move_dir[0] * speed * self.air_control * 0.1
            self.velocity[2] += move_dir[2] * speed * self.air_control * 0.1

            # Cap horizontal air speed
            horizontal_speed = math.sqrt(self.velocity[0]**2 + self.velocity[2]**2)
            if horizontal_speed > speed:
                scale = speed / horizontal_speed
                self.velocity[0] *= scale
                self.velocity[2] *= scale

# Button class for pressure plates
class Button:
    def __init__(self, x, y, z, width, height, depth, texture_id=None):
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth
        self.texture_id = texture_id
        self.pressed = False
        self.activation_time = 0
        self.objects_on_button = []
        
        # Calculate vertices
        half_width = width / 2
        half_height = height / 2
        half_depth = depth / 2
        
        self.vertices = [
            # Top face (slightly raised when not pressed)
            [x - half_width, y + half_height, z - half_depth],
            [x + half_width, y + half_height, z - half_depth],
            [x + half_width, y + half_height, z + half_depth],
            [x - half_width, y + half_height, z + half_depth],
            
            # Base (always at the same height)
            [x - half_width, y - half_height, z - half_depth],
            [x + half_width, y - half_height, z - half_depth],
            [x + half_width, y - half_height, z + half_depth],
            [x - half_width, y - half_height, z + half_depth]
        ]
        
    def draw(self):
        # Save current state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glPushMatrix()
        
        # Translate to button position
        glTranslatef(self.x, self.y, self.z)
        
        # Adjust height based on pressed state
        button_height = 0.05 if self.pressed else 0.15
        
        # Draw the button base (always the same)
        glBegin(GL_QUADS)
        # Set color for the base (dark gray)
        glColor3f(0.3, 0.3, 0.3)
        
        # Base top
        glVertex3f(-self.width/2, -self.height/2 + 0.01, -self.depth/2)
        glVertex3f(self.width/2, -self.height/2 + 0.01, -self.depth/2)
        glVertex3f(self.width/2, -self.height/2 + 0.01, self.depth/2)
        glVertex3f(-self.width/2, -self.height/2 + 0.01, self.depth/2)
        glEnd()
        
        # Draw the button top (changes height when pressed)
        glBegin(GL_QUADS)
        # Set color based on pressed state
        if self.pressed:
            glColor3f(1.0, 0.0, 0.0)  # Red when pressed
        else:
            glColor3f(0.8, 0.0, 0.0)  # Darker red when not pressed
        
        # Button top
        glVertex3f(-self.width/2, button_height, -self.depth/2)
        glVertex3f(self.width/2, button_height, -self.depth/2)
        glVertex3f(self.width/2, button_height, self.depth/2)
        glVertex3f(-self.width/2, button_height, self.depth/2)
        
        # Button sides
        glColor3f(0.6, 0.0, 0.0)  # Darker red for sides
        
        # Front
        glVertex3f(-self.width/2, -self.height/2 + 0.01, self.depth/2)
        glVertex3f(self.width/2, -self.height/2 + 0.01, self.depth/2)
        glVertex3f(self.width/2, button_height, self.depth/2)
        glVertex3f(-self.width/2, button_height, self.depth/2)
        
        # Back
        glVertex3f(-self.width/2, -self.height/2 + 0.01, -self.depth/2)
        glVertex3f(self.width/2, -self.height/2 + 0.01, -self.depth/2)
        glVertex3f(self.width/2, button_height, -self.depth/2)
        glVertex3f(-self.width/2, button_height, -self.depth/2)
        
        # Left
        glVertex3f(-self.width/2, -self.height/2 + 0.01, -self.depth/2)
        glVertex3f(-self.width/2, -self.height/2 + 0.01, self.depth/2)
        glVertex3f(-self.width/2, button_height, self.depth/2)
        glVertex3f(-self.width/2, button_height, -self.depth/2)
        
        # Right
        glVertex3f(self.width/2, -self.height/2 + 0.01, -self.depth/2)
        glVertex3f(self.width/2, -self.height/2 + 0.01, self.depth/2)
        glVertex3f(self.width/2, button_height, self.depth/2)
        glVertex3f(self.width/2, button_height, -self.depth/2)
        glEnd()
        
        # Restore state
        glPopMatrix()
        glPopAttrib()
    
    def check_collision(self, pos, radius):
        """
        Check if an object is on the button.
        Greatly expanded collision area to make buttons much easier to activate.
        """
        # Increase detection area for much easier activation
        expanded_radius = radius * 2.0  # 100% larger radius (doubled)
        half_width = self.width / 2 + expanded_radius
        half_depth = self.depth / 2 + expanded_radius
        
        # Only check collision with the top of the button
        button_height = 0.05 if self.pressed else 0.15
        
        # Greatly expanded vertical range for much easier activation
        vertical_range = 1.0  # Increased to 1.0 to make it much easier to activate
        
        # Print debug info when near button
        dist_to_button = np.linalg.norm(np.array([pos[0] - self.x, pos[1] - self.y, pos[2] - self.z]))
        if dist_to_button < 3.0:  # If within 3 units of button
            print(f"Near button at ({self.x}, {self.y}, {self.z}), distance: {dist_to_button:.2f}")
            print(f"Player position: {pos}")
            print(f"Button collision bounds: X: {self.x - half_width} to {self.x + half_width}, " +
                  f"Y: {self.y - self.height/2} to {self.y + button_height + vertical_range}, " +
                  f"Z: {self.z - half_depth} to {self.z + half_depth}")
        
        return (pos[0] >= self.x - half_width and pos[0] <= self.x + half_width and
                pos[1] <= self.y + button_height + vertical_range and pos[1] >= self.y - self.height/2 and
                pos[2] >= self.z - half_depth and pos[2] <= self.z + half_depth)
    
    def update(self, objects):
        """
        Update button state based on objects above it.
        Added sticky behavior to make buttons stay pressed longer.
        """
        # Reset the list of objects on the button
        self.objects_on_button = []
        
        # Check if any objects are on the button
        for obj in objects:
            if self.check_collision(obj.position, obj.radius):
                self.objects_on_button.append(obj)
        
        # Update pressed state
        was_pressed = self.pressed
        
        # If objects are on the button, press it permanently
        if len(self.objects_on_button) > 0:
            if not self.pressed:
                self.pressed = True
                self.activation_time = time.time()
                print(f"Button pressed permanently! Objects on button: {len(self.objects_on_button)}")
        
        # Once pressed, the button stays pressed forever
        # This ensures the door stays open permanently
        
        # If button state changed, update activation time and print message
        if was_pressed != self.pressed:
            if self.pressed:
                self.activation_time = time.time()
                print("Button pressed! Will stay active for 3 seconds.")
                # You could add a button press sound here
            else:
                print("Button released!")
                # You could add a button release sound here
        
        return self.pressed

# PickupBlock class for movable objects
class PickupBlock:
    def __init__(self, position, size=0.5, texture_id=None):
        self.position = np.array(position)
        self.size = size
        self.radius = size  # For collision detection
        self.texture_id = texture_id
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.on_ground = False
        self.being_carried = False
        self.carrier = None
        self.last_collision_time = 0  # Track last collision time for better physics
        
        # Physics parameters
        self.mass = 5.0
        self.terminal_velocity = -20.0
        self.friction = 0.8
        self.restitution = 0.5  # Bounciness factor (0-1)
    
    def draw(self):
        # Save current state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glPushMatrix()
        
        # Translate to block position
        glTranslatef(self.position[0], self.position[1], self.position[2])
        
        # Set material properties
        if ENABLE_LIGHTING:
            if self.texture_id:
                # Use bright white material with texture to preserve texture colors
                ambient = [1.0, 1.0, 1.0, 1.0]
                diffuse = [1.0, 1.0, 1.0, 1.0]
                specular = [0.5, 0.5, 0.5, 1.0]
                
                # Enable color material to work better with textures
                glEnable(GL_COLOR_MATERIAL)
                glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
                glColor4f(1.0, 1.0, 1.0, 1.0)  # Full white to show texture properly
            else:
                # Use colored material without texture
                ambient = [0.2, 0.2, 0.2, 1.0]
                diffuse = [0.8, 0.8, 0.8, 1.0]
                specular = [0.5, 0.5, 0.5, 1.0]
                
                # Disable color material when not using textures
                glDisable(GL_COLOR_MATERIAL)
            
            glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
            glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
            glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
            glMaterialf(GL_FRONT, GL_SHININESS, 32.0)
        
        # Bind texture if available
        if self.texture_id:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # Draw the cube
        glBegin(GL_QUADS)
        
        # Set color
        if self.texture_id:
            glColor3f(1.0, 1.0, 1.0)  # White for textured objects
        else:
            glColor3f(0.7, 0.5, 0.3)  # Brown/wooden color for untextured objects
        
        # Define vertices for a cube
        s = self.size
        vertices = [
            # Front face
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
            # Back face
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        ]
        
        # Define texture coordinates
        tex_coords = [
            [0, 0], [1, 0], [1, 1], [0, 1]
        ]
        
        # Define faces (indices of vertices)
        faces = [
            [0, 1, 2, 3],  # Front
            [4, 7, 6, 5],  # Back
            [1, 5, 6, 2],  # Right
            [0, 3, 7, 4],  # Left
            [3, 2, 6, 7],  # Top
            [0, 4, 5, 1]   # Bottom
        ]
        
        # Draw each face
        for i, face in enumerate(faces):
            for j, vertex_idx in enumerate(face):
                if self.texture_id:
                    glTexCoord2fv(tex_coords[j])
                glVertex3fv(vertices[vertex_idx])
        
        glEnd()
        
        # Disable textures
        glDisable(GL_TEXTURE_2D)
        
        # Restore state
        glPopMatrix()
        glPopAttrib()
    
    def update(self, dt, level):
        # Skip physics update if being carried
        if self.being_carried:
            return
        
        # Apply gravity if not on ground
        if not self.on_ground:
            self.velocity[1] = max(self.velocity[1] - GRAVITY * dt, self.terminal_velocity)
        else:
            # Apply friction to horizontal movement
            self.velocity[0] *= (1.0 - self.friction * dt)
            self.velocity[2] *= (1.0 - self.friction * dt)
            
            # Stop horizontal movement if very slow
            if abs(self.velocity[0]) < 0.01:
                self.velocity[0] = 0
            if abs(self.velocity[2]) < 0.01:
                self.velocity[2] = 0
        
        # Use continuous collision detection to prevent clipping through walls
        # First, check if our current position is already in a wall
        collision, normal = level.check_wall_collisions(self.position, self.radius)
        if collision:
            # If we're already in a wall, push ourselves out along the normal
            push_distance = 0.1  # Adjust as needed
            self.position += np.array(normal) * push_distance
            
            # Adjust velocity to prevent moving back into the wall
            dot_product = np.dot(self.velocity, normal)
            if dot_product < 0:  # If moving toward the wall
                # Remove the component of velocity going into the wall
                self.velocity -= dot_product * np.array(normal)
        
        # Now perform the regular movement update with smaller steps to prevent tunneling
        num_steps = 3  # Divide the movement into multiple smaller steps
        step_dt = dt / num_steps
        
        for _ in range(num_steps):
            # Calculate the step movement
            step_movement = self.velocity * step_dt
            new_position = self.position + step_movement
            
            # Check for collisions with walls
            collision, normal = level.check_wall_collisions(new_position, self.radius)
            
            if collision:
                # Calculate reflection with proper physics
                dot_product = np.dot(self.velocity, normal)
                
                # Only reflect if moving toward the surface
                if dot_product < 0:
                    # Reflect velocity off wall (with damping)
                    normal_array = np.array(normal)
                    reflection = self.velocity - 2 * dot_product * normal_array
                    
                    # Apply damping based on surface type
                    damping = self.restitution  # Use restitution for bounciness
                    self.velocity = reflection * damping
                    
                    # Check if we're on the ground - ensure normal is a valid array
                    if normal is not None and isinstance(normal, (list, np.ndarray)) and normal[1] > 0.7:  # Normal pointing mostly up
                        self.on_ground = True
                        self.velocity[1] = 0  # Stop vertical movement
                    
                    # Calculate a safe position that doesn't penetrate the wall
                    # Project the movement along the wall surface
                    normal_array = np.array(normal)
                    tangent = step_movement - np.dot(step_movement, normal_array) * normal_array
                    safe_position = self.position + tangent * 0.9  # Slightly reduced to avoid edge cases
                    
                    # Check if the safe position is actually safe
                    safe_collision, _ = level.check_wall_collisions(safe_position, self.radius)
                    if not safe_collision:
                        new_position = safe_position
                    else:
                        # If the safe position is still colliding, just don't move
                        new_position = self.position
                else:
                    # If we're moving away from the wall but still colliding,
                    # we might be inside a wall - push out along normal
                    push_distance = 0.1  # Adjust as needed
                    if normal is not None and isinstance(normal, (list, np.ndarray)):
                        normal_array = np.array(normal)
                        new_position = self.position + normal_array * push_distance
                    else:
                        # Fallback if normal is invalid
                        new_position = self.position + np.array([0.0, 0.1, 0.0])  # Push up slightly
            else:
                # We're not colliding, so we're not on the ground
                # But only set on_ground to False if we're not on the ground in this step
                if self.on_ground:
                    # Check if there's ground below us
                    ground_check_pos = new_position.copy()
                    ground_check_pos[1] -= 0.1  # Check slightly below our feet
                    ground_collision, ground_normal = level.check_wall_collisions(ground_check_pos, self.radius)
                    
                    # Fix for numpy array comparison
                    # Check if ground_normal exists and if its y component is less than 0.7
                    if not ground_collision or (ground_normal is not None and isinstance(ground_normal, np.ndarray) and ground_normal[1] < 0.7):
                        self.on_ground = False
            
            # Update position for this step
            self.position = new_position
            
            # Check for collisions with other blocks
            self.check_block_collisions(level.pickup_blocks)
    
    def check_block_collisions(self, blocks):
        """Check for collisions with other pickup blocks and resolve them"""
        # Skip if being carried
        if self.being_carried:
            return
            
        for other_block in blocks:
            # Skip self-collision and blocks being carried
            if other_block is self or other_block.being_carried:
                continue
                
            # Calculate distance between block centers
            distance = np.linalg.norm(self.position - other_block.position)
            
            # If blocks are overlapping
            min_distance = self.size + other_block.size
            if distance < min_distance:
                # Calculate collision normal (direction from other block to this block)
                if distance < 0.0001:  # Avoid division by zero
                    # If blocks are exactly at the same position, push in a random direction
                    collision_normal = np.array([1.0, 0.0, 0.0])  # Default to X axis
                else:
                    collision_normal = (self.position - other_block.position) / distance
                
                # Calculate overlap amount
                overlap = min_distance - distance
                
                # Push blocks apart based on their relative masses
                total_mass = self.mass + other_block.mass
                self_ratio = other_block.mass / total_mass
                other_ratio = self.mass / total_mass
                
                # Move this block away from collision
                self.position += collision_normal * overlap * self_ratio
                
                # Calculate relative velocity along collision normal
                relative_velocity = np.dot(self.velocity - other_block.velocity, collision_normal)
                
                # Only apply impulse if blocks are moving toward each other
                if relative_velocity < 0:
                    # Calculate impulse scalar
                    impulse = -(1 + self.restitution) * relative_velocity
                    impulse /= (1/self.mass + 1/other_block.mass)
                    
                    # Apply impulse to velocities
                    self.velocity += (impulse / self.mass) * collision_normal
                    other_block.velocity -= (impulse / other_block.mass) * collision_normal
                    
                    # Check if either block is now on the ground
                    if collision_normal[1] > 0.7:  # Normal pointing mostly up for this block
                        self.on_ground = True
                        self.velocity[1] = 0  # Stop vertical movement
                    elif collision_normal[1] < -0.7:  # Normal pointing mostly up for other block
                        other_block.on_ground = True
                        other_block.velocity[1] = 0  # Stop vertical movement
    
    def get_corners(self):
        """Get the 8 corners of the cube for collision detection"""
        s = self.size
        corners = []
        for x in [-s, s]:
            for y in [-s, s]:
                for z in [-s, s]:
                    corners.append(self.position + np.array([x, y, z]))
        return corners
    
    def check_collision(self, pos, radius):
        """Check if a position is colliding with this cube"""
        # Calculate distance from position to cube center
        dist = np.linalg.norm(np.array(pos) - self.position)
        
        # Quick check - if distance is greater than sum of radii, no collision
        if dist > (radius + self.size * 1.414):  # 1.414 is sqrt(2) for cube diagonal
            return False
        
        # More precise check - find closest point on cube to position
        closest_point = np.array([
            max(self.position[0] - self.size, min(pos[0], self.position[0] + self.size)),
            max(self.position[1] - self.size, min(pos[1], self.position[1] + self.size)),
            max(self.position[2] - self.size, min(pos[2], self.position[2] + self.size))
        ])
        
        # Calculate distance from closest point to position
        dist_to_closest = np.linalg.norm(np.array(pos) - closest_point)
        
        # Collision if distance is less than radius
        return dist_to_closest <= radius
    
    def get_collision_normal(self, pos):
        """Get the normal vector pointing away from the cube at the collision point"""
        # Find closest point on cube to position
        closest_point = np.array([
            max(self.position[0] - self.size, min(pos[0], self.position[0] + self.size)),
            max(self.position[1] - self.size, min(pos[1], self.position[1] + self.size)),
            max(self.position[2] - self.size, min(pos[2], self.position[2] + self.size))
        ])
        
        # Calculate vector from closest point to position
        normal = np.array(pos) - closest_point
        
        # Normalize the vector
        norm = np.linalg.norm(normal)
        if norm < 0.0001:
            # If we're exactly at the closest point, use a default normal
            # Find which face we're closest to
            distances = [
                abs(closest_point[0] - (self.position[0] - self.size)),  # -X face
                abs(closest_point[0] - (self.position[0] + self.size)),  # +X face
                abs(closest_point[1] - (self.position[1] - self.size)),  # -Y face
                abs(closest_point[1] - (self.position[1] + self.size)),  # +Y face
                abs(closest_point[2] - (self.position[2] - self.size)),  # -Z face
                abs(closest_point[2] - (self.position[2] + self.size))   # +Z face
            ]
            
            # Find the closest face
            closest_face = distances.index(min(distances))
            
            # Return normal based on closest face
            if closest_face == 0:
                return np.array([-1.0, 0.0, 0.0])
            elif closest_face == 1:
                return np.array([1.0, 0.0, 0.0])
            elif closest_face == 2:
                return np.array([0.0, -1.0, 0.0])
            elif closest_face == 3:
                return np.array([0.0, 1.0, 0.0])
            elif closest_face == 4:
                return np.array([0.0, 0.0, -1.0])
            else:
                return np.array([0.0, 0.0, 1.0])
        
        return normal / norm
    
    def check_if_standing_on(self, pos, radius):
        """Check if an object is standing on top of this cube"""
        # Only consider positions that are above the cube
        if pos[1] < self.position[1] + self.size - 0.1:  # Allow slight overlap
            return False
        
        # Check if position is within horizontal bounds of cube (with some margin)
        # Use a more generous margin to make it easier to stand on the cube
        margin = 0.2  # Additional margin beyond the radius
        x_min = self.position[0] - self.size - radius - margin
        x_max = self.position[0] + self.size + radius + margin
        z_min = self.position[2] - self.size - radius - margin
        z_max = self.position[2] + self.size + radius + margin
        
        if pos[0] < x_min or pos[0] > x_max or pos[2] < z_min or pos[2] > z_max:
            return False
        
        # Check if position is close enough to the top face
        top_face_y = self.position[1] + self.size
        distance_to_top = pos[1] - radius - top_face_y
        
        # If we're very close to or slightly penetrating the top face
        # Use a more generous threshold to make it easier to stand on the cube
        return abs(distance_to_top) < 0.2  # Increased from 0.1 for better detection
    
    def pick_up(self, carrier):
        self.being_carried = True
        self.carrier = carrier
        self.velocity = np.array([0.0, 0.0, 0.0])  # Stop all movement
        print("Block picked up")
    
    def drop(self, throw_velocity=None):
        self.being_carried = False
        self.carrier = None
        
        # Apply throw velocity if provided
        if throw_velocity is not None:
            self.velocity = throw_velocity
        
        print("Block dropped")

# Game class
class PortalGame:
    def __init__(self):
        # Load textures - use default texture if files don't exist
        print("\nLoading game textures...")
        print("=" * 50)
        print("Note: If textures are missing, please add them to the assets/textures directory.")
        print("Especially make sure concrete-wall.jpeg exists for wall textures.")
        print("=" * 50)

        # Load each texture individually with better error handling
        print("\nLoading game textures...")
        
        # Wall texture - try multiple formats
        print("Attempting to load wall texture...")
        self.wall_texture = None
        for ext in ['.jpeg', '.jpg', '.png', '.bmp']:
            try:
                texture_name = f"concrete-wall{ext}"
                print(f"Trying {texture_name}...")
                self.wall_texture = load_texture(texture_name)
                if self.wall_texture:
                    print(f"Successfully loaded wall texture: {texture_name}")
                    break
            except Exception as e:
                print(f"Failed to load {texture_name}: {e}")
        
        # If all attempts failed, use default texture
        if not self.wall_texture:
            print("Using default wall texture")
            self.wall_texture = create_default_texture()
        
        # Load other textures with individual error handling
        print("\nLoading floor texture...")
        try:
            self.floor_texture = load_texture("floor_concrete.png")
        except Exception as e:
            print(f"Error loading floor texture: {e}")
            self.floor_texture = create_default_texture()
            
        print("\nLoading ceiling texture...")
        try:
            self.ceiling_texture = load_texture("ceiling_tile.png")
        except Exception as e:
            print(f"Error loading ceiling texture: {e}")
            self.ceiling_texture = create_default_texture()
            
        print("\nLoading portal textures...")
        try:
            self.portal_blue_texture = load_texture("portal_blue_rim.png")
        except Exception as e:
            print(f"Error loading blue portal texture: {e}")
            self.portal_blue_texture = create_default_texture()
            
        try:
            self.portal_orange_texture = load_texture("portal_orange_rim.png")
        except Exception as e:
            print(f"Error loading orange portal texture: {e}")
            self.portal_orange_texture = create_default_texture()
            
        print("\nLoading object textures...")
        try:
            self.cube_texture = load_texture("cube_texture.png")
        except Exception as e:
            print(f"Error loading cube texture: {e}")
            self.cube_texture = create_default_texture()
            
        try:
            self.button_texture = load_texture("button_texture.png")
        except Exception as e:
            print(f"Error loading button texture: {e}")
            self.button_texture = create_default_texture()

        print("\nTexture loading complete. If you see any checkerboard patterns in the game,")
        print("it means the corresponding texture file is missing.")
        print("=" * 50)

        # Load sounds - handle missing sound files
        try:
            print("\nLoading game sounds...")
            print("=" * 50)

            # Try to load the new portal-shoot.mp3 sound
            print("Attempting to load portal-shoot.mp3...")
            self.portal_shoot_sound = load_sound("portal-shoot.mp3")
            if self.portal_shoot_sound:
                print("Successfully loaded portal-shoot.mp3")
            else:
                print("Could not load portal-shoot.mp3, will fall back to default sounds")

            # Load other sounds as fallbacks
            self.portal_blue_sound = load_sound("portal_open_blue.wav")
            self.portal_orange_sound = load_sound("portal_open_orange.wav")
            self.portal_enter_sound = load_sound("portal_enter.wav")
            self.level_complete_sound = load_sound("level_complete.wav")

            print("Sound loading complete.")
            print("=" * 50)
        except Exception as e:
            print(f"Error loading sounds: {e}")
            self.portal_shoot_sound = None
            self.portal_blue_sound = None
            self.portal_orange_sound = None
            self.portal_enter_sound = None
            self.level_complete_sound = None
        
        # Create player
        self.player = None
        
        # Create portals
        self.blue_portal = None
        self.orange_portal = None
        
        # Create pickup blocks and buttons
        self.pickup_blocks = []
        self.buttons = []
        self.holding_block = False
        self.held_block = None
        self.pickup_distance = 2.0  # Maximum distance to pick up a block
        self.carry_distance = 2.0   # Distance to carry block in front of player
        
        # Create levels
        self.levels = [self.create_level1(), self.create_level2(), self.create_level3()]
        self.current_level_index = 0
        self.current_level = self.levels[self.current_level_index]
        
        # Reset player position and objects
        self.reset_player()
        
        # Game state
        self.running = True
        self.level_complete = False
        self.last_time = time.time()
        self.fps_counter = 0
        self.fps_timer = 0
        self.fps = 0
        
        # Lock mouse to center of screen
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
    
    def create_level1(self):
        # Create walls with textures - Tutorial level
        walls = [
            # Floor
            Wall(0, -1, 0, 20, 0.5, 20, self.floor_texture),
            
            # Ceiling
            Wall(0, 5, 0, 20, 0.5, 20, self.ceiling_texture),
            
            # Surrounding walls
            Wall(-10, 2, 0, 0.5, 6, 20, self.wall_texture),  # Left wall
            Wall(10, 2, 0, 0.5, 6, 20, self.wall_texture),   # Right wall
            Wall(0, 2, -10, 20, 6, 0.5, self.wall_texture),  # Back wall
            Wall(0, 2, 10, 20, 6, 0.5, self.wall_texture),   # Front wall
            
            # Simple obstacle in the middle
            Wall(0, 1, 0, 4, 2, 0.5, self.wall_texture),     # Middle wall
            
            # Goal platform - enclosed area that can only be accessed through the door
            Wall(7, 0, -7, 4, 0.5, 4, self.floor_texture),   # Goal platform
            
            # Walls to enclose the goal area - making it only accessible through the door
            Wall(7, 1, -9, 4, 2, 0.5, self.wall_texture),    # Back wall of goal area
            Wall(9, 1, -7, 0.5, 2, 4, self.wall_texture),    # Right wall of goal area
            Wall(5, 1, -7, 0.5, 2, 4, self.wall_texture),    # Left wall of goal area
            
            # Barrier wall that forces player to use the door
            Wall(7, 1, -3, 4, 2, 0.5, self.wall_texture),    # Front barrier wall
            
            # Side walls to create a corridor to the door
            Wall(8, 1, -4, 0.5, 2, 2, self.wall_texture),    # Right corridor wall
            Wall(6, 1, -4, 0.5, 2, 2, self.wall_texture),    # Left corridor wall
        ]
        
        # Create buttons
        buttons = [
            Button(5, -0.7, 5, 1.5, 0.1, 1.5, self.button_texture)  # Button near the middle
        ]
        
        # Create pickup blocks
        pickup_blocks = [
            PickupBlock([3, 0.5, 8], 0.5, self.cube_texture)  # Block near the player start
        ]
        
        # Create doors
        doors = [
            Door(7, 1, -5, 2, 2, 0.5, self.wall_texture)  # Door to the goal area
        ]
        
        # Set door direction (direction player must move to pass through)
        doors[0].direction = normalize(np.array([0, 0, -1]))  # Player must move in negative Z direction
        
        # Connect the button to the door
        doors[0].required_buttons = [buttons[0]]
        
        goal_pos = (7, 1, -7)  # Position of the goal
        player_start = (0, 0, 8)  # Starting position of the player
        
        # Create the level
        level = Level(walls, goal_pos, player_start, pickup_blocks, buttons, doors)
        
        # Set the exit door
        level.exit_door = doors[0]
        
        return level
    
    def create_level2(self):
        # Create a more complex level with height differences
        walls = [
            # Floor
            Wall(0, -1, 0, 30, 0.5, 30, self.floor_texture),
            
            # Ceiling
            Wall(0, 8, 0, 30, 0.5, 30, self.ceiling_texture),
            
            # Surrounding walls
            Wall(-15, 3.5, 0, 0.5, 9, 30, self.wall_texture),  # Left wall
            Wall(15, 3.5, 0, 0.5, 9, 30, self.wall_texture),   # Right wall
            Wall(0, 3.5, -15, 30, 9, 0.5, self.wall_texture),  # Back wall
            Wall(0, 3.5, 15, 30, 9, 0.5, self.wall_texture),   # Front wall
            
            # Interior structures
            Wall(-10, 1, -5, 0.5, 4, 10, self.wall_texture),   # Left divider
            Wall(10, 1, 5, 0.5, 4, 10, self.wall_texture),     # Right divider
            Wall(0, 1, 0, 10, 4, 0.5, self.wall_texture),      # Middle divider
            
            # Platforms at different heights
            Wall(-10, 2, 10, 6, 0.5, 6, self.floor_texture),   # Platform 1
            Wall(10, 4, -10, 6, 0.5, 6, self.floor_texture),   # Platform 2 (higher)
            Wall(0, 6, 0, 4, 0.5, 4, self.floor_texture),      # Central platform (highest)
            
            # Additional obstacles
            Wall(-5, 0.5, -8, 3, 1, 3, self.wall_texture),     # Low obstacle 1
            Wall(5, 0.5, 8, 3, 1, 3, self.wall_texture),       # Low obstacle 2
            
            # Completely removed all walls and barriers
            # Added only a platform for the goal
            Wall(0, 4, 0, 8, 0.1, 8, self.floor_texture),      # Large platform for the goal area
        ]
        
        # Create buttons - made much larger and more visible
        buttons = [
            Button(-10, 2.3, 10, 3.0, 0.1, 3.0, self.button_texture),  # Button on platform 1 - doubled size
            Button(10, 4.3, -10, 3.0, 0.1, 3.0, self.button_texture)   # Button on platform 2 - doubled size
        ]
        
        # Create pickup blocks
        pickup_blocks = [
            PickupBlock([-12, 0.5, 8], 0.5, self.cube_texture),  # Block near the player start
            PickupBlock([12, 0.5, 8], 0.5, self.cube_texture)    # Block on the other side
        ]
        
        # No doors - completely open level
        doors = []
        
        # No doors to connect buttons to
        
        goal_pos = (0, 5, 0)  # Goal in the center of the area - lowered for better visibility
        player_start = (-12, 0, 12)  # Starting in a corner
        
        # Create the level
        level = Level(walls, goal_pos, player_start, pickup_blocks, buttons, doors)
        
        # No exit door
        
        return level
    
    def create_level3(self):
        # Advanced level with momentum puzzles
        walls = [
            # Floor
            Wall(0, -1, 0, 40, 0.5, 40, self.floor_texture),
            
            # Ceiling
            Wall(0, 12, 0, 40, 0.5, 40, self.ceiling_texture),
            
            # Surrounding walls
            Wall(-20, 5.5, 0, 0.5, 13, 40, self.wall_texture),  # Left wall
            Wall(20, 5.5, 0, 0.5, 13, 40, self.wall_texture),   # Right wall
            Wall(0, 5.5, -20, 40, 13, 0.5, self.wall_texture),  # Back wall
            Wall(0, 5.5, 20, 40, 13, 0.5, self.wall_texture),   # Front wall
            
            # Main chamber dividers
            Wall(-10, 3, 0, 0.5, 8, 30, self.wall_texture),     # Left chamber wall
            Wall(10, 3, 0, 0.5, 8, 30, self.wall_texture),      # Right chamber wall
            
            # Platforms for momentum puzzles
            Wall(-15, 6, -15, 8, 0.5, 8, self.floor_texture),   # High platform 1
            Wall(15, 2, 15, 8, 0.5, 8, self.floor_texture),     # Low platform 2
            
            # Central structure - modified to be only accessible through the door
            Wall(0, 4, 0, 12, 8, 12, self.wall_texture, False), # Central non-portal structure
            Wall(0, 9, 0, 6, 0.5, 6, self.floor_texture),       # Top platform with goal
            
            # Ramps and additional structures
            Wall(-15, 1, 10, 8, 4, 0.5, self.wall_texture),     # Wall 1
            Wall(15, 1, -10, 8, 4, 0.5, self.wall_texture),     # Wall 2
            Wall(0, 0, -15, 30, 0.5, 4, self.floor_texture),    # Extended floor section
            
            # Walls to completely enclose the central platform and goal
            Wall(0, 10, 3, 6, 2, 0.5, self.wall_texture),       # Front wall of goal area
            Wall(0, 10, -3, 6, 2, 0.5, self.wall_texture),      # Back wall of goal area
            Wall(3, 10, 0, 0.5, 2, 6, self.wall_texture),       # Right wall of goal area
            Wall(-3, 10, 0, 0.5, 2, 6, self.wall_texture),      # Left wall of goal area
            Wall(0, 12, 0, 6, 0.5, 6, self.wall_texture),       # Ceiling of goal area
            
            # Barrier walls to force using the door
            Wall(0, 6, 9, 12, 6, 0.5, self.wall_texture),       # Front barrier wall
            Wall(-6, 6, 7.5, 0.5, 6, 3, self.wall_texture),     # Left corridor wall
            Wall(6, 6, 7.5, 0.5, 6, 3, self.wall_texture),      # Right corridor wall
            
            # Additional barriers to prevent portal shortcuts
            Wall(0, 6, 3, 12, 6, 0.5, self.wall_texture),       # Inner barrier wall
            
            # Access corridor to the goal
            Wall(0, 9, 0, 2, 0.5, 6, self.wall_texture),        # Corridor floor
            Wall(1, 10, 0, 0.5, 2, 6, self.wall_texture),       # Corridor right wall
            Wall(-1, 10, 0, 0.5, 2, 6, self.wall_texture),      # Corridor left wall
        ]
        
        # Create buttons
        buttons = [
            Button(-15, 6.3, -15, 1.5, 0.1, 1.5, self.button_texture),  # Button on high platform
            Button(15, 2.3, 15, 1.5, 0.1, 1.5, self.button_texture)     # Button on low platform
        ]
        
        # Create pickup blocks
        pickup_blocks = [
            PickupBlock([0, 0.5, 18], 0.5, self.cube_texture),    # Block near the player start
            PickupBlock([-15, 0.5, 10], 0.5, self.cube_texture),  # Block near wall 1
            PickupBlock([15, 0.5, -10], 0.5, self.cube_texture)   # Block near wall 2
        ]
        
        # Create doors
        doors = [
            Door(0, 6, 6, 2, 6, 0.5, self.wall_texture)  # Door to the central structure
        ]
        
        # Set door direction (direction player must move to pass through)
        doors[0].direction = normalize(np.array([0, 0, -1]))  # Player must move in negative Z direction
        
        # Connect both buttons to the door (requires both to be pressed)
        doors[0].required_buttons = buttons
        
        goal_pos = (0, 10, 0)  # Goal on top of central structure
        player_start = (0, 0, 18)  # Starting at front
        
        # Create the level
        level = Level(walls, goal_pos, player_start, pickup_blocks, buttons, doors)
        
        # Set the exit door
        level.exit_door = doors[0]
        
        return level
    
    def reset_player(self):
        """
        Reset the player and level state.
        This is called when starting a new level or when the player presses R.
        """
        try:
            # Create a new player at the level's starting position
            self.player = Player(self.current_level.player_start)
            
            # Reset portals
            self.blue_portal = None
            self.orange_portal = None
            
            # Reset level completion status
            self.level_complete = False
            
            # Reset pickup blocks and buttons
            self.pickup_blocks = self.current_level.pickup_blocks.copy()
            self.buttons = self.current_level.buttons.copy()
            self.holding_block = False
            self.held_block = None
            
            # Reset level state
            if hasattr(self.current_level, 'ready_for_next_level'):
                self.current_level.ready_for_next_level = False
            
            # Reset door states
            for door in self.current_level.doors:
                door.is_open = False
                door.open_amount = 0.0
                door.player_entered = False
                door.player_passed_through = False
                door.force_close = False
            
            # Reset level transition timer
            self.current_level.level_transition_timer = 0
            self.current_level.player_entered_exit = False
            
            print(f"Player reset to starting position in Level {self.current_level_index + 1}")
        except Exception as e:
            print(f"Error in reset_player: {e}")
            import traceback
            traceback.print_exc()
    
    def next_level(self):
        """
        Advance to the next level in the game.
        This method is called automatically when a level is completed,
        or can be triggered manually with the 'N' key when a level is marked as complete.
        """
        try:
            print("Moving to next level...")
            
            # Create a smooth transition effect
            # Fade out effect could be added here in a more advanced implementation
            
            # Increment level index with wraparound
            self.current_level_index = (self.current_level_index + 1) % len(self.levels)
            
            # Set the new current level
            self.current_level = self.levels[self.current_level_index]
            
            # Reset player and level state
            self.reset_player()
            
            # Reset all portals
            self.blue_portal = None
            self.orange_portal = None
            
            # Reset pickup blocks and buttons
            self.pickup_blocks = self.current_level.pickup_blocks.copy()
            self.buttons = self.current_level.buttons.copy()
            self.holding_block = False
            self.held_block = None
            
            # Display message about the new level
            print(f"Now playing Level {self.current_level_index + 1}")
            
            # Add a visual indicator that the level has changed
            pygame.display.set_caption(f"Portal Remake - Level {self.current_level_index + 1}")
            
            # Force reset of any level completion flags
            self.level_complete = False
            if hasattr(self.current_level, 'ready_for_next_level'):
                self.current_level.ready_for_next_level = False
                
            # Reset any door states in the level
            for door in self.current_level.doors:
                door.is_open = False
                door.open_amount = 0.0
                door.player_entered = False
                door.player_passed_through = False
                door.force_close = False
                
            # Reset level transition timer
            self.current_level.level_transition_timer = 0
            self.current_level.player_entered_exit = False
            
            # Reset exit hallway state
            if hasattr(self.current_level, 'exit_hallway'):
                self.current_level.exit_hallway.entered = False
                self.current_level.exit_hallway.door_closed = False
                self.current_level.exit_hallway.door_position = 0.0
                self.current_level.exit_hallway.exit_cube.touched = False
                self.current_level.exit_hallway.exit_cube.transition_timer = 0
            
            # Add a brief pause for smoother transition
            time.sleep(0.2)
            
            print("Level transition complete!")
            
        except Exception as e:
            print(f"Error in next_level: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_events(self):
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.player.jump()
                    elif event.key == pygame.K_r:
                        self.reset_player()
                        print("Level reset!")
                    elif event.key == pygame.K_n:
                        # Allow N key to work even if level is not complete (for testing)
                        # But print different messages
                        if self.level_complete:
                            print("Level complete! Moving to next level...")
                        else:
                            print("DEBUG: Skipping to next level...")
                        self.next_level()
                    elif event.key == pygame.K_e:  # E key for picking up/dropping blocks
                        self.handle_pickup_drop()
                    elif event.key == pygame.K_f:  # F key for throwing blocks
                        self.throw_held_block()
                    # Add debug key to force level completion
                    elif event.key == pygame.K_c:
                        self.level_complete = True
                        print("DEBUG: Level marked as complete. Press N to go to next level.")
                        pygame.display.set_caption(f"Portal Remake - Level {self.current_level_index + 1} COMPLETE! Press N for next level")
                    
                    # Add debug key to force open all doors
                    elif event.key == pygame.K_o:
                        print("DEBUG: Forcing all doors open!")
                        for door in self.current_level.doors:
                            door.is_open = True
                            door.open_amount = 1.0
                            door.force_close = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click - Blue portal
                        if not self.holding_block:  # Only create portals if not holding a block
                            self.create_portal("blue")
                    elif event.button == 3:  # Right click - Orange portal
                        if not self.holding_block:  # Only create portals if not holding a block
                            self.create_portal("orange")
        except Exception as e:
            print(f"Error in handle_events: {e}")
            import traceback
            traceback.print_exc()

        # Handle continuous key presses
        keys = pygame.key.get_pressed()

        # Only reset movement velocity when on ground
        if self.player.on_ground:
            self.player.velocity[0] = 0
            self.player.velocity[2] = 0

        # Apply movement based on keys
        if keys[pygame.K_w]:
            self.player.move("forward", MOVEMENT_SPEED)
        if keys[pygame.K_s]:
            self.player.move("backward", MOVEMENT_SPEED)
        if keys[pygame.K_a]:
            self.player.move("left", MOVEMENT_SPEED)
        if keys[pygame.K_d]:
            self.player.move("right", MOVEMENT_SPEED)

        # Handle mouse movement for camera
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        self.player.yaw += mouse_dx * MOUSE_SENSITIVITY
        self.player.pitch -= mouse_dy * MOUSE_SENSITIVITY

        # Clamp pitch to avoid flipping
        self.player.pitch = max(-89, min(89, self.player.pitch))

    def create_portal(self, color):
        try:
            # Get player's camera position and look direction
            camera_pos = self.player.get_camera_position()
            look_dir = self.player.get_look_direction()
            
            # Print debug info
            print(f"Attempting to create {color} portal")
            print(f"Camera position: {camera_pos}")
            print(f"Look direction: {look_dir}")

            # Cast ray to find portal placement location
            intersection, normal, distance, wall = self.current_level.ray_cast(camera_pos, look_dir)
            
            # Debug info about ray cast
            if intersection is None:
                print("Ray cast failed: No intersection found")
                return
            else:
                print(f"Ray cast hit at: {intersection} with normal: {normal}")
                print(f"Distance: {distance}, Wall portal_surface: {wall.portal_surface if wall else 'No wall'}")

            # If we hit a valid surface for portal placement
            if intersection is not None and wall is not None and wall.portal_surface:
                # Check if the portal would be placed in a restricted area (near goal)
                if self.current_level.exit_door and not self.current_level.exit_door.player_passed_through:
                    # Calculate distance from intersection to goal
                    dist_to_goal = np.linalg.norm(np.array(intersection) - np.array(self.current_level.goal_pos))
                    # If intersection is too close to goal and player hasn't passed through door
                    if dist_to_goal < 5.0:  # Adjust this value as needed
                        print(f"Cannot place {color} portal: Area is restricted until puzzle is solved")
                        if self.portal_shoot_sound:
                            self.portal_shoot_sound.set_volume(0.3)
                            self.portal_shoot_sound.play()
                            self.portal_shoot_sound.set_volume(1.0)
                        return
                
                # Adjust portal position to be centered at player height
                # This ensures the portal is positioned to accommodate the player's height
                up_vector = np.array([0, 1, 0])
                if abs(np.dot(normal, up_vector)) > 0.9:
                    # Portal on floor/ceiling - no adjustment needed
                    adjusted_position = intersection
                else:
                    # Portal on wall - adjust height to match player
                    # Find the up direction relative to the wall
                    if abs(normal[1]) < 0.1:  # If normal is mostly horizontal
                        height_offset = (PORTAL_HEIGHT * 0.5) - PLAYER_HEIGHT * 0.4
                        adjusted_position = intersection + np.array([0, height_offset, 0])
                    else:
                        # For other surfaces, use the original position
                        adjusted_position = intersection
                
                # Move the portal slightly away from the wall to prevent z-fighting
                adjusted_position += normal * 0.01
                
                print(f"Adjusted position: {adjusted_position}")

                # Create a temporary portal to check for overlap
                temp_portal = Portal(adjusted_position, normal, color)
                
                # Check if the new portal would overlap with an existing portal
                if color == "blue" and self.orange_portal and self.orange_portal.active and temp_portal.check_overlap(self.orange_portal):
                    print("Cannot place blue portal: Would overlap with orange portal")
                    # Play error sound or visual feedback here if available
                    if self.portal_shoot_sound:
                        # Modify pitch to indicate error
                        self.portal_shoot_sound.set_volume(0.3)
                        self.portal_shoot_sound.play()
                        self.portal_shoot_sound.set_volume(1.0)
                    return
                elif color == "orange" and self.blue_portal and self.blue_portal.active and temp_portal.check_overlap(self.blue_portal):
                    print("Cannot place orange portal: Would overlap with blue portal")
                    # Play error sound or visual feedback here if available
                    if self.portal_shoot_sound:
                        # Modify pitch to indicate error
                        self.portal_shoot_sound.set_volume(0.3)
                        self.portal_shoot_sound.play()
                        self.portal_shoot_sound.set_volume(1.0)
                    return
                    
                # Also check if the portal would overlap with itself (replacing an existing portal)
                # This ensures portals are placed with enough distance between them
                if color == "blue" and self.blue_portal and self.blue_portal.active:
                    # Check if new blue portal would be too close to existing blue portal
                    if np.linalg.norm(adjusted_position - self.blue_portal.position) < max(PORTAL_WIDTH, PORTAL_HEIGHT) * 0.8:
                        print("Cannot place blue portal: Too close to existing blue portal")
                        if self.portal_shoot_sound:
                            self.portal_shoot_sound.set_volume(0.3)
                            self.portal_shoot_sound.play()
                            self.portal_shoot_sound.set_volume(1.0)
                        return
                elif color == "orange" and self.orange_portal and self.orange_portal.active:
                    # Check if new orange portal would be too close to existing orange portal
                    if np.linalg.norm(adjusted_position - self.orange_portal.position) < max(PORTAL_WIDTH, PORTAL_HEIGHT) * 0.8:
                        print("Cannot place orange portal: Too close to existing orange portal")
                        if self.portal_shoot_sound:
                            self.portal_shoot_sound.set_volume(0.3)
                            self.portal_shoot_sound.play()
                            self.portal_shoot_sound.set_volume(1.0)
                        return
                    
                # Create the portal with adjusted position
                if color == "blue":
                    self.blue_portal = Portal(adjusted_position, normal, "blue", self.portal_blue_texture)
                    # Set up portal linking
                    if self.orange_portal and self.orange_portal.active:
                        self.blue_portal.linked_portal = self.orange_portal
                        self.orange_portal.linked_portal = self.blue_portal
                    # Play portal shoot sound
                    if self.portal_shoot_sound:
                        self.portal_shoot_sound.play()
                    elif self.portal_blue_sound:  # Fallback
                        self.portal_blue_sound.play()
                    print("Blue portal created successfully")
                else:  # Orange
                    self.orange_portal = Portal(adjusted_position, normal, "orange", self.portal_orange_texture)
                    # Set up portal linking
                    if self.blue_portal and self.blue_portal.active:
                        self.orange_portal.linked_portal = self.blue_portal
                        self.blue_portal.linked_portal = self.orange_portal
                    # Play portal shoot sound
                    if self.portal_shoot_sound:
                        self.portal_shoot_sound.play()
                    elif self.portal_orange_sound:  # Fallback
                        self.portal_orange_sound.play()
                    print("Orange portal created successfully")
            else:
                print(f"Cannot place {color} portal: Invalid surface")
                # Play error sound
                if self.portal_shoot_sound:
                    self.portal_shoot_sound.set_volume(0.3)
                    self.portal_shoot_sound.play()
                    self.portal_shoot_sound.set_volume(1.0)
        except Exception as e:
            print(f"Error creating portal: {e}")
            import traceback
            traceback.print_exc()

    def handle_pickup_drop(self):
        # If already holding a block, drop it
        if self.holding_block and self.held_block:
            self.drop_held_block()
            return
            
        # If not holding a block, try to pick one up
        camera_pos = self.player.get_camera_position()
        look_dir = self.player.get_look_direction()
        
        # Check each block to see if it's in front of the player
        closest_block = None
        closest_distance = self.pickup_distance
        
        for block in self.pickup_blocks:
            if block.being_carried:
                continue
                
            # Calculate vector from camera to block
            to_block = block.position - camera_pos
            distance = np.linalg.norm(to_block)
            
            # Check if block is within pickup distance
            if distance <= self.pickup_distance:
                # Check if block is in front of player (dot product with look direction is positive)
                if np.dot(normalize(to_block), look_dir) > 0.7:  # Within ~45 degrees of look direction
                    if distance < closest_distance:
                        closest_block = block
                        closest_distance = distance
        
        # If found a block to pick up
        if closest_block:
            self.pickup_block(closest_block)
    
    def pickup_block(self, block):
        if not self.holding_block:
            # Check if the block would be in a valid position when picked up
            camera_pos = self.player.get_camera_position()
            look_dir = self.player.get_look_direction()
            
            # Calculate where the block would be positioned when held
            desired_position = camera_pos + look_dir * self.carry_distance
            desired_position[1] -= 0.3  # Adjust height to be at eye level
            
            # Check if this position would cause a collision
            collision, _ = self.current_level.check_wall_collisions(desired_position, block.radius)
            
            if collision:
                # Try to find a valid position by moving the block closer to the player
                valid_position_found = False
                for distance_factor in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
                    test_position = camera_pos + look_dir * (self.carry_distance * distance_factor)
                    test_position[1] -= 0.3  # Maintain eye level adjustment
                    
                    # Check if this position is safe
                    test_collision, _ = self.current_level.check_wall_collisions(test_position, block.radius)
                    if not test_collision:
                        valid_position_found = True
                        break
                
                if not valid_position_found:
                    print("Cannot pick up block - no valid position found")
                    return
            
            # If we get here, we can pick up the block
            self.holding_block = True
            self.held_block = block
            block.pick_up(self.player)
            print("Picked up block")
    
    def drop_held_block(self):
        if self.holding_block and self.held_block:
            # Check if the current position of the block is valid for dropping
            # (it might be inside a wall due to the player moving into a wall)
            current_pos = self.held_block.position
            collision, normal = self.current_level.check_wall_collisions(current_pos, self.held_block.radius)
            
            if collision:
                # The block is currently in an invalid position
                # Try to find a safe position to drop it
                camera_pos = self.player.get_camera_position()
                look_dir = self.player.get_look_direction()
                
                # Try different distances in front of the player
                safe_position_found = False
                for distance_factor in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    test_position = camera_pos + look_dir * (self.carry_distance * distance_factor)
                    test_position[1] -= 0.3  # Maintain eye level adjustment
                    
                    # Check if this position is safe
                    test_collision, _ = self.current_level.check_wall_collisions(test_position, self.held_block.radius)
                    if not test_collision:
                        # Found a safe position, update the block's position
                        self.held_block.position = test_position
                        safe_position_found = True
                        break
                
                if not safe_position_found:
                    # If we couldn't find a safe position in front, try dropping it at the player's feet
                    feet_position = self.player.position.copy()
                    feet_position[1] += self.held_block.size + 0.1  # Position block just above the ground
                    
                    # Check if this position is safe
                    feet_collision, _ = self.current_level.check_wall_collisions(feet_position, self.held_block.radius)
                    if not feet_collision:
                        self.held_block.position = feet_position
                    else:
                        # If all else fails, push the block out along the collision normal
                        push_distance = self.held_block.radius + 0.1
                        self.held_block.position = current_pos + np.array(normal) * push_distance
                        print("Warning: Block dropped in tight space")
            
            # Now drop the block
            self.held_block.drop()
            self.holding_block = False
            self.held_block = None
            print("Dropped block")
    
    def throw_held_block(self):
        if self.holding_block and self.held_block:
            # Check if the current position of the block is valid for throwing
            # (it might be inside a wall due to the player moving into a wall)
            current_pos = self.held_block.position
            collision, normal = self.current_level.check_wall_collisions(current_pos, self.held_block.radius)
            
            if collision:
                # The block is currently in an invalid position
                # Try to find a safe position to throw it from
                camera_pos = self.player.get_camera_position()
                look_dir = self.player.get_look_direction()
                
                # Try different distances in front of the player
                safe_position_found = False
                for distance_factor in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    test_position = camera_pos + look_dir * (self.carry_distance * distance_factor)
                    test_position[1] -= 0.3  # Maintain eye level adjustment
                    
                    # Check if this position is safe
                    test_collision, _ = self.current_level.check_wall_collisions(test_position, self.held_block.radius)
                    if not test_collision:
                        # Found a safe position, update the block's position
                        self.held_block.position = test_position
                        safe_position_found = True
                        break
                
                if not safe_position_found:
                    # If we couldn't find a safe position, just push it out along the normal
                    push_distance = self.held_block.radius + 0.1
                    self.held_block.position = current_pos + np.array(normal) * push_distance
                    print("Warning: Block thrown from tight space")
            
            # Calculate throw velocity based on player's look direction
            throw_strength = 10.0
            throw_velocity = self.player.get_look_direction() * throw_strength
            
            # Add a slight upward component
            throw_velocity[1] += 2.0
            
            # Check if throwing in this direction would immediately hit a wall
            test_position = self.held_block.position + throw_velocity * 0.1  # Check a small distance ahead
            test_collision, test_normal = self.current_level.check_wall_collisions(test_position, self.held_block.radius)
            
            if test_collision:
                # If we would hit a wall, adjust the throw direction
                # Reflect the throw velocity off the wall
                dot_product = np.dot(throw_velocity, test_normal)
                if dot_product < 0:  # Only reflect if moving toward the wall
                    reflection = throw_velocity - 2 * dot_product * np.array(test_normal)
                    throw_velocity = reflection * 0.8  # Reduce velocity a bit after reflection
                    print("Adjusted throw direction to avoid wall")
            
            # Drop with throw velocity
            self.held_block.drop(throw_velocity)
            self.holding_block = False
            self.held_block = None
            print("Threw block")
    
    def update_held_block(self):
        if self.holding_block and self.held_block:
            # Position the block in front of the player
            camera_pos = self.player.get_camera_position()
            look_dir = self.player.get_look_direction()
            
            # Calculate desired position for the block
            desired_position = camera_pos + look_dir * self.carry_distance
            
            # Adjust height to be at eye level
            desired_position[1] -= 0.3  # Slightly below eye level
            
            # Check if the desired position would cause a collision with walls
            collision, normal = self.current_level.check_wall_collisions(desired_position, self.held_block.radius)
            
            if collision:
                # If there would be a collision, find a safe position
                # Start by moving the block closer to the player
                for distance_factor in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
                    test_position = camera_pos + look_dir * (self.carry_distance * distance_factor)
                    test_position[1] -= 0.3  # Maintain eye level adjustment
                    
                    # Check if this position is safe
                    test_collision, _ = self.current_level.check_wall_collisions(test_position, self.held_block.radius)
                    if not test_collision:
                        # Found a safe position
                        desired_position = test_position
                        break
                
                # If we couldn't find a safe position by reducing distance,
                # try pushing the block away from the wall along the normal
                if collision:
                    # Push away from the wall
                    push_distance = 0.1  # Adjust as needed
                    desired_position = desired_position + np.array(normal) * push_distance
            
            # Update the block's position to the safe position
            self.held_block.position = desired_position
    
    def check_portal_transport(self):
        try:
            # Check if both portals exist
            if self.blue_portal is None or self.orange_portal is None or not self.blue_portal.active or not self.orange_portal.active:
                return
                
            # Get current time for cooldown check
            current_time = time.time()
            
            # Reduced cooldown time for more responsive portal transitions
            cooldown_time = 0.15  # Further reduced from 0.2 for better responsiveness
            
            # Check if player is colliding with either portal
            if self.blue_portal.check_collision(self.player.position, self.player.radius):
                # Check if we're moving toward the portal (dot product of velocity and normal is negative)
                dot_product = np.dot(self.player.velocity, self.blue_portal.normal)
                
                # Calculate distance to portal center
                dist_to_portal = np.linalg.norm(self.player.position - self.blue_portal.position)
                
                # Calculate the player's speed
                player_speed = np.linalg.norm(self.player.velocity)
                
                # Calculate the projected position (where the player would be in the next frame)
                projected_pos = self.player.position + self.player.velocity * 0.05  # Assuming 50ms frame time
                
                # Check if the projected position crosses the portal plane
                current_side = np.dot(self.player.position - self.blue_portal.position, self.blue_portal.normal)
                projected_side = np.dot(projected_pos - self.blue_portal.position, self.blue_portal.normal)
                crossing_portal = (current_side * projected_side <= 0)  # Sign change indicates crossing
                
                # Transport if:
                # 1. Moving toward the portal OR
                # 2. Very close to the portal OR
                # 3. Jumping near the portal (vertical velocity is significant) OR
                # 4. Moving fast enough (helps with momentum-based puzzles) OR
                # 5. About to cross the portal plane
                if ((dot_product < 0) or 
                    (dist_to_portal < self.player.radius * 3.0) or  # Increased from 2.5
                    (abs(self.player.velocity[1]) > 1.0 and dist_to_portal < self.player.radius * 4.0) or  # Increased from 3.5
                    (player_speed > 5.0 and dist_to_portal < self.player.radius * 4.5) or  # Increased from 4.0
                    crossing_portal) and \
                   (current_time - self.blue_portal.last_transport_time) > cooldown_time:
                    
                    # Transport from blue to orange
                    self.transport_player(self.blue_portal, self.orange_portal)
                    # Update last transport time
                    self.blue_portal.last_transport_time = current_time
                    self.orange_portal.last_transport_time = current_time
                    print("Transported through blue portal")

            elif self.orange_portal.check_collision(self.player.position, self.player.radius):
                # Check if we're moving toward the portal
                dot_product = np.dot(self.player.velocity, self.orange_portal.normal)
                
                # Calculate distance to portal center
                dist_to_portal = np.linalg.norm(self.player.position - self.orange_portal.position)
                
                # Calculate the player's speed
                player_speed = np.linalg.norm(self.player.velocity)
                
                # Calculate the projected position (where the player would be in the next frame)
                projected_pos = self.player.position + self.player.velocity * 0.05  # Assuming 50ms frame time
                
                # Check if the projected position crosses the portal plane
                current_side = np.dot(self.player.position - self.orange_portal.position, self.orange_portal.normal)
                projected_side = np.dot(projected_pos - self.orange_portal.position, self.orange_portal.normal)
                crossing_portal = (current_side * projected_side <= 0)  # Sign change indicates crossing
                
                # Transport if:
                # 1. Moving toward the portal OR
                # 2. Very close to the portal OR
                # 3. Jumping near the portal (vertical velocity is significant) OR
                # 4. Moving fast enough (helps with momentum-based puzzles) OR
                # 5. About to cross the portal plane
                if ((dot_product < 0) or 
                    (dist_to_portal < self.player.radius * 3.0) or  # Increased from 2.5
                    (abs(self.player.velocity[1]) > 1.0 and dist_to_portal < self.player.radius * 4.0) or  # Increased from 3.5
                    (player_speed > 5.0 and dist_to_portal < self.player.radius * 4.5) or  # Increased from 4.0
                    crossing_portal) and \
                   (current_time - self.orange_portal.last_transport_time) > cooldown_time:
                    
                    # Transport from orange to blue
                    self.transport_player(self.orange_portal, self.blue_portal)
                    # Update last transport time
                    self.orange_portal.last_transport_time = current_time
                    self.blue_portal.last_transport_time = current_time
                    print("Transported through orange portal")
                    
            # Check if any pickup blocks are colliding with portals
            for block in self.pickup_blocks:
                # Skip blocks that are being carried
                if block.being_carried:
                    continue
                    
                # Calculate the block's speed
                block_speed = np.linalg.norm(block.velocity)
                
                # Check collision with blue portal
                if self.blue_portal.check_collision(block.position, block.radius):
                    # Check if block is moving toward the portal
                    dot_product = np.dot(block.velocity, self.blue_portal.normal)
                    
                    # Calculate distance to portal center
                    dist_to_portal = np.linalg.norm(block.position - self.blue_portal.position)
                    
                    # Calculate the projected position (where the block would be in the next frame)
                    projected_pos = block.position + block.velocity * 0.05  # Assuming 50ms frame time
                    
                    # Check if the projected position crosses the portal plane
                    current_side = np.dot(block.position - self.blue_portal.position, self.blue_portal.normal)
                    projected_side = np.dot(projected_pos - self.blue_portal.position, self.blue_portal.normal)
                    crossing_portal = (current_side * projected_side <= 0)  # Sign change indicates crossing
                    
                    # Transport if:
                    # 1. Moving toward the portal OR
                    # 2. Very close to the portal OR
                    # 3. Moving fast enough OR
                    # 4. About to cross the portal plane
                    if ((dot_product < 0) or 
                        (dist_to_portal < block.radius * 3.0) or  # Increased from 2.5
                        (block_speed > 3.0 and dist_to_portal < block.radius * 3.5) or  # Increased from 3.0
                        crossing_portal) and \
                       (current_time - self.blue_portal.last_transport_time) > cooldown_time:
                        
                        # Transport block from blue to orange
                        self.transport_object(block, self.blue_portal, self.orange_portal)
                        # Update last transport time
                        self.blue_portal.last_transport_time = current_time
                        self.orange_portal.last_transport_time = current_time
                        print("Block transported through blue portal")
                
                # Check collision with orange portal
                elif self.orange_portal.check_collision(block.position, block.radius):
                    # Check if block is moving toward the portal
                    dot_product = np.dot(block.velocity, self.orange_portal.normal)
                    
                    # Calculate distance to portal center
                    dist_to_portal = np.linalg.norm(block.position - self.orange_portal.position)
                    
                    # Calculate the projected position (where the block would be in the next frame)
                    projected_pos = block.position + block.velocity * 0.05  # Assuming 50ms frame time
                    
                    # Check if the projected position crosses the portal plane
                    current_side = np.dot(block.position - self.orange_portal.position, self.orange_portal.normal)
                    projected_side = np.dot(projected_pos - self.orange_portal.position, self.orange_portal.normal)
                    crossing_portal = (current_side * projected_side <= 0)  # Sign change indicates crossing
                    
                    # Transport if:
                    # 1. Moving toward the portal OR
                    # 2. Very close to the portal OR
                    # 3. Moving fast enough OR
                    # 4. About to cross the portal plane
                    if ((dot_product < 0) or 
                        (dist_to_portal < block.radius * 3.0) or  # Increased from 2.5
                        (block_speed > 3.0 and dist_to_portal < block.radius * 3.5) or  # Increased from 3.0
                        crossing_portal) and \
                       (current_time - self.orange_portal.last_transport_time) > cooldown_time:
                        
                        # Transport block from orange to blue
                        self.transport_object(block, self.orange_portal, self.blue_portal)
                        # Update last transport time
                        self.orange_portal.last_transport_time = current_time
                        self.blue_portal.last_transport_time = current_time
                        print("Block transported through orange portal")
        
        except Exception as e:
            print(f"Error in portal transport check: {e}")
            import traceback
            traceback.print_exc()

    def transport_player(self, entry_portal, exit_portal):
        try:
            # Play portal enter sound
            if self.portal_enter_sound:
                self.portal_enter_sound.play()
                
            # Calculate the exact intersection point with the portal plane
            # This ensures seamless transitions between portals
            
            # Get the player's position and velocity
            player_pos = np.array(self.player.position, dtype=np.float64)  # Use float64 for better precision
            player_vel = np.array(self.player.velocity, dtype=np.float64)
            
            # Store original velocity magnitude for conservation of momentum
            original_vel_magnitude = np.linalg.norm(player_vel)
            
            # Calculate the normal of the entry portal
            entry_normal = np.array(entry_portal.normal, dtype=np.float64)
            
            # Calculate the distance from the player to the portal plane
            # The plane equation is: dot(normal, point - position) = 0
            dist_to_portal = np.dot(entry_normal, player_pos - entry_portal.position)
            
            # Calculate the exact intersection point with the portal plane
            # This is the point where the player crosses the portal
            intersection_point = player_pos - dist_to_portal * entry_normal
            
            # Calculate the offset from the entry portal to the intersection point
            offset_from_entry = intersection_point - entry_portal.position
            
            # Create coordinate systems for both portals
            entry_normal = np.array(entry_portal.normal, dtype=np.float64)
            entry_up = np.array(entry_portal.up, dtype=np.float64)
            entry_right = np.array(entry_portal.right, dtype=np.float64)
            
            exit_normal = -np.array(exit_portal.normal, dtype=np.float64)  # Negative because we're exiting
            exit_up = np.array(exit_portal.up, dtype=np.float64)
            exit_right = -np.array(exit_portal.right, dtype=np.float64)  # Negative to maintain right-hand rule
            
            # Ensure all vectors are normalized
            entry_normal = normalize(entry_normal)
            entry_up = normalize(entry_up)
            entry_right = normalize(entry_right)
            exit_normal = normalize(exit_normal)
            exit_up = normalize(exit_up)
            exit_right = normalize(exit_right)
            
            # Create transformation matrices
            # From world to entry portal coordinates
            entry_to_world = np.column_stack((entry_right, entry_up, entry_normal))
            
            # Check if the matrix is invertible
            det = np.linalg.det(entry_to_world)
            if abs(det) < 0.001:
                # Matrix is nearly singular, use a fallback approach
                print("Warning: Portal transformation matrix is nearly singular. Using fallback.")
                # Simple offset-based teleportation as fallback
                portal_offset = exit_portal.position - entry_portal.position
                new_position = player_pos + portal_offset
                new_velocity = player_vel + exit_portal.normal * 2.0  # Add boost in exit direction
                new_yaw = self.player.yaw
                new_pitch = self.player.pitch
            else:
                try:
                    # Normal transformation path
                    world_to_entry = np.linalg.inv(entry_to_world)
                    
                    # From exit portal to world coordinates
                    exit_to_world = np.column_stack((exit_right, exit_up, exit_normal))
                    
                    # Transform the offset to the entry portal's coordinate system
                    offset_in_entry = np.dot(world_to_entry, offset_from_entry)
                    
                    # Flip the normal component to go through the portal
                    offset_in_entry[2] = -offset_in_entry[2]
                    
                    # Transform back to world coordinates in the exit portal's frame
                    offset_in_world = np.dot(exit_to_world, offset_in_entry)
                    
                    # Calculate the new position
                    new_position = exit_portal.position + offset_in_world
                    
                    # Now transform the velocity vector - preserve magnitude for consistent physics
                    vel_in_entry = np.dot(world_to_entry, player_vel)
                    vel_in_entry[2] = -vel_in_entry[2]  # Flip normal component
                    new_velocity = np.dot(exit_to_world, vel_in_entry)
                    
                    # Preserve velocity magnitude to ensure momentum conservation
                    if original_vel_magnitude > 0:
                        # Scale the new velocity to match the original magnitude
                        current_magnitude = np.linalg.norm(new_velocity)
                        if current_magnitude > 0:  # Avoid division by zero
                            new_velocity = new_velocity * (original_vel_magnitude / current_magnitude)
                        
                        # Add a boost in the direction of the exit portal normal
                        # This helps prevent getting stuck in walls
                        boost_factor = 2.0  # Increased boost factor to prevent sticking
                        new_velocity += exit_portal.normal * boost_factor
                    
                    # Calculate the new yaw and pitch angles
                    # First, get the player's look direction as a unit vector
                    look_dir = np.array([
                        np.cos(np.radians(self.player.yaw)) * np.cos(np.radians(self.player.pitch)),
                        np.sin(np.radians(self.player.pitch)),
                        np.sin(np.radians(self.player.yaw)) * np.cos(np.radians(self.player.pitch))
                    ], dtype=np.float64)
                    
                    # Transform the look direction through the portal
                    # This is the key part that makes the view direction change correctly
                    
                    # Step 1: Transform to entry portal's local space
                    look_in_entry = np.dot(world_to_entry, look_dir)
                    
                    # Step 2: Flip the direction perpendicular to the portal
                    look_in_entry[2] = -look_in_entry[2]  # Flip normal component
                    
                    # Step 3: Transform from exit portal's local space to world space
                    new_look_dir = np.dot(exit_to_world, look_in_entry)
                    
                    # Normalize the new look direction
                    new_look_dir = normalize(new_look_dir)
                    
                    print(f"Portal transport: Original look direction: {look_dir}")
                    print(f"Portal transport: New look direction: {new_look_dir}")
                    
                    # Calculate the new yaw angle from the new look direction
                    # Use arctan2 to get the correct quadrant
                    new_yaw = np.degrees(np.arctan2(new_look_dir[2], new_look_dir[0]))
                    
                    # Calculate the new pitch angle from the new look direction
                    # Clamp the input to arcsin to avoid domain errors
                    pitch_input = np.clip(new_look_dir[1], -1.0, 1.0)
                    new_pitch = np.degrees(np.arcsin(pitch_input))
                    
                    print(f"Portal transport: Original yaw/pitch: {self.player.yaw}/{self.player.pitch}")
                    print(f"Portal transport: New yaw/pitch: {new_yaw}/{new_pitch}")
                    
                except Exception as e:
                    print(f"Matrix transformation failed: {e}. Using fallback.")
                    # Fallback if matrix operations fail
                    portal_offset = exit_portal.position - entry_portal.position
                    new_position = player_pos + portal_offset
                    new_velocity = player_vel + exit_portal.normal * 2.0  # Add boost in exit direction
                    new_yaw = self.player.yaw
                    new_pitch = self.player.pitch
            
            # Ensure the new position is valid (no NaN values)
            if np.isnan(new_position).any() or np.isinf(new_position).any():
                print("Warning: Invalid position calculated. Using fallback position.")
                new_position = exit_portal.position + exit_portal.normal * 1.0
            
            # Ensure the new velocity is valid (no NaN values)
            if np.isnan(new_velocity).any() or np.isinf(new_velocity).any():
                print("Warning: Invalid velocity calculated. Using fallback velocity.")
                new_velocity = exit_portal.normal * 2.0
            
            # Update player properties
            self.player.position = new_position
            self.player.velocity = new_velocity
            self.player.yaw = new_yaw
            self.player.pitch = new_pitch
            
            # Adjust the player's position further away from the exit portal
            # to prevent immediate re-entry and avoid clipping
            self.player.position += exit_portal.normal * 0.5  # Increased offset for better clearance
            
            # Perform a collision check at the new position to ensure we're not in a wall
            collision, normal = self.current_level.check_wall_collisions(self.player.position, self.player.radius)
            if collision:
                # If we're in a wall, adjust position along the normal
                adjustment = normal * 0.5  # Increased adjustment
                self.player.position += adjustment
                
                # Also adjust velocity to prevent moving into the wall
                dot_product = np.dot(self.player.velocity, normal)
                if dot_product < 0:  # If moving toward the wall
                    # Remove the component of velocity going into the wall
                    self.player.velocity -= dot_product * np.array(normal)
                    
                    # Add a larger upward component to help clear obstacles
                    self.player.velocity[1] += 2.0
                    
                    # Add a small random component to help break symmetry in case of getting stuck
                    self.player.velocity[0] += (np.random.random() - 0.5) * 0.5
                    self.player.velocity[2] += (np.random.random() - 0.5) * 0.5

            # Play portal transport sound
            if self.portal_enter_sound:
                self.portal_enter_sound.play()
                
        except Exception as e:
            print(f"Error during player teleportation: {e}")
            import traceback
            traceback.print_exc()
            
            # Emergency fallback - just move the player to a safe position
            # This prevents crashes even if the teleportation math fails
            self.player.position = exit_portal.position + exit_portal.normal * 1.5
            self.player.velocity = exit_portal.normal * 3.0  # Give a good push in the exit direction
            
            # Play portal sound to indicate teleportation still happened
            if self.portal_enter_sound:
                self.portal_enter_sound.play()
            
    def transport_object(self, obj, entry_portal, exit_portal):
        """
        Transport an object (like a pickup block) through a portal.
        This preserves the object's velocity and momentum.
        
        Args:
            obj: The object to transport (must have position and velocity attributes)
            entry_portal: The portal the object is entering
            exit_portal: The portal the object will exit from
        """
        try:
            # Play portal enter sound
            if self.portal_enter_sound:
                self.portal_enter_sound.play()
                
            # Get the object's position and velocity
            obj_pos = np.array(obj.position, dtype=np.float64)  # Use float64 for better precision
            obj_vel = np.array(obj.velocity, dtype=np.float64)
            
            # Store original velocity magnitude for conservation of momentum
            original_vel_magnitude = np.linalg.norm(obj_vel)
            
            # Calculate the normal of the entry portal
            entry_normal = np.array(entry_portal.normal, dtype=np.float64)
            
            # Calculate the distance from the object to the portal plane
            dist_to_portal = np.dot(entry_normal, obj_pos - entry_portal.position)
            
            # Calculate the exact intersection point with the portal plane
            intersection_point = obj_pos - dist_to_portal * entry_normal
            
            # Calculate the offset from the entry portal to the intersection point
            offset_from_entry = intersection_point - entry_portal.position
            
            # Create coordinate systems for both portals
            entry_normal = np.array(entry_portal.normal, dtype=np.float64)
            entry_up = np.array(entry_portal.up, dtype=np.float64)
            entry_right = np.array(entry_portal.right, dtype=np.float64)
            
            exit_normal = -np.array(exit_portal.normal, dtype=np.float64)  # Negative because we're exiting
            exit_up = np.array(exit_portal.up, dtype=np.float64)
            exit_right = -np.array(exit_portal.right, dtype=np.float64)  # Negative to maintain right-hand rule
            
            # Ensure all vectors are normalized
            entry_normal = normalize(entry_normal)
            entry_up = normalize(entry_up)
            entry_right = normalize(entry_right)
            exit_normal = normalize(exit_normal)
            exit_up = normalize(exit_up)
            exit_right = normalize(exit_right)
            
            # Create transformation matrices
            # From world to entry portal coordinates
            entry_to_world = np.column_stack((entry_right, entry_up, entry_normal))
            
            # Check if the matrix is invertible
            det = np.linalg.det(entry_to_world)
            if abs(det) < 0.001:
                # Matrix is nearly singular, use a fallback approach
                print("Warning: Portal transformation matrix is nearly singular. Using fallback for object.")
                # Simple offset-based teleportation as fallback
                portal_offset = exit_portal.position - entry_portal.position
                new_position = obj_pos + portal_offset
                
                # Reflect velocity across portal normal
                new_velocity = obj_vel - 2 * np.dot(obj_vel, entry_normal) * entry_normal
                
                # Add boost in exit portal direction
                new_velocity += exit_portal.normal * 3.0  # Increased boost for better clearance
            else:
                try:
                    # Normal transformation path
                    world_to_entry = np.linalg.inv(entry_to_world)
                    
                    # From exit portal to world coordinates
                    exit_to_world = np.column_stack((exit_right, exit_up, exit_normal))
                    
                    # Transform the offset to the entry portal's coordinate system
                    offset_in_entry = np.dot(world_to_entry, offset_from_entry)
                    
                    # Flip the normal component to go through the portal
                    offset_in_entry[2] = -offset_in_entry[2]
                    
                    # Transform back to world coordinates in the exit portal's frame
                    offset_in_world = np.dot(exit_to_world, offset_in_entry)
                    
                    # Calculate the new position
                    new_position = exit_portal.position + offset_in_world
                    
                    # Now transform the velocity vector - preserve magnitude for consistent physics
                    vel_in_entry = np.dot(world_to_entry, obj_vel)
                    vel_in_entry[2] = -vel_in_entry[2]  # Flip normal component
                    new_velocity = np.dot(exit_to_world, vel_in_entry)
                    
                    # Preserve velocity magnitude to ensure momentum conservation
                    if original_vel_magnitude > 0:
                        # Scale the new velocity to match the original magnitude
                        current_magnitude = np.linalg.norm(new_velocity)
                        if current_magnitude > 0:  # Avoid division by zero
                            new_velocity = new_velocity * (original_vel_magnitude / current_magnitude)
                        
                        # Add a boost in the direction of the exit portal normal
                        # This helps prevent getting stuck in walls
                        boost_factor = 3.0  # Increased boost for objects
                        new_velocity += exit_portal.normal * boost_factor
                
                except Exception as e:
                    print(f"Matrix transformation failed for object: {e}. Using fallback.")
                    # Fallback if matrix operations fail
                    portal_offset = exit_portal.position - entry_portal.position
                    new_position = obj_pos + portal_offset
                    new_velocity = exit_portal.normal * 3.0  # Simple velocity in exit direction
            
            # Ensure the new position is valid (no NaN values)
            if np.isnan(new_position).any() or np.isinf(new_position).any():
                print("Warning: Invalid object position calculated. Using fallback position.")
                new_position = exit_portal.position + exit_portal.normal * (obj.radius + 0.5)
            
            # Ensure the new velocity is valid (no NaN values)
            if np.isnan(new_velocity).any() or np.isinf(new_velocity).any():
                print("Warning: Invalid object velocity calculated. Using fallback velocity.")
                new_velocity = exit_portal.normal * 3.0
            
            # Update object properties
            obj.position = new_position
            obj.velocity = new_velocity
            
            # Adjust the object's position further away from the exit portal
            # to prevent immediate re-entry and avoid clipping
            obj.position += exit_portal.normal * 0.5  # Increased offset for better clearance
            
            # Perform a collision check at the new position to ensure we're not in a wall
            collision, normal = self.current_level.check_wall_collisions(obj.position, obj.radius)
            if collision:
                # If we're in a wall, adjust position along the normal
                adjustment = normal * 0.5  # Increased adjustment
                obj.position += adjustment
                
                # Also adjust velocity to prevent moving into the wall
                dot_product = np.dot(obj.velocity, normal)
                if dot_product < 0:  # If moving toward the wall
                    # Remove the component of velocity going into the wall
                    obj.velocity -= dot_product * np.array(normal)
                    
                    # Add a larger upward component to help objects clear obstacles
                    obj.velocity[1] += 3.0  # Increased upward boost
                    
                    # Add a random component to help break symmetry in case of getting stuck
                    obj.velocity[0] += (np.random.random() - 0.5) * 1.0  # Increased randomness
                    obj.velocity[2] += (np.random.random() - 0.5) * 1.0  # Increased randomness
        
        except Exception as e:
            print(f"Error during object teleportation: {e}")
            import traceback
            traceback.print_exc()
            
            # Emergency fallback - just move the object to a safe position
            # This prevents crashes even if the teleportation math fails
            obj.position = exit_portal.position + exit_portal.normal * (obj.radius + 1.0)  # Increased safety margin
            obj.velocity = exit_portal.normal * 4.0  # Give it a stronger push outward

    def update(self):
        try:
            # Calculate delta time
            current_time = time.time()
            dt = min(current_time - self.last_time, 0.1)  # Cap dt to avoid large jumps
            self.last_time = current_time
    
            # Update FPS counter
            self.fps_counter += 1
            self.fps_timer += dt
            if self.fps_timer >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_timer = 0
                
                # Update caption with level information
                if self.level_complete:
                    pygame.display.set_caption(f"Portal Remake - Level {self.current_level_index + 1} COMPLETE! Transitioning to next level... - FPS: {self.fps}")
                else:
                    pygame.display.set_caption(f"Portal Remake - Level {self.current_level_index + 1} - FPS: {self.fps}")
    
            # Update player
            if self.player:
                self.player.update(dt, self.current_level)
            
            # Update pickup blocks
            for block in self.pickup_blocks:
                if not block.being_carried:
                    block.update(dt, self.current_level)
            
            # Update held block position
            if self.holding_block and self.held_block:
                self.update_held_block()
                
            # Check for block-to-block collisions
            # This is done after individual updates to ensure all blocks have updated positions
            for i, block1 in enumerate(self.pickup_blocks):
                if block1.being_carried:
                    continue
                    
                # Check collisions with other blocks
                for j, block2 in enumerate(self.pickup_blocks[i+1:], i+1):
                    if block2.being_carried:
                        continue
                        
                    # Calculate distance between block centers
                    distance = np.linalg.norm(block1.position - block2.position)
                    
                    # If blocks are overlapping
                    min_distance = block1.size + block2.size
                    if distance < min_distance:
                        # Calculate collision normal (direction from block2 to block1)
                        if distance < 0.0001:  # Avoid division by zero
                            # If blocks are exactly at the same position, push in a random direction
                            collision_normal = np.array([1.0, 0.0, 0.0])  # Default to X axis
                        else:
                            collision_normal = (block1.position - block2.position) / distance
                        
                        # Calculate overlap amount
                        overlap = min_distance - distance
                        
                        # Push blocks apart based on their relative masses
                        total_mass = block1.mass + block2.mass
                        block1_ratio = block2.mass / total_mass
                        block2_ratio = block1.mass / total_mass
                        
                        # Move blocks away from collision
                        if not block1.on_ground:
                            block1.position += collision_normal * overlap * block1_ratio * 0.5
                        if not block2.on_ground:
                            block2.position -= collision_normal * overlap * block2_ratio * 0.5
                        
                        # Calculate relative velocity along collision normal
                        relative_velocity = np.dot(block1.velocity - block2.velocity, collision_normal)
                        
                        # Only apply impulse if blocks are moving toward each other
                        if relative_velocity < 0:
                            # Calculate impulse scalar
                            restitution = min(block1.restitution, block2.restitution)
                            impulse = -(1 + restitution) * relative_velocity
                            impulse /= (1/block1.mass + 1/block2.mass)
                            
                            # Apply impulse to velocities
                            block1.velocity += (impulse / block1.mass) * collision_normal
                            block2.velocity -= (impulse / block2.mass) * collision_normal
                            
                            # Check if either block is now on the ground
                            if collision_normal[1] > 0.7:  # Normal pointing mostly up for block1
                                block1.on_ground = True
                                block1.velocity[1] = 0  # Stop vertical movement
                            elif collision_normal[1] < -0.7:  # Normal pointing mostly up for block2
                                block2.on_ground = True
                                block2.velocity[1] = 0  # Stop vertical movement
            
            # Update buttons
            all_objects = [self.player] + self.pickup_blocks
            for button in self.buttons:
                button.update(all_objects)
                
            # Update doors and check for level completion via door
            ready_for_next_level = self.current_level.update(dt, self.player.position, self.player.radius)
            
            # Check for level completion via goal collision (direct method)
            if self.current_level.check_goal_collision(self.player.position, self.player.radius):
                ready_for_next_level = True
                print("Player reached the goal directly!")
            
            # If player has passed through exit door and timer has elapsed, mark level as complete
            if ready_for_next_level and not self.level_complete:
                self.level_complete = True
                print("Level complete! Automatically transitioning to next level...")
                if self.level_complete_sound:
                    self.level_complete_sound.play()
                
                # Display completion message
                pygame.display.set_caption(f"Portal Remake - Level {self.current_level_index + 1} COMPLETE! Transitioning to next level...")
                
                # Automatically transition to next level after a short delay
                # This creates a seamless transition without requiring the player to press N
                self.level_transition_timer = 0
                self.auto_transition = True
                
                # Immediately go to next level for truly seamless transition
                self.next_level_timer = 0.5  # Short delay before transition
            
            # Handle automatic level transition
            if hasattr(self, 'auto_transition') and self.auto_transition:
                if hasattr(self, 'next_level_timer'):
                    self.next_level_timer -= dt
                    if self.next_level_timer <= 0:
                        self.next_level()
                        self.auto_transition = False
    
            # Check for portal transport (player and objects)
            self.check_portal_transport()
    
            # Check for level completion via goal collision (legacy method)
            if self.current_level.check_goal_collision(self.player.position, self.player.radius):
                if not self.level_complete:
                    self.level_complete = True
                    print("Level complete! Press N to go to the next level.")
                    if self.level_complete_sound:
                        self.level_complete_sound.play()
                    
                    # Display completion message
                    pygame.display.set_caption(f"Portal Remake - Level {self.current_level_index + 1} COMPLETE! Press N for next level")
        except Exception as e:
            print(f"Error in update method: {e}")
            import traceback
            traceback.print_exc()

    def render(self):
        """
        Render the game using the dedicated renderer module.
        This method delegates all rendering to the renderer module,
        which handles portal rendering, level rendering, and UI elements.
        """
        # Use the dedicated renderer to render the entire scene
        renderer.render_scene(self)
        
        # Draw HUD (which contains game-specific UI elements)
        self.draw_hud()

    # The setup_lighting method has been moved to the renderer module
    # This comment is kept here to document the change

    def draw_hud(self):
        # Draw crosshair
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Draw level completion or transition messages
        show_message = False
        message_lines = []
        message_color = (0, 255, 0)  # Default green
        
        # Check if level is complete
        if self.level_complete:
            show_message = True
            message_lines = ["Level Complete!", "Loading next level..."]
        # Check if player has entered the exit door but level isn't complete yet
        elif self.current_level.player_entered_exit and self.current_level.exit_door and self.current_level.exit_door.player_entered:
            show_message = True
            message_lines = ["Entering exit...", "Door will close behind you"]
            message_color = (255, 215, 0)  # Gold color
        
        if show_message:
            # Create a semi-transparent background for the message
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(0.0, 0.0, 0.0, 0.7)
            glBegin(GL_QUADS)
            glVertex2f(SCREEN_WIDTH / 2 - 200, SCREEN_HEIGHT / 2 - 50)
            glVertex2f(SCREEN_WIDTH / 2 + 200, SCREEN_HEIGHT / 2 - 50)
            glVertex2f(SCREEN_WIDTH / 2 + 200, SCREEN_HEIGHT / 2 + 50)
            glVertex2f(SCREEN_WIDTH / 2 - 200, SCREEN_HEIGHT / 2 + 50)
            glEnd()
            
            # Draw text using pygame's font rendering
            # We need to temporarily exit OpenGL mode to render text
            pygame.display.flip()  # Ensure the OpenGL buffer is displayed
            
            # Create a temporary surface for text rendering
            text_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            font = pygame.font.Font(None, 36)
            
            # Render the message lines
            text1 = font.render(message_lines[0], True, message_color)
            text2 = font.render(message_lines[1], True, (255, 255, 255))
            
            # Position the text
            text_rect1 = text1.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
            text_rect2 = text2.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            
            # Blit the text onto the surface
            text_surface.blit(text1, text_rect1)
            text_surface.blit(text2, text_rect2)
            
            # Convert the surface to an OpenGL texture
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            text_width, text_height = text_surface.get_size()
            
            # Create and bind the texture
            text_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, text_texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width, text_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            
            # Draw the texture
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(1.0, 1.0, 1.0, 1.0)
            
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(0, 0)
            glTexCoord2f(1, 0); glVertex2f(SCREEN_WIDTH, 0)
            glTexCoord2f(1, 1); glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT)
            glTexCoord2f(0, 1); glVertex2f(0, SCREEN_HEIGHT)
            glEnd()
            
            # Clean up
            glDeleteTextures(1, [text_texture])
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_BLEND)

        # Draw simple crosshair
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        glVertex2f(SCREEN_WIDTH / 2 - 10, SCREEN_HEIGHT / 2)
        glVertex2f(SCREEN_WIDTH / 2 + 10, SCREEN_HEIGHT / 2)
        glVertex2f(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 10)
        glVertex2f(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 10)
        glEnd()

        # Draw level complete message if applicable
        if self.level_complete:
            # This is a very simple text rendering - in a real game you'd use proper text rendering
            glColor3f(0.0, 1.0, 0.0)
            glBegin(GL_QUADS)
            glVertex2f(SCREEN_WIDTH / 2 - 100, SCREEN_HEIGHT / 2 - 50)
            glVertex2f(SCREEN_WIDTH / 2 + 100, SCREEN_HEIGHT / 2 - 50)
            glVertex2f(SCREEN_WIDTH / 2 + 100, SCREEN_HEIGHT / 2 + 50)
            glVertex2f(SCREEN_WIDTH / 2 - 100, SCREEN_HEIGHT / 2 + 50)
            glEnd()

        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glEnable(GL_DEPTH_TEST)

    def run(self):
        # Create a local clock if the global one is not available
        game_clock = pygame.time.Clock()

        while self.running:
            # Handle events
            self.handle_events()

            # Update game state
            self.update()

            # Render frame
            self.render()

            # Cap frame rate
            game_clock.tick(MAX_FPS)

# Performance monitoring class - now uses our enhanced implementation
# This class is maintained for backward compatibility
# New code should use the performance_monitor module directly
class PerformanceMonitor:
    """
    Legacy performance monitoring class that delegates to the new implementation.
    
    This class is maintained for backward compatibility with existing code.
    New code should use the performance_monitor module directly.
    """
    def __init__(self, window_size=60):
        """Initialize by setting up references to the new implementation."""
        # We'll just use the global perf_monitor instance
        self.frame_times = perf_monitor.frame_times
        self.window_size = perf_monitor.window_size
        self.last_time = perf_monitor.last_time
        self.fps_history = perf_monitor.fps_history
        self.adaptive_settings = perf_monitor.adaptive_settings

    def update(self):
        """Update performance metrics by delegating to the new implementation."""
        # Start timing the frame if not already started
        if not hasattr(perf_monitor, 'frame_start_time'):
            perf_monitor.start_frame()
            
        # End the frame and get the frame time
        return perf_monitor.end_frame()

    def get_avg_fps(self):
        """Get average FPS by delegating to the new implementation."""
        return perf_monitor.get_avg_fps()

    def adjust_settings(self):
        """Adjust graphics settings by delegating to the new implementation."""
        perf_monitor.adjust_settings()
        
    def print_performance_report(self):
        """Print a detailed performance report."""
        perf_monitor.print_performance_report()

# Main function
def main():
    try:
        # Make sure assets directories exist
        os.makedirs(os.path.join("assets", "textures"), exist_ok=True)
        os.makedirs(os.path.join("assets", "sounds"), exist_ok=True)
        
        print("\n" + "="*50)
        print("Portal Game with Enhanced Graphics")
        print("="*50)
        print("Features:")
        print("- Tessellation for more detailed geometry")
        print("- Enhanced lighting for realistic reflections")
        print("- Dynamic Level of Detail (LOD) for optimal performance")
        print("- Adaptive graphics settings based on performance")
        print("- Fullscreen mode support (F11)")
        print("\nGame Mechanics:")
        print("- Fully functional portals for teleportation")
        print("- Improved portal rendering with realistic view through portals")
        print("- Portals cannot overlap with each other")
        print("- Pickup blocks that can be placed on buttons")
        print("- Physics-based interactions between objects")
        print("- Doors that open when buttons are pressed")
        print("- Complete levels by bringing cubes to buttons to open doors")
        print("- Doors close automatically after you pass through them")
        print("- Next level loads automatically when you go through the exit door")
        print("- Goal areas are protected - you must solve the puzzle to reach them")
        print("\nControls:")
        print("- WASD: Move")
        print("- Mouse: Look around")
        print("- Space: Jump")
        print("- Left Click: Place Blue Portal")
        print("- Right Click: Place Orange Portal")
        print("- E: Pick up/drop blocks")
        print("- F: Throw held block")
        print("- R: Reset Level")
        print("- ESC: Exit Game")
        print("\nGraphics Controls:")
        print("- L: Toggle enhanced lighting")
        print("- T: Toggle tessellation")
        print("- +/-: Increase/decrease tessellation level")
        print("- F11: Toggle fullscreen mode")
        print("="*50 + "\n")

        # Create a non-OpenGL display for splash and loading screens
        temp_display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.HWSURFACE)
        pygame.display.set_caption("Portal Game")
        
        # Show splash screen
        splash = SplashScreen(temp_display)
        result = splash.run()
        
        # Handle splash screen result
        if result == "exit":
            pygame.quit()
            sys.exit()
            
        # Create loading screen
        loading = LoadingScreen(temp_display)
        
        # Initialize performance monitor (using our global instance)
        # This is kept for backward compatibility
        legacy_perf_monitor = PerformanceMonitor()
        
        # Show initial loading screen
        loading.update(0.1, "Initializing game engine...")
        loading.draw()
        
        # Pre-initialize some game components
        try:
            # Preload common assets
            loading.update(0.3, "Loading textures...")
            loading.draw()
            
            # Define common textures to preload
            common_textures = [
                "concrete-wall.jpeg", "metal-floor.png", "portal-blue.png", 
                "portal-orange.png", "button.png", "cube.png"
            ]
            
            # Preload textures with the asset manager
            asset_manager.preload_assets(texture_list=common_textures)
            time.sleep(0.3)  # Reduced loading time
            
            loading.update(0.5, "Loading sounds...")
            loading.draw()
            
            # Define common sounds to preload
            common_sounds = [
                "portal_open.wav", "portal_enter.wav", "portal_exit.wav",
                "button_press.wav", "button_release.wav"
            ]
            
            # Preload sounds with the asset manager
            asset_manager.preload_assets(sound_list=common_sounds)
            time.sleep(0.3)  # Reduced loading time
            
            loading.update(0.7, "Setting up physics...")
            loading.draw()
            time.sleep(0.3)  # Reduced loading time
            
            loading.update(0.9, "Preparing level...")
            loading.draw()
            time.sleep(0.3)  # Reduced loading time
            
            loading.update(1.0, "Ready to play!")
            loading.draw()
            time.sleep(0.3)  # Reduced loading time
        except Exception as e:
            print(f"Error during loading: {e}")
        
        # Switch to OpenGL mode for the actual game
        display_flags = DOUBLEBUF | OPENGL | pygame.HWSURFACE
        if FULLSCREEN:
            display_flags |= pygame.FULLSCREEN
        display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display_flags)
        pygame.display.set_caption("Portal Remake - Enhanced Graphics")
        
        # Set up OpenGL again after mode change
        glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV, SCREEN_WIDTH / SCREEN_HEIGHT, NEAR_PLANE, FAR_PLANE)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Re-enable lighting if it was enabled
        if ENABLE_LIGHTING:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_LIGHT1)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
            
            # Reset light positions and properties
            light_position = [5.0, 5.0, 5.0, 1.0]
            ambient_light = [0.3, 0.3, 0.3, 1.0]
            diffuse_light = [0.8, 0.8, 0.8, 1.0]
            specular_light = [1.0, 1.0, 1.0, 1.0]
            
            glLightfv(GL_LIGHT0, GL_POSITION, light_position)
            glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
            glLightfv(GL_LIGHT0, GL_SPECULAR, specular_light)
            
            light_position2 = [0.0, 10.0, 0.0, 1.0]
            ambient_light2 = [0.1, 0.1, 0.1, 1.0]
            diffuse_light2 = [0.4, 0.4, 0.4, 1.0]
            specular_light2 = [0.2, 0.2, 0.2, 1.0]
            
            glLightfv(GL_LIGHT1, GL_POSITION, light_position2)
            glLightfv(GL_LIGHT1, GL_AMBIENT, ambient_light2)
            glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse_light2)
            glLightfv(GL_LIGHT1, GL_SPECULAR, specular_light2)
            
            glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0)
            glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
            glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.01)
        
        # Initialize game after switching to OpenGL mode
        game = PortalGame()
        
        # Set up OpenGL again after mode change
        glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV, SCREEN_WIDTH / SCREEN_HEIGHT, NEAR_PLANE, FAR_PLANE)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Re-enable lighting if it was enabled
        if ENABLE_LIGHTING:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_LIGHT1)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
            
            # Reset light positions and properties
            light_position = [5.0, 5.0, 5.0, 1.0]
            ambient_light = [0.3, 0.3, 0.3, 1.0]
            diffuse_light = [0.8, 0.8, 0.8, 1.0]
            specular_light = [1.0, 1.0, 1.0, 1.0]
            
            glLightfv(GL_LIGHT0, GL_POSITION, light_position)
            glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
            glLightfv(GL_LIGHT0, GL_SPECULAR, specular_light)
            
            light_position2 = [0.0, 10.0, 0.0, 1.0]
            ambient_light2 = [0.1, 0.1, 0.1, 1.0]
            diffuse_light2 = [0.4, 0.4, 0.4, 1.0]
            specular_light2 = [0.2, 0.2, 0.2, 1.0]
            
            glLightfv(GL_LIGHT1, GL_POSITION, light_position2)
            glLightfv(GL_LIGHT1, GL_AMBIENT, ambient_light2)
            glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse_light2)
            glLightfv(GL_LIGHT1, GL_SPECULAR, specular_light2)
            
            glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0)
            glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
            glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.01)

        # Add key handlers for graphics settings
        original_handle_events = game.handle_events

        def enhanced_handle_events():
            # Store the original pygame.event.get function
            original_event_get = pygame.event.get
            
            # Create a list to store events
            events = original_event_get()
            
            # Replace pygame.event.get with a function that returns our stored events
            def custom_event_get():
                return events.copy()
            
            # Temporarily replace pygame.event.get
            pygame.event.get = custom_event_get
            
            try:
                # Call original handle_events with our stored events
                original_handle_events()
                
                # Now process the same events for our enhanced functionality
                global ENABLE_LIGHTING, ENABLE_TESSELLATION, TESSELLATION_LEVEL, FULLSCREEN
                
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_l:
                            # Toggle enhanced lighting
                            config.toggle_lighting()
                            print(f"Enhanced lighting: {'Enabled' if config.enable_lighting else 'Disabled'}")
                            
                            # Update OpenGL state based on new setting
                            if config.enable_lighting:
                                glEnable(GL_LIGHTING)
                                glEnable(GL_LIGHT0)
                                glEnable(GL_LIGHT1)
                            else:
                                glDisable(GL_LIGHTING)
    
                        elif event.key == pygame.K_t:
                            # Toggle tessellation
                            config.toggle_tessellation()
                            print(f"Tessellation: {'Enabled' if config.enable_tessellation else 'Disabled'}")
    
                        elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                            # Increase tessellation level
                            new_level = config.increase_tessellation()
                            print(f"Tessellation level: {new_level}")
    
                        elif event.key == pygame.K_MINUS:
                            # Decrease tessellation level
                            new_level = config.decrease_tessellation()
                            print(f"Tessellation level: {new_level}")
                            
                        elif event.key == pygame.K_F11:
                            # Toggle fullscreen mode
                            config.toggle_fullscreen()
                            print(f"Fullscreen mode: {'Enabled' if config.fullscreen else 'Disabled'}")
                            
                            # Update display mode
                            display_flags = DOUBLEBUF | OPENGL | pygame.HWSURFACE
                            if config.fullscreen:
                                display_flags |= pygame.FULLSCREEN
                            
                            # Save current OpenGL state
                            glPushAttrib(GL_ALL_ATTRIB_BITS)
                            
                            # Reset display
                            pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), display_flags)
                            
                            # Restore OpenGL state
                            glPopAttrib()
                            
                            # Reset viewport and projection
                            glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
                            glMatrixMode(GL_PROJECTION)
                            glLoadIdentity()
                            gluPerspective(FOV, SCREEN_WIDTH / SCREEN_HEIGHT, NEAR_PLANE, FAR_PLANE)
                            glMatrixMode(GL_MODELVIEW)
                            glLoadIdentity()
            finally:
                # Restore the original pygame.event.get function
                pygame.event.get = original_event_get

        # Replace event handler with enhanced version
        game.handle_events = enhanced_handle_events

        # Modify the game loop to include performance monitoring
        original_run = game.run

        def enhanced_run():
            # Create a local clock if the global one is not available
            game_clock = pygame.time.Clock()

            # Display initial graphics settings
            print(f"Initial graphics settings:")
            print(f"- Enhanced lighting: {'Enabled' if config.enable_lighting else 'Disabled'}")
            print(f"- Tessellation: {'Enabled' if config.enable_tessellation else 'Disabled'}")
            print(f"- Tessellation level: {config.tessellation_level}")
            print(f"- Dynamic LOD: {'Enabled' if config.dynamic_lod else 'Disabled'}")
            
            # Performance report key
            perf_report_key = pygame.K_p
            last_report_time = 0
            report_cooldown = 5.0  # seconds between performance reports

            while game.running:
                # Start frame timing
                perf_monitor.start_frame()
                
                # Start input timing
                perf_monitor.start_section('input')
                
                # Handle events
                game.handle_events()
                
                # End input timing
                perf_monitor.end_section('input')
                
                # Start physics timing
                perf_monitor.start_section('physics')
                
                # Update game state
                game.update()
                
                # End physics timing
                perf_monitor.end_section('physics')
                
                # Start render timing
                perf_monitor.start_section('render')
                
                                           # Render frame
                game.render()
                
                # End render timing
                perf_monitor.end_section('render')
                
                # End frame timing
                perf_monitor.end_frame()

                # Update FPS display with more detailed info
                avg_fps = perf_monitor.get_avg_fps()
                physics_stats = perf_monitor.get_section_stats('physics')
                render_stats = perf_monitor.get_section_stats('render')
                
                # Create a more informative caption
                pygame.display.set_caption(
                    f"Portal Remake - FPS: {avg_fps:.1f} - " +
                    f"Physics: {physics_stats['avg']:.1f}ms - " +
                    f"Render: {render_stats['avg']:.1f}ms - " +
                    f"Lighting: {'On' if config.enable_lighting else 'Off'} - " +
                    f"Tess: {config.tessellation_level if config.enable_tessellation else 'Off'}"
                )
                
                # Check for performance report key (P)
                keys = pygame.key.get_pressed()
                current_time = time.time()
                if keys[perf_report_key] and current_time - last_report_time > report_cooldown:
                    perf_monitor.print_performance_report()
                    last_report_time = current_time

                # Cap frame rate
                game_clock.tick(config.max_fps)

        # Replace run method with enhanced version
        game.run = enhanced_run

        # Run game loop
        game.run()

    except Exception as e:
        print(f"Error running game: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        pygame.quit()
        sys.exit()

# Run the game
if __name__ == "__main__":
    main()