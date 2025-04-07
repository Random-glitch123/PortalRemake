import pygame
import sys
import math
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import os

# Constants for optimization
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
FOV = 60
NEAR_PLANE = 0.1
FAR_PLANE = 100.0
GRAVITY = 9.8
JUMP_FORCE = 5.0
PLAYER_HEIGHT = 1.8
PLAYER_RADIUS = 0.5
MOUSE_SENSITIVITY = 0.2
MOVEMENT_SPEED = 5.0
PORTAL_DISTANCE = 100.0
PORTAL_RADIUS = 0.7
PORTAL_SEGMENTS = 20  # Lower for better performance
MAX_FPS = 60

# Initialize Pygame and OpenGL
pygame.init()
pygame.mixer.init()
display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Portal Remake - Optimized")
clock = pygame.time.Clock()

# Set up OpenGL
glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(FOV, SCREEN_WIDTH / SCREEN_HEIGHT, NEAR_PLANE, FAR_PLANE)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
glEnable(GL_DEPTH_TEST)
glEnable(GL_CULL_FACE)
glCullFace(GL_BACK)

# Texture loading function
def load_texture(filename):
    texture_path = os.path.join("assets", "textures", filename)
    try:
        texture_surface = pygame.image.load(texture_path)
        texture_data = pygame.image.tostring(texture_surface, "RGBA", 1)
        width, height = texture_surface.get_size()
        
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        return texture_id
    except pygame.error as e:
        print(f"Could not load texture {filename}: {e}")
        # Return a default texture ID (checkerboard pattern)
        return create_default_texture()

# Create a default checkerboard texture when texture files are missing
def create_default_texture():
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

# Sound loading function
def load_sound(filename):
    sound_path = os.path.join("assets", "sounds", filename)
    try:
        return pygame.mixer.Sound(sound_path)
    except pygame.error as e:
        print(f"Could not load sound {filename}: {e}")
        return None

# Vector operations
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

def cross_product(v1, v2):
    return np.cross(v1, v2)

def reflect_vector(v, normal):
    return v - 2 * dot_product(v, normal) * np.array(normal)

# Wall class
class Wall:
    def __init__(self, x, y, z, width, height, depth, texture_id=None, portal_surface=True):
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth
        self.texture_id = texture_id
        self.portal_surface = portal_surface
        
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
        if self.texture_id:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
        else:
            glDisable(GL_TEXTURE_2D)
        
        for i, face in enumerate(self.faces):
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
        
        glDisable(GL_TEXTURE_2D)
    
    def check_collision(self, pos, radius):
        # Simple AABB collision detection
        half_width = self.width / 2 + radius
        half_height = self.height / 2 + radius
        half_depth = self.depth / 2 + radius
        
        return (pos[0] >= self.x - half_width and pos[0] <= self.x + half_width and
                pos[1] >= self.y - half_height and pos[1] <= self.y + half_height and
                pos[2] >= self.z - half_depth and pos[2] <= self.z + half_depth)
    
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
            ndotu = dot_product(normal, ray_direction)
            if abs(ndotu) < 0.0001:  # Ray is parallel to face
                continue
            
            # Calculate intersection point
            w = np.array(ray_origin) - np.array(v0)
            t = -dot_product(normal, w) / ndotu
            
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
        self.radius = PORTAL_RADIUS
        
        # Calculate portal orientation
        self.up = np.array([0, 1, 0])
        if abs(dot_product(self.normal, self.up)) > 0.99:
            # If normal is close to up vector, use a different up vector
            self.up = np.array([0, 0, 1])
        
        self.right = normalize(cross_product(self.up, self.normal))
        self.up = normalize(cross_product(self.normal, self.right))
    
    def draw(self):
        if not self.active:
            return
        
        # Save current matrix
        glPushMatrix()
        
        # Translate to portal position
        glTranslatef(self.position[0], self.position[1], self.position[2])
        
        # Rotate to align with portal normal
        # This is a simplified rotation that might not work for all cases
        # For a more robust solution, you'd use a rotation matrix
        
        # Draw portal rim
        if self.texture_id:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        glBegin(GL_TRIANGLE_FAN)
        
        # Set color based on portal type
        if self.color == "blue":
            glColor3f(0.0, 0.5, 1.0)
        else:  # Orange
            glColor3f(1.0, 0.5, 0.0)
        
        # Center point
        glVertex3f(0, 0, 0.01)  # Slightly offset to avoid z-fighting
        
        # Draw portal rim
        for i in range(PORTAL_SEGMENTS + 1):
            angle = 2.0 * math.pi * i / PORTAL_SEGMENTS
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            
            # Transform point to portal orientation
            point = x * self.right + y * self.up
            
            if self.texture_id:
                # Calculate texture coordinates
                tx = 0.5 + 0.5 * math.cos(angle)
                ty = 0.5 + 0.5 * math.sin(angle)
                glTexCoord2f(tx, ty)
            
            glVertex3f(point[0], point[1], point[2] + 0.01)  # Slightly offset
        
        glEnd()
        
        if self.texture_id:
            glDisable(GL_TEXTURE_2D)
        
        # Restore matrix
        glPopMatrix()
    
    def check_collision(self, pos, radius):
        # Check if position is close to portal plane
        dist_to_plane = abs(dot_product(pos - self.position, self.normal))
        if dist_to_plane > radius:
            return False
        
        # Project position onto portal plane
        projected_pos = pos - dot_product(pos - self.position, self.normal) * self.normal
        
        # Calculate distance from portal center to projected position
        dist_to_center = np.linalg.norm(projected_pos - self.position)
        
        # Check if within portal radius
        return dist_to_center <= self.radius

# Level class
class Level:
    def __init__(self, walls, goal_pos, player_start):
        self.walls = walls
        self.goal_pos = goal_pos
        self.player_start = player_start
        self.goal_radius = 0.5
    
    def draw(self):
        # Draw walls
        for wall in self.walls:
            wall.draw()
        
        # Draw goal
        glPushMatrix()
        glTranslatef(self.goal_pos[0], self.goal_pos[1], self.goal_pos[2])
        glColor3f(0.0, 1.0, 0.0)  # Green
        
        # Draw a simple cube for the goal
        try:
            from OpenGL.GLUT import *
            glutSolidCube(self.goal_radius * 2)
        except:
            # Fallback if GLUT is not available
            glBegin(GL_QUADS)
            # Draw a cube manually
            r = self.goal_radius
            vertices = [
                [-r, -r, -r], [r, -r, -r], [r, r, -r], [-r, r, -r],
                [-r, -r, r], [r, -r, r], [r, r, r], [-r, r, r]
            ]
            faces = [
                [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]
            ]
            for face in faces:
                for vertex in face:
                    glVertex3fv(vertices[vertex])
            glEnd()
        
        glPopMatrix()
    
    def check_wall_collisions(self, pos, radius):
        for wall in self.walls:
            if wall.check_collision(pos, radius):
                normal = wall.get_collision_normal(pos)
                return True, normal
        
        return False, None
    
    def check_goal_collision(self, pos, radius):
        dist = np.linalg.norm(np.array(pos) - np.array(self.goal_pos))
        return dist <= (radius + self.goal_radius)
    
    def ray_cast(self, ray_origin, ray_direction, max_distance=PORTAL_DISTANCE):
        closest_intersection = None
        closest_distance = max_distance
        intersection_normal = None
        intersected_wall = None
        
        for wall in self.walls:
            intersection, normal, distance = wall.ray_intersection(ray_origin, ray_direction)
            
            if intersection is not None and distance < closest_distance:
                closest_intersection = intersection
                closest_distance = distance
                intersection_normal = normal
                intersected_wall = wall
        
        return closest_intersection, intersection_normal, closest_distance, intersected_wall

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
        # Apply gravity if not on ground
        if not self.on_ground:
            self.velocity[1] -= GRAVITY * dt
        
        # Update position based on velocity
        new_position = self.position + self.velocity * dt
        
        # Check for collisions with walls
        collision, normal = level.check_wall_collisions(new_position, self.radius)
        
        if collision:
            # Reflect velocity off wall (with damping)
            self.velocity = reflect_vector(self.velocity, normal) * 0.5
            
            # Check if we're on the ground
            if normal[1] > 0.7:  # Normal pointing mostly up
                self.on_ground = True
                self.velocity[1] = 0  # Stop vertical movement
            
            # Adjust position to avoid penetration
            new_position = self.position + self.velocity * dt
        else:
            self.on_ground = False
        
        # Update position
        self.position = new_position
    
    def jump(self):
        if self.on_ground:
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
        
        # Apply movement
        self.velocity[0] = move_dir[0] * speed
        self.velocity[2] = move_dir[2] * speed

# Game class
class PortalGame:
    def __init__(self):
        # Load textures
        self.wall_texture = load_texture("wall_concrete.png")
        self.floor_texture = load_texture("floor_concrete.png")
        self.ceiling_texture = load_texture("ceiling_tile.png")
        self.portal_blue_texture = load_texture("portal_blue_rim.png")
        self.portal_orange_texture = load_texture("portal_orange_rim.png")
        
        # Load sounds
        self.portal_blue_sound = load_sound("portal_open_blue.wav")
        self.portal_orange_sound = load_sound("portal_open_orange.wav")
        self.portal_enter_sound = load_sound("portal_enter.wav")
        self.level_complete_sound = load_sound("level_complete.wav")
        
        # Create player
        self.player = None
        
        # Create portals
        self.blue_portal = None
        self.orange_portal = None
        
        # Create levels
        self.levels = [self.create_level1(), self.create_level2(), self.create_level3()]
        self.current_level_index = 0
        self.current_level = self.levels[self.current_level_index]
        
        # Reset player position
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
            
            # Goal platform
            Wall(7, 0, -7, 4, 0.5, 4, self.floor_texture),   # Goal platform
        ]
        
        goal_pos = (7, 1, -7)  # Position of the goal
        player_start = (0, 0, 8)  # Starting position of the player
        
        return Level(walls, goal_pos, player_start)
    
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
        ]
        
        goal_pos = (0, 7, 0)  # Goal on the highest platform
        player_start = (-12, 0, 12)  # Starting in a corner
        
        return Level(walls, goal_pos, player_start)
    
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
            
            # Central structure
            Wall(0, 4, 0, 12, 8, 12, self.wall_texture, False), # Central non-portal structure
            Wall(0, 9, 0, 6, 0.5, 6, self.floor_texture),       # Top platform with goal
            
            # Ramps and additional structures
            Wall(-15, 1, 10, 8, 4, 0.5, self.wall_texture),     # Wall 1
            Wall(15, 1, -10, 8, 4, 0.5, self.wall_texture),     # Wall 2
            Wall(0, 0, -15, 30, 0.5, 4, self.floor_texture),    # Extended floor section
        ]
        
        goal_pos = (0, 10, 0)  # Goal on top of central structure
        player_start = (0, 0, 18)  # Starting at front
        
        return Level(walls, goal_pos, player_start)
    
    def reset_player(self):
        self.player = Player(self.current_level.player_start)
        self.blue_portal = None
        self.orange_portal = None
        self.level_complete = False
    
    def next_level(self):
        self.current_level_index = (self.current_level_index + 1) % len(self.levels)
        self.current_level = self.levels[self.current_level_index]
        self.reset_player()
    
    def handle_events(self):
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
                elif event.key == pygame.K_n and self.level_complete:
                    self.next_level()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click - Blue portal
                    self.create_portal("blue")
                elif event.button == 3:  # Right click - Orange portal
                    self.create_portal("orange")
        
        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        
        # Reset movement velocity
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
        self.player.yaw += mouse_dx *