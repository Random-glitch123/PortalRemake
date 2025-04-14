import numpy as np
import math

# Constants for physics calculations
GRAVITY = 9.8
JUMP_FORCE = 5.0
PLAYER_HEIGHT = 1.8
PLAYER_RADIUS = 0.5
MOUSE_SENSITIVITY = 0.2
MOVEMENT_SPEED = 5.0

# Anti-jitter physics constants
COLLISION_TOLERANCE = 0.02  # Increased tolerance to prevent jitter at collision boundaries
WALL_SLIDE_FACTOR = 0.9     # Increased to reduce sliding friction against walls
STEP_HEIGHT = 0.4           # Increased to make climbing obstacles easier
GROUND_SNAP_DISTANCE = 0.15 # Increased to improve ground detection
STABILIZATION_FACTOR = 0.8  # Increased for more stable movement
PENETRATION_CORRECTION = 0.3 # How much to correct penetration (0-1)

# Physics constants for cubes
CUBE_FRICTION = 0.7         # Increased for more stable stacking
CUBE_RESTITUTION = 0.2      # Reduced further for less bouncing
CUBE_TERMINAL_VELOCITY = -15.0 # Reduced for more control
CUBE_MASS = 8.0             # Increased for more stability

# Vector operations
def normalize(v):
    """
    Normalize a vector to unit length.
    Handles numpy arrays and lists safely.
    
    Args:
        v: Vector to normalize (numpy array or list)
        
    Returns:
        Normalized vector with unit length
    """
    try:
        # Convert to numpy array if it's not already
        v_array = np.array(v, dtype=float)
        
        # Calculate the norm
        norm = np.linalg.norm(v_array)
        
        # Avoid division by zero
        if norm < 0.0001:
            return np.zeros_like(v_array)
            
        # Return normalized vector
        return v_array / norm
    except Exception as e:
        print(f"Error in normalize function: {e}")
        # Return a safe default
        if isinstance(v, (list, np.ndarray)) and len(v) >= 3:
            return np.array([0.0, 1.0, 0.0])  # Default up vector
        elif isinstance(v, (list, np.ndarray)) and len(v) == 2:
            return np.array([1.0, 0.0])  # Default right vector
        else:
            return np.array([0.0, 1.0, 0.0])  # Default fallback

def dot_product(v1, v2):
    """
    Calculate dot product between two vectors.
    Handles numpy arrays and lists safely.
    
    Args:
        v1, v2: Vectors to calculate dot product between (numpy arrays or lists)
        
    Returns:
        Dot product scalar value
    """
    try:
        # Convert to numpy arrays if they're not already
        v1_array = np.array(v1, dtype=float)
        v2_array = np.array(v2, dtype=float)
        
        # Use numpy's dot product
        return np.dot(v1_array, v2_array)
    except Exception as e:
        print(f"Error in dot_product function: {e}")
        # Return a safe default
        return 0.0

def cross_product(v1, v2):
    """
    Calculate cross product between two vectors.
    Handles numpy arrays and lists safely.
    
    Args:
        v1, v2: Vectors to calculate cross product between (numpy arrays or lists)
        
    Returns:
        Cross product vector
    """
    try:
        # Convert to numpy arrays if they're not already
        v1_array = np.array(v1, dtype=float)
        v2_array = np.array(v2, dtype=float)
        
        # Use numpy's cross product
        return np.cross(v1_array, v2_array)
    except Exception as e:
        print(f"Error in cross_product function: {e}")
        # Return a safe default
        if len(v1) >= 3 and len(v2) >= 3:
            return np.array([0.0, 1.0, 0.0])  # Default up vector
        else:
            return np.zeros(max(len(v1), len(v2)))

def reflect_vector(v, normal):
    """
    Reflect a vector across a normal.
    Handles numpy arrays and lists safely.
    
    Args:
        v: Vector to reflect (numpy array or list)
        normal: Normal vector to reflect across (numpy array or list)
        
    Returns:
        Reflected vector
    """
    try:
        # Convert to numpy arrays if they're not already
        v_array = np.array(v, dtype=float)
        normal_array = np.array(normal, dtype=float)
        
        # Normalize the normal vector
        normal_array = normalize(normal_array)
        
        # Calculate the reflection
        dot = np.dot(v_array, normal_array)
        return v_array - 2 * dot * normal_array
    except Exception as e:
        print(f"Error in reflect_vector function: {e}")
        # Return a safe default
        return v  # Return the original vector as a fallback

def bilinear_interpolate(v0, v1, v2, v3, u, v):
    """Bilinear interpolation of 3D points"""
    p0 = [(1-u)*v0[0] + u*v1[0], (1-u)*v0[1] + u*v1[1], (1-u)*v0[2] + u*v1[2]]
    p1 = [(1-u)*v3[0] + u*v2[0], (1-u)*v3[1] + u*v2[1], (1-u)*v3[2] + u*v2[2]]
    return [(1-v)*p0[0] + v*p1[0], (1-v)*p0[1] + v*p1[1], (1-v)*p0[2] + v*p1[2]]

def bilinear_interpolate_2d(t0, t1, t2, t3, u, v):
    """Bilinear interpolation of 2D texture coordinates"""
    p0 = [(1-u)*t0[0] + u*t1[0], (1-u)*t0[1] + u*t1[1]]
    p1 = [(1-u)*t3[0] + u*t2[0], (1-u)*t3[1] + u*t2[1]]
    return [(1-v)*p0[0] + v*p1[0], (1-v)*p0[1] + v*p1[1]]

# Physics calculations
def calculate_rotation_matrix(normal, up=np.array([0.0, 1.0, 0.0])):
    """
    Calculate rotation matrix that aligns with a given normal vector.
    
    Args:
        normal: The normal vector to align with
        up: The up vector to use for orientation (default: world up)
        
    Returns:
        A 4x4 rotation matrix in column-major order ready for OpenGL
    """
    # Ensure normal is normalized
    normal = np.array(normal, dtype=np.float32)
    normal = normal / np.linalg.norm(normal)
    
    # Calculate right vector
    right = np.cross(up, normal)
    if np.linalg.norm(right) < 0.001:  # If normal is parallel to up
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Use x-axis as right
    right = right / np.linalg.norm(right)
    
    # Recalculate up vector to ensure orthogonality
    up = np.cross(normal, right)
    up = up / np.linalg.norm(up)
    
    # Set up the rotation matrix in column-major order for OpenGL
    # OpenGL expects matrices in column-major order
    rotation_matrix = np.array([
        right[0], up[0], normal[0], 0,
        right[1], up[1], normal[1], 0,
        right[2], up[2], normal[2], 0,
        0, 0, 0, 1
    ], dtype=np.float32)
    
    # Reshape to 4x4 matrix for OpenGL
    # This is critical for proper use with glMultMatrixf
    rotation_matrix = rotation_matrix.reshape(4, 4)
    
    # Ensure the matrix is contiguous in memory for OpenGL
    rotation_matrix = np.ascontiguousarray(rotation_matrix)
    
    # Reshape to 4x4 matrix for OpenGL
    # This is critical for proper use with glMultMatrixf
    rotation_matrix = rotation_matrix.reshape(4, 4)
    
    # Ensure the matrix is contiguous in memory for OpenGL
    rotation_matrix = np.ascontiguousarray(rotation_matrix)
    
    return rotation_matrix

def calculate_rodrigues_rotation(vector, axis, angle):
    """Apply Rodrigues rotation formula to rotate a vector around an axis by an angle"""
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    return vector * cos_angle + \
           np.cross(axis, vector) * sin_angle + \
           axis * np.dot(axis, vector) * (1 - cos_angle)

def calculate_portal_transform(entry_portal, exit_portal, position, velocity, yaw):
    """Calculate the transformed position, velocity and yaw when going through a portal"""
    # Calculate offset from entry portal
    offset = position - entry_portal.position

    # Project offset onto portal plane
    projected_offset = offset - np.dot(offset, entry_portal.normal) * entry_portal.normal

    # Normalize the offset relative to the portal's oval dimensions
    local_x = np.dot(projected_offset, entry_portal.right)
    local_y = np.dot(projected_offset, entry_portal.up)

    # Scale coordinates to account for oval shape
    normalized_x = local_x / (entry_portal.width * 0.5)
    normalized_y = local_y / (entry_portal.height * 0.5)

    # Apply the normalized coordinates to the exit portal's dimensions
    exit_x = normalized_x * (exit_portal.width * 0.5)
    exit_y = normalized_y * (exit_portal.height * 0.5)

    # Calculate the exit offset
    exit_offset = exit_x * exit_portal.right + exit_y * exit_portal.up

    # Set new position slightly away from exit portal to avoid re-entry
    new_position = exit_portal.position + exit_offset + exit_portal.normal * (PLAYER_RADIUS + 0.1)

    # Transform velocity through portal
    entry_normal = entry_portal.normal
    exit_normal = exit_portal.normal

    # Calculate the rotation matrix between the two portals
    dot_product_portals = np.dot(entry_normal, exit_normal)
    cross_product_portals = np.cross(entry_normal, exit_normal)
    cross_magnitude = np.linalg.norm(cross_product_portals)

    if cross_magnitude > 0.001:  # If portals aren't parallel
        # Normalize the cross product
        cross_product_portals = cross_product_portals / cross_magnitude

        # Calculate rotation angle
        angle = math.acos(max(-1.0, min(1.0, dot_product_portals)))

        # Apply rotation to velocity using Rodrigues formula
        new_velocity = calculate_rodrigues_rotation(velocity, cross_product_portals, angle)
    else:
        # If portals are parallel, just reflect the velocity
        new_velocity = velocity - 2 * np.dot(velocity, entry_normal) * entry_normal

    # Adjust yaw based on portal orientation difference
    yaw_diff = math.degrees(math.atan2(exit_portal.normal[0], exit_portal.normal[2]) -
                           math.atan2(entry_normal[0], entry_normal[2]))
    new_yaw = (yaw + yaw_diff) % 360

    return new_position, new_velocity, new_yaw

def check_portal_collision(portal, position, radius):
    """Check if a position is colliding with a portal"""
    try:
        # Check if position is close to portal plane
        dist_to_plane = abs(np.dot(position - portal.position, portal.normal))

        # Increase collision detection distance to make it easier to go through portals
        # This is especially important for jumping through portals
        if dist_to_plane > radius * 3.0:  # Increased from 2.0 for better detection
            return False

        # Project position onto portal plane
        projected_pos = position - np.dot(position - portal.position, portal.normal) * portal.normal

        # Calculate distance from portal center to projected position
        offset = projected_pos - portal.position

        # Check if within elliptical portal shape
        # Transform to portal's local coordinate system
        local_x = np.dot(offset, portal.right)
        local_y = np.dot(offset, portal.up)

        # Make the collision area significantly larger than the visual portal
        # This makes it much easier to go through portals, especially when jumping
        width_factor = 1.5  # 50% wider collision area (increased from 1.25)
        height_factor = 1.5  # 50% taller collision area (increased from 1.25)
        
        # Add the radius to make it even easier to detect collisions
        effective_width = portal.width * 0.5 * width_factor + radius * 0.8
        effective_height = portal.height * 0.5 * height_factor + radius * 0.8

        # Check if point is inside ellipse: (x/a)² + (y/b)² <= 1
        # Use a more forgiving threshold
        normalized_dist = (local_x / effective_width)**2 + (local_y / effective_height)**2

        return normalized_dist <= 1.2  # More forgiving threshold (was 1.0)
    
    except Exception as e:
        print(f"Error in portal collision detection: {e}")
        # Default to no collision in case of error
        return False

# Cube collision physics
def check_cube_collision(cube_position, cube_size, position, radius):
    """
    Check if a position is colliding with a cube.
    
    Args:
        cube_position: Center position of the cube (numpy array or list)
        cube_size: Half-size of the cube (scalar)
        position: Position to check for collision (numpy array or list)
        radius: Radius of the object (player, block, etc.)
        
    Returns:
        Boolean indicating whether a collision is occurring
    """
    try:
        # Convert to numpy arrays if they're not already
        cube_pos = np.array(cube_position, dtype=float)
        pos = np.array(position, dtype=float)
        
        # Calculate distance from position to cube center
        dist = np.linalg.norm(pos - cube_pos)
        
        # Quick check - if distance is greater than sum of radii, no collision
        # Add a small buffer to prevent jittering at the boundary
        if dist > (radius + cube_size * 1.414 + COLLISION_TOLERANCE):  # 1.414 is sqrt(2) for cube diagonal
            return False
        
        # More precise check - find closest point on cube to position
        closest_point = np.array([
            max(cube_pos[0] - cube_size, min(pos[0], cube_pos[0] + cube_size)),
            max(cube_pos[1] - cube_size, min(pos[1], cube_pos[1] + cube_size)),
            max(cube_pos[2] - cube_size, min(pos[2], cube_pos[2] + cube_size))
        ])
        
        # Calculate distance from closest point to position
        dist_to_closest = np.linalg.norm(pos - closest_point)
        
        # Collision if distance is less than radius (with tolerance)
        # The tolerance helps prevent jittering when just barely touching
        return dist_to_closest <= (radius - COLLISION_TOLERANCE)
    except Exception as e:
        print(f"Error in check_cube_collision: {e}")
        return False

def get_cube_collision_normal(cube_position, cube_size, position):
    """
    Get the normal vector pointing away from the cube at the collision point.
    
    Args:
        cube_position: Center position of the cube (numpy array or list)
        cube_size: Half-size of the cube (scalar)
        position: Position of the colliding object (numpy array or list)
        
    Returns:
        Normalized vector pointing away from the cube
    """
    try:
        # Convert to numpy arrays if they're not already
        cube_pos = np.array(cube_position, dtype=float)
        pos = np.array(position, dtype=float)
        
        # Find closest point on cube to position
        closest_point = np.array([
            max(cube_pos[0] - cube_size, min(pos[0], cube_pos[0] + cube_size)),
            max(cube_pos[1] - cube_size, min(pos[1], cube_pos[1] + cube_size)),
            max(cube_pos[2] - cube_size, min(pos[2], cube_pos[2] + cube_size))
        ])
        
        # Calculate vector from closest point to position
        normal = pos - closest_point
        
        # Normalize the vector
        norm = np.linalg.norm(normal)
        if norm < 0.0001:
            # If we're exactly at the closest point, use a default normal
            # Find which face we're closest to
            distances = [
                abs(closest_point[0] - (cube_pos[0] - cube_size)),  # -X face
                abs(closest_point[0] - (cube_pos[0] + cube_size)),  # +X face
                abs(closest_point[1] - (cube_pos[1] - cube_size)),  # -Y face
                abs(closest_point[1] - (cube_pos[1] + cube_size)),  # +Y face
                abs(closest_point[2] - (cube_pos[2] - cube_size)),  # -Z face
                abs(closest_point[2] - (cube_pos[2] + cube_size))   # +Z face
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
        
        # Check if we're close to an edge or corner
        # If so, we want to favor the normal that helps with climbing
        normal_vec = normal / norm
        
        # If we're trying to climb up (moving upward and hitting a vertical face)
        if abs(normal_vec[1]) < 0.3 and pos[1] < closest_point[1] + STEP_HEIGHT:
            # Check if we're close to the top edge
            top_edge_dist = abs(pos[1] - (cube_pos[1] + cube_size))
            if top_edge_dist < STEP_HEIGHT:
                # Blend the normal with an upward vector to help climbing
                # This creates a smoother transition when climbing edges
                up_vector = np.array([0.0, 1.0, 0.0])
                blend_factor = 1.0 - (top_edge_dist / STEP_HEIGHT)
                blended_normal = normal_vec * (1.0 - blend_factor) + up_vector * blend_factor
                return normalize(blended_normal)
        
        return normal_vec
    except Exception as e:
        print(f"Error in get_cube_collision_normal: {e}")
        return np.array([0.0, 1.0, 0.0])  # Default up vector

def check_standing_on_cube(cube_position, cube_size, position, radius):
    """
    Check if an object is standing on top of a cube.
    
    Args:
        cube_position: Center position of the cube (numpy array or list)
        cube_size: Half-size of the cube (scalar)
        position: Position of the object (numpy array or list)
        radius: Radius of the object (player, block, etc.)
        
    Returns:
        Boolean indicating whether the object is standing on the cube
    """
    try:
        # Convert to numpy arrays if they're not already
        cube_pos = np.array(cube_position, dtype=float)
        pos = np.array(position, dtype=float)
        
        # Only consider positions that are above the cube
        if pos[1] < cube_pos[1] + cube_size:
            return False
        
        # Check if position is within horizontal bounds of cube (with some margin)
        x_min = cube_pos[0] - cube_size - radius
        x_max = cube_pos[0] + cube_size + radius
        z_min = cube_pos[2] - cube_size - radius
        z_max = cube_pos[2] + cube_size + radius
        
        if pos[0] < x_min or pos[0] > x_max or pos[2] < z_min or pos[2] > z_max:
            return False
        
        # Check if position is close enough to the top face
        top_face_y = cube_pos[1] + cube_size
        distance_to_top = pos[1] - radius - top_face_y
        
        # If we're very close to or slightly penetrating the top face
        return abs(distance_to_top) < 0.1
    except Exception as e:
        print(f"Error in check_standing_on_cube: {e}")
        return False

def resolve_cube_collision(cube1_pos, cube1_size, cube1_vel, cube2_pos, cube2_size, cube2_vel, restitution=0.5):
    """
    Resolve collision between two cubes by updating their velocities.
    
    Args:
        cube1_pos: Position of first cube (numpy array)
        cube1_size: Half-size of first cube (scalar)
        cube1_vel: Velocity of first cube (numpy array)
        cube2_pos: Position of second cube (numpy array)
        cube2_size: Half-size of second cube (scalar)
        cube2_vel: Velocity of second cube (numpy array)
        restitution: Coefficient of restitution (bounciness)
        
    Returns:
        Tuple of (new_vel1, new_vel2) - the updated velocities after collision
    """
    try:
        # Convert to numpy arrays if they're not already
        pos1 = np.array(cube1_pos, dtype=float)
        pos2 = np.array(cube2_pos, dtype=float)
        vel1 = np.array(cube1_vel, dtype=float)
        vel2 = np.array(cube2_vel, dtype=float)
        
        # Calculate collision normal (from cube2 to cube1)
        collision_normal = pos1 - pos2
        distance = np.linalg.norm(collision_normal)
        
        # Avoid division by zero
        if distance < 0.0001:
            collision_normal = np.array([0.0, 1.0, 0.0])  # Default up vector
        else:
            collision_normal = collision_normal / distance
        
        # Calculate relative velocity
        relative_velocity = vel1 - vel2
        
        # Calculate velocity along the normal
        velocity_along_normal = np.dot(relative_velocity, collision_normal)
        
        # If objects are moving away from each other, don't resolve collision
        if velocity_along_normal > 0:
            return vel1, vel2
        
        # Calculate impulse scalar
        j = -(1 + restitution) * velocity_along_normal
        j /= 2  # Assuming equal mass for simplicity
        
        # Apply impulse
        impulse = j * collision_normal
        new_vel1 = vel1 + impulse
        new_vel2 = vel2 - impulse
        
        return new_vel1, new_vel2
    except Exception as e:
        print(f"Error in resolve_cube_collision: {e}")
        return cube1_vel, cube2_vel

def resolve_cube_penetration(cube1_pos, cube1_size, cube2_pos, cube2_size):
    """
    Resolve penetration between two cubes by calculating the minimum translation vector.
    
    Args:
        cube1_pos: Position of first cube (numpy array)
        cube1_size: Half-size of first cube (scalar)
        cube2_pos: Position of second cube (numpy array)
        cube2_size: Half-size of second cube (scalar)
        
    Returns:
        Tuple of (new_pos1, new_pos2) - the updated positions after resolving penetration
    """
    try:
        # Convert to numpy arrays if they're not already
        pos1 = np.array(cube1_pos, dtype=float)
        pos2 = np.array(cube2_pos, dtype=float)
        
        # Calculate vector from cube2 to cube1
        direction = pos1 - pos2
        
        # Calculate overlap along each axis
        overlap_x = (cube1_size + cube2_size) - abs(direction[0])
        overlap_y = (cube1_size + cube2_size) - abs(direction[1])
        overlap_z = (cube1_size + cube2_size) - abs(direction[2])
        
        # If there's no overlap, return original positions
        if overlap_x <= 0 or overlap_y <= 0 or overlap_z <= 0:
            return pos1, pos2
        
        # Add a small buffer to prevent objects from being exactly touching
        buffer = COLLISION_TOLERANCE * 2
        
        # Prefer resolving along Y axis (up/down) for stacking stability
        # This helps with stacking cubes and climbing
        if overlap_y <= overlap_x * 1.2 and overlap_y <= overlap_z * 1.2:
            # Resolve along Y axis with preference
            if direction[1] < 0:
                mtv = np.array([0, -overlap_y - buffer, 0])
            else:
                mtv = np.array([0, overlap_y + buffer, 0])
        # Otherwise find the minimum overlap axis
        elif overlap_x <= overlap_y and overlap_x <= overlap_z:
            # Resolve along X axis
            if direction[0] < 0:
                mtv = np.array([-overlap_x - buffer, 0, 0])
            else:
                mtv = np.array([overlap_x + buffer, 0, 0])
        elif overlap_z <= overlap_x and overlap_z <= overlap_y:
            # Resolve along Z axis
            if direction[2] < 0:
                mtv = np.array([0, 0, -overlap_z - buffer])
            else:
                mtv = np.array([0, 0, overlap_z + buffer])
        else:
            # Fallback to Y axis
            if direction[1] < 0:
                mtv = np.array([0, -overlap_y - buffer, 0])
            else:
                mtv = np.array([0, overlap_y + buffer, 0])
        
        # Special case for stacking - if one cube is mostly above the other,
        # resolve purely in the Y direction for stability
        if pos1[1] > pos2[1] and abs(pos1[0] - pos2[0]) < cube1_size * 0.8 and abs(pos1[2] - pos2[2]) < cube1_size * 0.8:
            # Cube1 is above cube2 and mostly aligned - resolve vertically
            mtv = np.array([0, overlap_y + buffer, 0])
        elif pos2[1] > pos1[1] and abs(pos1[0] - pos2[0]) < cube1_size * 0.8 and abs(pos1[2] - pos2[2]) < cube1_size * 0.8:
            # Cube2 is above cube1 and mostly aligned - resolve vertically
            mtv = np.array([0, -overlap_y - buffer, 0])
        
        # Apply penetration correction factor to make movement smoother
        mtv *= PENETRATION_CORRECTION
        
        # Move each cube in opposite directions, with the upper cube moving more
        # This creates more stable stacking
        if pos1[1] > pos2[1]:
            # Cube1 is higher, move it more
            new_pos1 = pos1 + mtv * 0.7
            new_pos2 = pos2 - mtv * 0.3
        elif pos2[1] > pos1[1]:
            # Cube2 is higher, move it more
            new_pos1 = pos1 + mtv * 0.3
            new_pos2 = pos2 - mtv * 0.7
        else:
            # Equal height, move equally
            new_pos1 = pos1 + mtv * 0.5
            new_pos2 = pos2 - mtv * 0.5
        
        return new_pos1, new_pos2
    except Exception as e:
        print(f"Error in resolve_cube_penetration: {e}")
        return cube1_pos, cube2_pos

def continuous_cube_collision_detection(cube_pos, cube_size, cube_vel, dt, walls, other_cubes=None):
    """
    Perform continuous collision detection for a cube against walls and other cubes.
    
    Args:
        cube_pos: Current position of the cube (numpy array)
        cube_size: Half-size of the cube (scalar)
        cube_vel: Current velocity of the cube (numpy array)
        dt: Time step (scalar)
        walls: List of walls to check for collisions
        other_cubes: Optional list of other cubes to check for collisions
        
    Returns:
        Tuple of (new_pos, new_vel, on_ground) - updated position, velocity, and ground status
    """
    try:
        # Convert to numpy arrays if they're not already
        pos = np.array(cube_pos, dtype=float)
        vel = np.array(cube_vel, dtype=float)
        
        # Calculate the target position
        target_pos = pos + vel * dt
        
        # Initialize variables
        new_pos = pos.copy()
        new_vel = vel.copy()
        on_ground = False
        
        # Number of substeps for continuous collision detection
        # Increased for smoother movement
        num_steps = 4
        step_dt = dt / num_steps
        
        # Track collision normals for better response
        collision_normals = []
        
        for step in range(num_steps):
            # Calculate the step movement
            step_vel = new_vel.copy()
            step_pos = new_pos + step_vel * step_dt
            
            # Check for wall collisions
            wall_collision = False
            
            for wall in walls:
                if check_cube_collision(step_pos, cube_size, wall.position, wall.radius):
                    wall_collision = True
                    
                    # Get collision normal
                    normal = get_cube_collision_normal(wall.position, wall.radius, step_pos)
                    collision_normals.append(normal)
                    
                    # Calculate reflection with proper physics
                    dot_product_val = np.dot(step_vel, normal)
                    
                    # Only reflect if moving toward the surface
                    if dot_product_val < 0:
                        # Check if we can step up this obstacle
                        can_step_up = False
                        if normal[1] < 0.3 and abs(step_pos[1] - (wall.position[1] + wall.radius)) < STEP_HEIGHT:
                            # We're hitting a vertical surface but close to the top
                            # Try to step up instead of bouncing off
                            step_up_pos = step_pos.copy()
                            step_up_pos[1] = wall.position[1] + wall.radius + cube_size + 0.05  # Slightly above
                            
                            # Check if the position above is clear
                            above_collision = False
                            for w in walls:
                                if check_cube_collision(step_up_pos, cube_size, w.position, w.radius):
                                    above_collision = True
                                    break
                            
                            if not above_collision:
                                # We can step up
                                can_step_up = True
                                step_pos = step_up_pos
                                new_vel[1] = 0  # Zero vertical velocity
                                on_ground = True
                        
                        if not can_step_up:
                            # Reflect velocity off wall (with damping)
                            # Use a smoother reflection with less bounciness
                            reflection = step_vel - 2 * dot_product_val * normal
                            
                            # Apply damping with stabilization
                            damping = CUBE_RESTITUTION * STABILIZATION_FACTOR
                            new_vel = reflection * damping
                            
                            # For horizontal collisions, allow sliding along the wall
                            if abs(normal[1]) < 0.5:  # Not a floor/ceiling
                                # Project velocity onto the wall plane for smoother sliding
                                slide_vel = new_vel - np.dot(new_vel, normal) * normal
                                slide_vel *= WALL_SLIDE_FACTOR  # Reduce sliding speed
                                new_vel = slide_vel
                            
                            # Check if we're on the ground
                            if normal[1] > 0.7:  # Normal pointing mostly up
                                on_ground = True
                                new_vel[1] = 0  # Stop vertical movement
                                
                                # Snap to ground to prevent bouncing
                                if abs(step_pos[1] - (wall.position[1] + wall.radius + cube_size)) < GROUND_SNAP_DISTANCE:
                                    step_pos[1] = wall.position[1] + wall.radius + cube_size
                    
                    # Adjust position to avoid penetration with a smoother correction
                    penetration_vector = normal * PENETRATION_CORRECTION
                    step_pos += penetration_vector
            
            # Check for collisions with other cubes
            if other_cubes:
                for other_cube in other_cubes:
                    # Skip self-collision
                    if np.array_equal(other_cube.position, pos):
                        continue
                    
                    # Check for collision
                    if check_cube_collision(step_pos, cube_size, other_cube.position, other_cube.size):
                        # Get collision normal
                        normal = get_cube_collision_normal(other_cube.position, other_cube.size, step_pos)
                        collision_normals.append(normal)
                        
                        # Resolve velocity collision with improved stability
                        new_vel, other_vel = resolve_cube_collision(
                            step_pos, cube_size, new_vel,
                            other_cube.position, other_cube.size, other_cube.velocity,
                            CUBE_RESTITUTION * STABILIZATION_FACTOR  # More stable collisions
                        )
                        
                        # Update other cube's velocity
                        other_cube.velocity = other_vel
                        
                        # Resolve penetration
                        step_pos, other_pos = resolve_cube_penetration(
                            step_pos, cube_size,
                            other_cube.position, other_cube.size
                        )
                        
                        # Update other cube's position
                        other_cube.position = other_pos
                        
                        # Check if we're on top of the other cube
                        if step_pos[1] > other_cube.position[1] and abs(step_pos[1] - (other_cube.position[1] + other_cube.size + cube_size)) < GROUND_SNAP_DISTANCE:
                            on_ground = True
                            new_vel[1] = 0  # Stop vertical movement
                            
                            # Snap to the top of the cube
                            step_pos[1] = other_cube.position[1] + other_cube.size + cube_size
            
            # Update position for this step
            new_pos = step_pos
            
            # If we're on the ground, apply extra stabilization
            if on_ground:
                # Dampen horizontal velocity for stability
                new_vel[0] *= STABILIZATION_FACTOR
                new_vel[2] *= STABILIZATION_FACTOR
                
                # Ensure we stay on the ground
                new_vel[1] = min(new_vel[1], 0)
        
        # If we had multiple collisions, average the response for smoother movement
        if len(collision_normals) > 1:
            avg_normal = np.zeros(3)
            for n in collision_normals:
                avg_normal += n
            avg_normal = normalize(avg_normal)
            
            # Adjust velocity to be more aligned with the average normal
            dot_with_avg = np.dot(new_vel, avg_normal)
            if dot_with_avg < 0:
                # Remove velocity component going into the average normal
                new_vel -= dot_with_avg * avg_normal
        
        return new_pos, new_vel, on_ground
    except Exception as e:
        print(f"Error in continuous_cube_collision_detection: {e}")
        return cube_pos, cube_vel, False

def update_cube_physics(cube, dt, level, other_cubes=None):
    """
    Update the physics for a cube, handling gravity, collisions, and friction.
    
    Args:
        cube: The cube object to update
        dt: Time step (scalar)
        level: The level containing walls and other obstacles
        other_cubes: Optional list of other cubes to check for collisions
        
    Returns:
        None (updates the cube object directly)
    """
    try:
        # Skip physics update if being carried
        if hasattr(cube, 'being_carried') and cube.being_carried:
            return
        
        # Apply gravity if not on ground
        if not cube.on_ground:
            # Apply a smoother gravity with time-based damping
            cube.velocity[1] = max(cube.velocity[1] - GRAVITY * dt * STABILIZATION_FACTOR, 
                                  CUBE_TERMINAL_VELOCITY)
            
            # Apply air resistance to horizontal movement for more stability
            air_resistance = 0.1 * dt
            cube.velocity[0] *= (1.0 - air_resistance)
            cube.velocity[2] *= (1.0 - air_resistance)
        else:
            # Apply friction to horizontal movement
            # Use a higher friction when on ground for stability
            ground_friction = CUBE_FRICTION * dt * STABILIZATION_FACTOR
            cube.velocity[0] *= (1.0 - ground_friction)
            cube.velocity[2] *= (1.0 - ground_friction)
            
            # Stop horizontal movement if very slow (increased threshold)
            if abs(cube.velocity[0]) < 0.05:
                cube.velocity[0] = 0
            if abs(cube.velocity[2]) < 0.05:
                cube.velocity[2] = 0
            
            # Keep vertical velocity at zero when on ground
            cube.velocity[1] = 0
        
        # Limit maximum velocity to prevent tunneling and instability
        max_speed = 20.0
        current_speed = np.linalg.norm(cube.velocity)
        if current_speed > max_speed:
            # Scale down velocity to maximum
            cube.velocity = cube.velocity * (max_speed / current_speed)
        
        # Check if we need to snap to ground
        if cube.on_ground:
            # Find any surface directly below us
            ground_check_pos = cube.position.copy()
            ground_check_pos[1] -= GROUND_SNAP_DISTANCE
            
            for wall in level.walls:
                if check_cube_collision(ground_check_pos, cube.size, wall.position, wall.radius):
                    # Snap to the ground
                    cube.position[1] = wall.position[1] + wall.radius + cube.size
                    break
            
            # Also check other cubes
            if other_cubes:
                for other_cube in other_cubes:
                    if other_cube == cube:
                        continue
                        
                    if check_cube_collision(ground_check_pos, cube.size, other_cube.position, other_cube.size):
                        # Snap to the top of the other cube
                        cube.position[1] = other_cube.position[1] + other_cube.size + cube.size
                        break
        
        # Perform continuous collision detection with improved stability
        new_position, new_velocity, on_ground = continuous_cube_collision_detection(
            cube.position, cube.size, cube.velocity, dt, level.walls, other_cubes
        )
        
        # Apply velocity smoothing for stability
        # Blend old and new velocity for smoother transitions
        blend_factor = 0.8  # Higher values = smoother but less responsive
        if hasattr(cube, 'prev_velocity') and cube.prev_velocity is not None:
            # Only blend if we're not changing direction drastically
            dot_product = np.dot(normalize(new_velocity), normalize(cube.prev_velocity))
            if dot_product > 0:  # Moving in roughly the same direction
                # Blend velocities for smoother movement
                smoothed_velocity = cube.prev_velocity * (1 - blend_factor) + new_velocity * blend_factor
                new_velocity = smoothed_velocity
        
        # Store previous velocity for next frame
        cube.prev_velocity = new_velocity.copy()
        
        # Update cube properties
        cube.position = new_position
        cube.velocity = new_velocity
        cube.on_ground = on_ground
        
        # Apply additional stabilization for stacked cubes
        if other_cubes and on_ground:
            stabilize_stacked_cubes([cube] + other_cubes)
    except Exception as e:
        print(f"Error in update_cube_physics: {e}")

def check_cube_portal_collision(cube, portal, radius_multiplier=1.2):
    """
    Check if a cube is colliding with a portal.
    
    Args:
        cube: The cube object
        portal: The portal object
        radius_multiplier: Multiplier for the effective collision radius
        
    Returns:
        Boolean indicating whether the cube is colliding with the portal
    """
    try:
        # Use the cube's size as the radius for portal collision
        effective_radius = cube.size * radius_multiplier
        
        # Check if cube is close to portal plane
        dist_to_plane = abs(np.dot(cube.position - portal.position, portal.normal))
        
        # Increase collision detection distance to make it easier to go through portals
        if dist_to_plane > effective_radius * 2.0:
            return False
        
        # Project cube position onto portal plane
        projected_pos = cube.position - np.dot(cube.position - portal.position, portal.normal) * portal.normal
        
        # Calculate distance from portal center to projected position
        offset = projected_pos - portal.position
        
        # Transform to portal's local coordinate system
        local_x = np.dot(offset, portal.right)
        local_y = np.dot(offset, portal.up)
        
        # Make the collision area slightly larger than the visual portal
        width_factor = 1.2
        height_factor = 1.2
        
        # Add the cube size to make it easier to detect collisions
        effective_width = portal.width * 0.5 * width_factor + effective_radius * 0.8
        effective_height = portal.height * 0.5 * height_factor + effective_radius * 0.8
        
        # Check if point is inside ellipse: (x/a)² + (y/b)² <= 1
        normalized_dist = (local_x / effective_width)**2 + (local_y / effective_height)**2
        
        return normalized_dist <= 1.1
    except Exception as e:
        print(f"Error in check_cube_portal_collision: {e}")
        return False

def transport_cube_through_portal(cube, entry_portal, exit_portal):
    """
    Transport a cube through a portal, updating its position and velocity.
    
    Args:
        cube: The cube object to transport
        entry_portal: The portal the cube is entering
        exit_portal: The portal the cube will exit from
        
    Returns:
        None (updates the cube object directly)
    """
    try:
        # Calculate offset from entry portal
        offset = cube.position - entry_portal.position
        
        # Project offset onto portal plane
        projected_offset = offset - np.dot(offset, entry_portal.normal) * entry_portal.normal
        
        # Normalize the offset relative to the portal's oval dimensions
        local_x = np.dot(projected_offset, entry_portal.right)
        local_y = np.dot(projected_offset, entry_portal.up)
        
        # Scale coordinates to account for oval shape
        normalized_x = local_x / (entry_portal.width * 0.5)
        normalized_y = local_y / (entry_portal.height * 0.5)
        
        # Apply the normalized coordinates to the exit portal's dimensions
        exit_x = normalized_x * (exit_portal.width * 0.5)
        exit_y = normalized_y * (exit_portal.height * 0.5)
        
        # Calculate the exit offset
        exit_offset = exit_x * exit_portal.right + exit_y * exit_portal.up
        
        # Set new position slightly away from exit portal to avoid re-entry
        cube.position = exit_portal.position + exit_offset + exit_portal.normal * (cube.size + 0.1)
        
        # Transform velocity through portal
        entry_normal = entry_portal.normal
        exit_normal = exit_portal.normal
        
        # Calculate the rotation matrix between the two portals
        dot_product_portals = np.dot(entry_normal, exit_normal)
        cross_product_portals = np.cross(entry_normal, exit_normal)
        cross_magnitude = np.linalg.norm(cross_product_portals)
        
        if cross_magnitude > 0.001:  # If portals aren't parallel
            # Normalize the cross product
            cross_product_portals = cross_product_portals / cross_magnitude
            
            # Calculate rotation angle
            angle = math.acos(max(-1.0, min(1.0, dot_product_portals)))
            
            # Apply rotation to velocity using Rodrigues formula
            cube.velocity = calculate_rodrigues_rotation(cube.velocity, cross_product_portals, angle)
        else:
            # If portals are parallel, just reflect the velocity
            cube.velocity = cube.velocity - 2 * np.dot(cube.velocity, entry_normal) * entry_normal
        
        # Add a small boost in the direction of the exit portal normal to prevent sticking
        cube.velocity += exit_portal.normal * 2.0
        
        # Reset on_ground status as we're now in the air
        cube.on_ground = False
        
        print(f"Cube transported through portal: {entry_portal.color} -> {exit_portal.color}")
    except Exception as e:
        print(f"Error in transport_cube_through_portal: {e}")

def stabilize_stacked_cubes(cubes):
    """
    Stabilize stacked cubes to prevent them from sliding off each other too easily.
    
    Args:
        cubes: List of cube objects to check for stacking
        
    Returns:
        None (updates the cube objects directly)
    """
    try:
        # Check each pair of cubes
        for i, cube1 in enumerate(cubes):
            for j, cube2 in enumerate(cubes):
                # Skip self-comparison
                if i == j:
                    continue
                
                # Check if cube1 is directly above cube2
                if check_standing_on_cube(cube2.position, cube2.size, cube1.position, cube1.size):
                    # Cube1 is on top of cube2
                    
                    # If cube1 is moving slowly horizontally, dampen its movement further
                    # This prevents cubes from sliding off each other too easily
                    if np.linalg.norm(cube1.velocity[:3:2]) < 1.0:  # Only x and z components
                        # Apply extra friction to make stacking more stable
                        cube1.velocity[0] *= 0.7
                        cube1.velocity[2] *= 0.7
                        
                        # If very slow, stop completely to prevent micro-movements
                        if abs(cube1.velocity[0]) < 0.05:
                            cube1.velocity[0] = 0
                        if abs(cube1.velocity[2]) < 0.05:
                            cube1.velocity[2] = 0
                        
                        # Ensure cube1 is perfectly aligned with cube2 if it's not moving
                        if cube1.velocity[0] == 0 and cube1.velocity[2] == 0:
                            # Center cube1 on top of cube2 if it's very close to center
                            dist_from_center = np.linalg.norm(
                                np.array([cube1.position[0], cube1.position[2]]) - 
                                np.array([cube2.position[0], cube2.position[2]])
                            )
                            
                            if dist_from_center < cube2.size * 0.8:
                                # Align centers
                                cube1.position[0] = cube2.position[0]
                                cube1.position[2] = cube2.position[2]
                                
                                # Ensure correct height
                                cube1.position[1] = cube2.position[1] + cube2.size + cube1.size
    except Exception as e:
        print(f"Error in stabilize_stacked_cubes: {e}")

def get_cube_corners(cube_position, cube_size):
    """
    Get the 8 corners of a cube for collision detection.
    
    Args:
        cube_position: Center position of the cube (numpy array or list)
        cube_size: Half-size of the cube (scalar)
        
    Returns:
        List of 8 corner positions as numpy arrays
    """
    try:
        pos = np.array(cube_position, dtype=float)
        s = cube_size
        corners = []
        for x in [-s, s]:
            for y in [-s, s]:
                for z in [-s, s]:
                    corners.append(pos + np.array([x, y, z]))
        return corners
    except Exception as e:
        print(f"Error in get_cube_corners: {e}")
        return [np.array(cube_position) for _ in range(8)]  # Return 8 copies of center as fallback