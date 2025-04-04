import pygame
import sys
import math
+
+# Initialize pygame
+pygame.init()
+
+# Constants
+WIDTH, HEIGHT = 800, 600
+PLAYER_SIZE = 30
+WALL_COLOR = (100, 100, 100)
+PLAYER_COLOR = (255, 200, 0)
+PORTAL1_COLOR = (0, 100, 255)  # Blue portal
+PORTAL2_COLOR = (255, 100, 0)  # Orange portal
+GOAL_COLOR = (0, 255, 0)
+BACKGROUND_COLOR = (30, 30, 30)
+PORTAL_SIZE = 40
+GRAVITY = 0.5
+JUMP_STRENGTH = 10
+MOVE_SPEED = 5
+
+# Create the screen
+screen = pygame.display.set_mode((WIDTH, HEIGHT))
+pygame.display.set_caption("Portal Puzzle Game")
+clock = pygame.time.Clock()
+
+class Player:
+    def __init__(self, x, y):
+        self.x = x
+        self.y = y
+        self.width = PLAYER_SIZE
+        self.height = PLAYER_SIZE
+        self.vel_x = 0
+        self.vel_y = 0
+        self.on_ground = False
+    
+    def update(self, walls):
+        # Apply gravity
+        self.vel_y += GRAVITY
+        
+        # Move horizontally
+        self.x += self.vel_x
+        
+        # Check for horizontal collisions
+        for wall in walls:
+            if self.collides_with(wall):
+                if self.vel_x > 0:  # Moving right
+                    self.x = wall.x - self.width
+                elif self.vel_x < 0:  # Moving left
+                    self.x = wall.x + wall.width
+                self.vel_x = 0
+        
+        # Move vertically
+        self.y += self.vel_y
+        self.on_ground = False
+        
+        # Check for vertical collisions
+        for wall in walls:
+            if self.collides_with(wall):
+                if self.vel_y > 0:  # Moving down
+                    self.y = wall.y - self.height
+                    self.on_ground = True
+                elif self.vel_y < 0:  # Moving up
+                    self.y = wall.y + wall.height
+                self.vel_y = 0
+    
+    def collides_with(self, rect):
+        return (self.x < rect.x + rect.width and
+                self.x + self.width > rect.x and
+                self.y < rect.y + rect.height and
+                self.y + self.height > rect.y)
+    
+    def draw(self):
+        pygame.draw.rect(screen, PLAYER_COLOR, (self.x, self.y, self.width, self.height))
+
+class Wall:
+    def __init__(self, x, y, width, height):
+        self.x = x
+        self.y = y
+        self.width = width
+        self.height = height
+    
+    def draw(self):
+        pygame.draw.rect(screen, WALL_COLOR, (self.x, self.y, self.width, self.height))
+
+class Portal:
+    def __init__(self, x, y, is_portal1=True):
+        self.x = x
+        self.y = y
+        self.is_portal1 = is_portal1
+        self.size = PORTAL_SIZE
+        self.active = True
+    
+    def draw(self):
+        if not self.active:
+            return
+        color = PORTAL1_COLOR if self.is_portal1 else PORTAL2_COLOR
+        pygame.draw.ellipse(screen, color, (self.x - self.size//2, self.y - self.size//2, 
+                                           self.size, self.size))
+
+class Goal:
+    def __init__(self, x, y, width, height):
+        self.x = x
+        self.y = y
+        self.width = width
+        self.height = height
+    
+    def collides_with(self, player):
+        return (player.x < self.x + self.width and
+                player.x + player.width > self.x and
+                player.y < self.y + self.height and
+                player.y + player.height > self.y)
+    
+    def draw(self):
+        pygame.draw.rect(screen, GOAL_COLOR, (self.x, self.y, self.width, self.height))
+
+class Level:
+    def __init__(self, walls, goal_pos, player_start):
+        self.walls = walls
+        self.goal = Goal(goal_pos[0], goal_pos[1], 40, 40)
+        self.player_start = player_start
+
+def create_level1():
+    walls = [
+        # Floor
+        Wall(0, HEIGHT - 50, WIDTH, 50),
+        # Left wall
+        Wall(0, 0, 20, HEIGHT),
+        # Right wall
+        Wall(WIDTH - 20, 0, 20, HEIGHT),
+        # Ceiling
+        Wall(0, 0, WIDTH, 20),
+        # Platforms
+        Wall(200, 400, 200, 20),
+        Wall(500, 300, 200, 20),
+        # Obstacle
+        Wall(350, 200, 100, 200)
+    ]
+    goal_pos = (700, HEIGHT - 90)
+    player_start = (50, HEIGHT - 100)
+    return Level(walls, goal_pos, player_start)
+
+def main():
+    # Create level
+    current_level = create_level1()
+    
+    # Create player
+    player = Player(current_level.player_start[0], current_level.player_start[1])
+    
+    # Portal variables
+    portal1 = None
+    portal2 = None
+    
+    # Game loop
+    running = True
+    while running:
+        # Handle events
+        for event in pygame.event.get():
+            if event.type == pygame.QUIT:
+                running = False
+            
+            # Handle key presses
+            if event.type == pygame.KEYDOWN:
+                if event.key == pygame.K_SPACE and player.on_ground:
+                    player.vel_y = -JUMP_STRENGTH
+                if event.key == pygame.K_r:  # Reset level
+                    player.x = current_level.player_start[0]
+                    player.y = current_level.player_start[1]
+                    player.vel_x = 0
+                    player.vel_y = 0
+                    portal1 = None
+                    portal2 = None
+            
+            # Handle mouse clicks for portal placement
+            if event.type == pygame.MOUSEBUTTONDOWN:
+                mouse_x, mouse_y = pygame.mouse.get_pos()
+                
+                # Check if we can place a portal here (not inside a wall)
+                can_place = True
+                for wall in current_level.walls:
+                    if (mouse_x > wall.x and mouse_x < wall.x + wall.width and
+                        mouse_y > wall.y and mouse_y < wall.y + wall.height):
+                        can_place = False
+                        break
+                
+                if can_place:
+                    if event.button == 1:  # Left click - blue portal
+                        portal1 = Portal(mouse_x, mouse_y, True)
+                    elif event.button == 3:  # Right click - orange portal
+                        portal2 = Portal(mouse_x, mouse_y, False)
+        
+        # Handle continuous key presses
+        keys = pygame.key.get_pressed()
+        player.vel_x = 0
+        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
+            player.vel_x = -MOVE_SPEED
+        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
+            player.vel_x = MOVE_SPEED
+        
+        # Update player
+        player.update(current_level.walls)
+        
+        # Check for portal teleportation
+        if portal1 and portal2 and portal1.active and portal2.active:
+            # Calculate distance from player center to portal centers
+            player_center_x = player.x + player.width / 2
+            player_center_y = player.y + player.height / 2
+            
+            dist_to_portal1 = math.sqrt((player_center_x - portal1.x)**2 + 
+                                        (player_center_y - portal1.y)**2)
+            dist_to_portal2 = math.sqrt((player_center_x - portal2.x)**2 + 
+                                        (player_center_y - portal2.y)**2)
+            
+            # If player is close to a portal, teleport to the other
+            if dist_to_portal1 < PORTAL_SIZE / 2:
+                player.x = portal2.x - player.width / 2
+                player.y = portal2.y - player.height / 2
+                # Preserve momentum but flip direction if needed
+                temp_vel_x = player.vel_x
+                temp_vel_y = player.vel_y
+                player.vel_x = temp_vel_x
+                player.vel_y = temp_vel_y
+            
+            elif dist_to_portal2 < PORTAL_SIZE / 2:
+                player.x = portal1.x - player.width / 2
+                player.y = portal1.y - player.height / 2
+                # Preserve momentum but flip direction if needed
+                temp_vel_x = player.vel_x
+                temp_vel_y = player.vel_y
+                player.vel_x = temp_vel_x
+                player.vel_y = temp_vel_y
+        
+        # Check if player reached the goal
+        if current_level.goal.collides_with(player):
+            print("Level completed!")
+            # Here you would typically load the next level
+            # For now, we'll just reset the player position
+            player.x = current_level.player_start[0]
+            player.y = current_level.player_start[1]
+        
+        # Draw everything
+        screen.fill(BACKGROUND_COLOR)
+        
+        # Draw walls
+        for wall in current_level.walls:
+            wall.draw()
+        
+        # Draw goal
+        current_level.goal.draw()
+        
+        # Draw portals
+        if portal1:
+            portal1.draw()
+        if portal2:
+            portal2.draw()
+        
+        # Draw player
+        player.draw()
+        
+        # Draw instructions
+        font = pygame.font.SysFont(None, 24)
+        instructions = [
+            "Left/Right or A/D: Move",
+            "Space: Jump",
+            "Left Click: Place Blue Portal",
+            "Right Click: Place Orange Portal",
+            "R: Reset Level"
+        ]
+        for i, text in enumerate(instructions):
+            text_surface = font.render(text, True, (255, 255, 255))
+            screen.blit(text_surface, (10, 10 + i * 25))
+        
+        # Update display
+        pygame.display.flip()
+        clock.tick(60)
+    
+    pygame.quit()
+    sys.exit()
+
+if __name__ == "__main__":
+    main()