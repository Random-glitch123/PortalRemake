# Portal Game Guide

## Controls
- **WASD**: Move
- **Mouse**: Look around
- **Space**: Jump
- **Left Click**: Place Blue Portal
- **Right Click**: Place Orange Portal
- **R**: Reset Level
- **ESC**: Exit Game

## Portal Placement
You can now place portals on:
- Walls
- Ceilings
- Floors

The portals will automatically orient themselves correctly based on the surface they're placed on.

## Adding Textures and Assets

### Texture Directory Structure
The game now supports textures and sounds. Place your assets in the following directories:

```
assets/
  textures/
    concrete-wall.jpeg
    floor_concrete.png
    ceiling_tile.png
    portal_blue_rim.png
    portal_orange_rim.png
  sounds/
    portal_open_blue.wav
    portal_open_orange.wav
    portal_enter.wav
```

### Creating Your Own Textures
You can create your own textures using any image editing software. For best results:
- Use power-of-two dimensions (128x128, 256x256, 512x512, etc.)
- Save in PNG format for transparency support
- Keep file sizes reasonable for better performance

### Finding Free Textures
You can find free textures at:
1. [OpenGameArt.org](https://opengameart.org/)
2. [Kenney.nl](https://kenney.nl/assets)
3. [Texture Haven](https://texturehaven.com/)

### Finding Free Sounds
You can find free sound effects at:
1. [Freesound.org](https://freesound.org/)
2. [OpenGameArt.org](https://opengameart.org/)
3. [Kenney.nl](https://kenney.nl/assets)

## Creating Custom Levels

To create a custom level, modify the `create_level1()` function in the code. Here's an example:

```python
def create_custom_level():
    walls = [
        # Floor
        Wall(0, -1, 0, 20, 0.5, 20),
        
        # Ceiling
        Wall(0, 5, 0, 20, 0.5, 20),
        
        # Surrounding walls
        Wall(-10, 2, 0, 0.5, 6, 20),  # Left wall
        Wall(10, 2, 0, 0.5, 6, 20),   # Right wall
        Wall(0, 2, -10, 20, 6, 0.5),  # Back wall
        Wall(0, 2, 10, 20, 6, 0.5),   # Front wall
        
        # Add your custom walls and platforms here
        Wall(-5, 0, -5, 3, 2, 0.5),   # Wall 1
        Wall(5, 0, 5, 0.5, 2, 3),     # Wall 2
        Wall(0, 1, 0, 4, 0.5, 4),     # Platform in center
        Wall(-7, 2, 7, 4, 0.5, 4),    # Elevated platform
    ]
    
    goal_pos = (7, 0, -7)  # Position of the goal
    player_start = (0, 0, 8)  # Starting position of the player
    
    return Level(walls, goal_pos, player_start)
```

Then update the main function to use your custom level:

```python
# Create level
current_level = create_custom_level()
```

## Tips for Creating Interesting Portal Puzzles

1. **Height Differences**: Create puzzles that require the player to gain height by placing portals strategically.

2. **Momentum Conservation**: Remember that momentum is conserved when going through portals. Design puzzles that require the player to build up speed.

3. **Limited Sightlines**: Create areas where the player can't see the goal directly but must use portals to navigate.

4. **Multiple Steps**: Design puzzles that require multiple portal placements to solve.

5. **Restricted Portal Surfaces**: Make some walls non-portal surfaces to increase the challenge.

Enjoy your enhanced Portal Game!