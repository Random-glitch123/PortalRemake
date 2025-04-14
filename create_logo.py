import pygame
import os
import sys

def create_portal_logo():
    # Create assets directory if it doesn't exist
    assets_dir = os.path.join("assets", "textures")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Set up logo dimensions
    width, height = 600, 200
    
    # Initialize pygame
    pygame.init()
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    
    # Colors
    blue_portal = (64, 138, 255)
    orange_portal = (255, 138, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)
    
    # Fill with transparent background
    surface.fill((0, 0, 0, 0))
    
    # Draw text shadow
    font = pygame.font.Font(None, 120)
    text_shadow = font.render("PORTAL", True, black)
    shadow_rect = text_shadow.get_rect(center=(width//2 + 4, height//2 + 4))
    surface.blit(text_shadow, shadow_rect)
    
    # Draw main text
    text = font.render("PORTAL", True, white)
    text_rect = text.get_rect(center=(width//2, height//2))
    surface.blit(text, text_rect)
    
    # Draw blue portal circle on the left
    pygame.draw.circle(surface, blue_portal, (100, height//2), 40)
    pygame.draw.circle(surface, black, (100, height//2), 40, 3)
    pygame.draw.circle(surface, (200, 200, 255), (100, height//2), 30)
    
    # Draw orange portal circle on the right
    pygame.draw.circle(surface, orange_portal, (width - 100, height//2), 40)
    pygame.draw.circle(surface, black, (width - 100, height//2), 40, 3)
    pygame.draw.circle(surface, (255, 200, 150), (width - 100, height//2), 30)
    
    # Add "GAME" text below
    font_small = pygame.font.Font(None, 60)
    game_text = font_small.render("GAME", True, white)
    game_rect = game_text.get_rect(center=(width//2, height//2 + 50))
    surface.blit(game_text, game_rect)
    
    # Save the logo
    logo_path = os.path.join(assets_dir, "portal_logo.png")
    pygame.image.save(surface, logo_path)
    print(f"Logo created and saved to {logo_path}")

if __name__ == "__main__":
    create_portal_logo()
    print("Run this script to create a logo for the Portal Game")
    print("The logo will be saved in the assets/textures directory")