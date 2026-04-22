import pygame
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
from game_ai import SnakeGameAI, Direction

game = SnakeGameAI()
game._update_ui()
print("UI updated successfully")
