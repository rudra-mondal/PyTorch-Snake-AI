# play.py

import torch
from agent import Agent # Reuse your agent class
from game_ai import SnakeGameAI # Reuse your game class

def play():
    # Load the trained model
    model_path = './model/model.pth'
    
    # We don't need a GPU for just playing, CPU is fine
    device = torch.device("cpu")
    agent = Agent() # Create an agent instance
    agent.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.model.eval() # Set the model to evaluation mode

    game = SnakeGameAI()
    
    while True:
        # Get the current state
        state_old = agent.get_state(game)

        # Let the model decide the action (no randomness)
        with torch.no_grad():
            state_tensor = torch.tensor(state_old, dtype=torch.float)
            prediction = agent.model(state_tensor)
            move_idx = torch.argmax(prediction).item()
        
        final_move = [0, 0, 0]
        final_move[move_idx] = 1

        # Perform the move
        reward, done, score = game.play_step(final_move)

        if done:
            game.reset()
            print(f"Final Score: {score}")

if __name__ == '__main__':
    play()