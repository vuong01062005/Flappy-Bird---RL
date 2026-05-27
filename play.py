import pygame
import sys
import torch
import numpy as np

from flappy_env import FlappyEnv
from dqn_model import DQN

MODEL_PATH = "./models/flappy_dqn_2800.pth"
model = DQN(8, 256, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

pygame.init()
screen = pygame.display.set_mode((432, 768))
clock = pygame.time.Clock()

bg = pygame.transform.scale2x(
    pygame.image.load("./assets/background-night.png").convert()
)

floor = pygame.transform.scale2x(
    pygame.image.load("./assets/floor.png").convert()
)
floor_x = 0

# Bird animation
bird_down = pygame.transform.scale2x(pygame.image.load('./assets/yellowbird-downflap.png').convert_alpha())
bird_mid  = pygame.transform.scale2x(pygame.image.load('./assets/yellowbird-midflap.png').convert_alpha())
bird_up   = pygame.transform.scale2x(pygame.image.load('./assets/yellowbird-upflap.png').convert_alpha())
bird_list = [bird_down, bird_mid, bird_up]
bird_index = 0

BIRDFLAP = pygame.USEREVENT + 1
pygame.time.set_timer(BIRDFLAP, 150)

pipe_surface = pygame.transform.scale2x(pygame.image.load("./assets/pipe-green.png").convert())

# Game Over Message
game_over_surface = pygame.transform.scale2x(
    pygame.image.load('./assets/message.png').convert_alpha()
)
game_over_rect = game_over_surface.get_rect(center=(216, 384))

# Font
game_font = pygame.font.Font('./04B_19.TTF', 40)

# Sounds
flap_sound  = pygame.mixer.Sound('./sound/sfx_wing.wav')
hit_sound   = pygame.mixer.Sound('./sound/sfx_hit.wav')
score_sound = pygame.mixer.Sound('./sound/sfx_point.wav')

# Score
score = 0
high_score = 0
passed_pipe = False

# ENV
env = FlappyEnv()

# Draw pipe
def draw_pipes(x, top, bottom):
    flip_pipe = pygame.transform.flip(pipe_surface, False, True)
    top_rect = flip_pipe.get_rect(midbottom=(x, top))
    bottom_rect = pipe_surface.get_rect(midtop=(x, bottom))

    screen.blit(flip_pipe, top_rect)
    screen.blit(pipe_surface, bottom_rect)

def rotate_bird(bird1, movement):
    return pygame.transform.rotozoom(bird1, -movement * 3, 1)

# Draw bird
def draw_bird(y, movement):
    rotated_bird = rotate_bird(bird_list[bird_index], movement)
    bird_rect = rotated_bird.get_rect(center=(100, y))
    screen.blit(rotated_bird, bird_rect)

# Score display
def score_display(game_state):
    if game_state == 'main game':
        s = game_font.render(str(int(score)), True, (255,255,255))
        screen.blit(s, s.get_rect(center=(216, 100)))

    else:
        s1 = game_font.render(f'Score: {score}', True, (255,255,255))
        s2 = game_font.render(f'High Score: {high_score}', True, (255,255,255))
        screen.blit(s1, s1.get_rect(center=(216, 100)))
        screen.blit(s2, s2.get_rect(center=(216, 630)))

# Intro screen
def intro_screen():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return

        screen.blit(bg, (0, 0))
        screen.blit(game_over_surface, game_over_rect)

        pygame.display.update()
        clock.tick(30)

# Game start
intro_screen()
state = env.reset()

# Main loop
while True:
    # Update animation
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == BIRDFLAP:
            bird_index = (bird_index + 1) % 3

    # AI chọn hành động
    with torch.no_grad():
        action = model(torch.tensor(state).float()).argmax().item()

    if action == 1:
        flap_sound.play()

    next_state, reward, done = env.step(action)
    state = next_state

    # SCORE FIX
    bird_x = 100
    pipe_x = env.pipe_x

    if pipe_x < bird_x and not passed_pipe:
        score += 1
        score_sound.play()
        passed_pipe = True

    # if score > 10:
    #     env.pipe_speed = 4

    if pipe_x > bird_x:
        passed_pipe = False

    # DRAW
    screen.blit(bg, (0, 0))

    draw_pipes(env.pipe_x, env.pipe_top, env.pipe_bottom)

    draw_bird(env.bird_y, env.bird_vel)
    score_display("main game")

    # Floor
    floor_x -= 1
    if floor_x <= -432:
        floor_x = 0

    screen.blit(floor, (floor_x, 650))
    screen.blit(floor, (floor_x + 432, 650))

    pygame.display.update()
    clock.tick(60)

    # Game over
    if done:
        hit_sound.play()
        high_score = max(high_score, score)

        screen.blit(bg, (0, 0))
        score_display("game_over")
        screen.blit(game_over_surface, game_over_rect)
        pygame.display.update()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False

        score = 0
        passed_pipe = False
        state = env.reset()