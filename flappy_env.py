import numpy as np
import random

class FlappyEnv:
    def __init__(self):
        self.gravity = 0.25
        self.jump_speed = -8

        self.bird_y = 300
        self.bird_vel = 0

        self.pipe_x = 500
        self.pipe_gap = 160
        self.pipe_speed = 3
        self.pipe_top = 0
        self.pipe_bottom = 0

        self.done = False

    def reset(self):
        self.bird_y = 300
        self.bird_vel = 0

        self.pipe_x = 500
        pipe_center = random.randint(200, 450)
        self.pipe_top = pipe_center - self.pipe_gap // 2
        self.pipe_bottom = pipe_center + self.pipe_gap // 2

        self.done = False

        return self._get_state()

    def _get_state(self):
        return np.array([
            self.bird_y,
            self.bird_vel,
            self.pipe_x,
            self.pipe_top,
            self.pipe_bottom,

            self.pipe_x - 100,
            self.bird_y - self.pipe_top,
            self.bird_y - self.pipe_bottom
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True

        if action == 1:
            self.bird_vel = self.jump_speed
        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel

        self.pipe_x -= self.pipe_speed
        if self.pipe_x < -50:
            self.pipe_x = 450
            pipe_center = random.randint(200, 450)
            self.pipe_top = pipe_center - self.pipe_gap // 2
            self.pipe_bottom = pipe_center + self.pipe_gap // 2

        if self.bird_y <= 0 or self.bird_y >= 600:
            self.done = True

        if 50 < self.pipe_x < 150:
            if not (self.pipe_top < self.bird_y < self.pipe_bottom):
                self.done = True


        reward = 0.1
        if action == 1:
            reward += 0.01
        if self.done:
            reward = -50

        if self.pipe_x == 150:
            reward += 10

        return self._get_state(), reward, self.done