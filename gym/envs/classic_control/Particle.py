import numpy as np


class Particle():
    def __init__(self, dt):
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.accumForce = np.zeros(2)
        self.mass = 1
        self.maxVelocity = 0.3 * np.ones(2)
        self.dt = dt
        # self.world

    def applyDragForce(self, ):
        if (np.linalg.norm(self.velocity) > 0.0005):
            self.accumForce += -self.velocity / np.linalg.norm(self.velocity) * (self.velocity ** 2) * 50
        # pass

    def advance(self):
        self.applyDragForce()
        self.velocity += self.dt * self.accumForce * 1.0 / self.mass
        self.velocity = np.clip(self.velocity, -self.maxVelocity, self.maxVelocity)
        self.position += self.dt * self.velocity
        self.accumForce = 0
