import numpy as np

class ParticlesSim():
    def __init__(self, numPart, dt):
        self.numParticles = numPart
        self.positions = np.zeros((numPart, 2), dtype=np.float32)
        self.velocities = np.zeros((numPart, 2), dtype=np.float32)
        self.accumForce = np.zeros((numPart, 2), dtype=np.float32)
        self.mass = np.ones((numPart, 1), dtype=np.float32)
        self.maxVelocities = 0.5  # * np.ones((numPart, 2), dtype=np.float32)
        self.dt = dt
        self.repelling_force_scale = 7
        self.force_distance = 0.3

    def applyDragForce(self, ):
        velNorm = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        velNorm[velNorm < 0.0005] = 1
        self.accumForce += -self.velocities / velNorm * (
                self.velocities ** 2) * 50

    #
    def applyPushForce(self, p0, p1, normDir):
        normDirs = np.ones((self.numParticles, 2)) * normDir

        # Check if particles are on one side of the virtual fan
        P1 = np.ones((self.numParticles, 2), dtype=np.float32) * p0
        P2 = np.ones((self.numParticles, 2), dtype=np.float32) * p1

        P1P = self.positions - P1
        P2P = self.positions - P2

        P1P2 = P2 - P1

        P1P2Norm = np.linalg.norm(P1P2, axis=1, keepdims=True)
        P1P2Norm[P1P2Norm < 0.005] = 1
        P1P2_norm = P1P2 / P1P2Norm
        # if(P1P2)
        combined = np.concatenate((np.expand_dims(P1P, axis=1), np.expand_dims(P1P2, axis=1)), axis=1)
        side = np.sign(np.linalg.det(combined)) >= 0

        # Check if particles are outside the range of the fan
        range_test_1 = np.sum(P1P * P1P2, axis=1) >= 0
        range_test_2 = np.sum(P2P * -P1P2, axis=1) >= 0
        # Check if the particles are too far away from the fan
        distance = np.linalg.norm(P1P - np.sum(P1P * P1P2_norm, axis=1, keepdims=True) * P1P2_norm, axis=1,
                                  keepdims=True)

        distance_check = distance[:, 0] <= self.force_distance
        selector = np.logical_and(np.logical_and(np.logical_and(side, range_test_1), range_test_2),
                                  distance_check).astype(int)
        selector = np.expand_dims(selector, axis=1)

        self.accumForce += selector * normDirs * self.repelling_force_scale

    def advance(self, p0, p1, normDir):
        self.applyDragForce()
        self.applyPushForce(p0, p1, normDir)
        self.velocities += self.dt * self.accumForce * 1.0 / self.mass
        self.velocities = np.clip(self.velocities, -self.maxVelocities, self.maxVelocities)
        self.positions += self.dt * self.velocities
        self.accumForce.fill(0)

    def randomInit(self, worldSize):
        pos = (np.random.rand(self.numParticles, 2, ) - 0.5) * worldSize
        vel = (np.random.rand(self.numParticles, 2) - 0.5) * 0.2

        self.positions = pos
        self.velocities = vel

    def fillDensityMap(self, worldSize, numCells):
        x_pos = self.positions[:, 0]
        y_pos = self.positions[:, 1]

        H, xedges, yedges = np.histogram2d(y_pos, x_pos, bins=[numCells, numCells],
                                           range=[[-worldSize / 2, worldSize / 2], [-worldSize / 2, worldSize / 2]])
        H = np.clip(H, 0, 5)
        return np.flip(H, axis=0)
