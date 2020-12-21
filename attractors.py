import pygame
import numpy as np
import math
import os
from pygame import QUIT
from colorsys import hsv_to_rgb

#Takes a 3d point, converts into a 2d projection.
#Angle is angle of camera (which rotates) at time of call.
#The nuts-and-bolts of this method are explained here:
#https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix
def screenXY(point, width, height, angle):
    #Rotate around Y Axis.
    rotationMat =    [[math.cos(angle), 0, -math.sin(angle)],
                      [0, 1, 0],
                      [math.sin(angle), 0, math.cos(angle)]]
    #Perspective projection matrix.
    projMat = [ [1, 0, 0],
                [0, 1, 0] ]
    
    twoDimRot =  np.dot(rotationMat, point)
    #Z value projected onto 2d space.
    zVal = (5 - twoDimRot[2]) ** (-1)
    
    projectedPoint = np.dot(projMat, twoDimRot)
    
    #Relativize to window dimensions.
    x = int(projectedPoint[0] * 20) + width // 2 + 100
    y = int(projectedPoint[1] * 20) + height // 2
    
    return (x, y)

#So pygame window pops up in middle of screen.
os.environ['SDL_VIDEO_CENTERED'] = '1000'
#Start pygame.
pygame.init()

#Put on scale of 0-255.
def hsvToRgbScaled(h, s, v):
    return tuple(round(i * 255) 
                 for i in hsv_to_rgb(h, s, v))

#Base class for chaotic attractors.
class attractor:
    #Takes dict of differential equations and list of intial points.
    def __init__(self, differentials, initConds):
        self.dx = differentials['dx']
        self.dy = differentials['dy']
        self.dz = differentials['dz']

        self.points = []
        for pt in initConds:
            self.points.append([pt])
    
    #For each initial condition, evolve by one step.
    def update(self):
        for index, ptList in enumerate(self.points):
            x, y, z = ptList[-1]

            dx = self.dx(x, y, z)
            dy = self.dy(x, y, z)
            dz = self.dz(x, y, z)

            x += dx
            y += dy
            z += dz

            self.points[index].append((x, y, z))
    
    #Graphically render the evolvng
    def show(self):
        screen = pygame.display.set_mode((1920, 1080))
        clock = pygame.time.Clock()
        
        keepGoing = 1
        angle = 0
        hue = 0.75
        
        while keepGoing:
            screen.fill((0, 0, 0))
            #45 FPS.
            clock.tick(45)
            
            hue += math.cos(angle) / 200
            if hue > 0.85 or hue < 0.75:
                hue = 0.75
                
            #If user exits, break.
            for event in pygame.event.get():
                if event.type == QUIT:
                    keepGoing = 0
            
            for listInd, ptList in enumerate(self.points):
                for index, point in enumerate(ptList[1:]):
                    projPt = screenXY(point, 1920, 1080, angle)
                    prevPt = screenXY(ptList[index - 1], 1920, 
                                      1080, angle)
                    
                    try:
                        pygame.draw.line(screen, hsvToRgbScaled(hue, 1, 1), 
                                         projPt, prevPt, 4)
                    #If number goes out of bounds.
                    except:
                        try:
                            self.points.pop(listInd)
                        #If multiple points go out of bound on same iter, ignore.
                        except:
                            pass
            
            self.update()
            
            #Rotate camera.
            angle += 0.01
            #Update visuals.
            pygame.display.update()
            
        pygame.quit()

#Generate num random 3d points between one and 5
def randPoints(num):
    return [np.random.uniform(1, 5, size = 3) 
            for i in range(num)]

#Lorenz attractor.
class lorenz(attractor):
    def __init__(self, sigma, rho, beta, inits, const):
        diff = {
            'dx' : lambda x, y, z : const * (sigma * (y - x)),
            'dy' : lambda x, y, z : const * (x * (rho - z) - y),
            'dz' : lambda x, y, z : const * (x * y - beta * z)
        }
        
        super().__init__(diff, inits)

inits = randPoints(10)
lorenzInst = lorenz(10, 28, 8/3, inits, 0.009)
lorenzInst.show()

#Rossler attractor.
class rossler(attractor):
    def __init__(self, a, b, c, inits, const):
        diff = {
            'dx' : lambda x, y, z : (-y - z) * const,
            'dy' : lambda x, y, z : (x + a * y) * const,
            'dz' : lambda x, y, z : (b + z * (x - c)) * const
        }
        
        super().__init__(diff, inits)

inits = randPoints(10)
rosslerInst = rossler(0.2, 0.2, 5.7, inits, 0.075)
rosslerInst.show()

#Thomas' cyclically symmetric attractor.
class thomas(attractor):
    def __init__(self, b, inits, const):
        diff = {
            'dx' : lambda x, y, z : (math.sin(y) - b * x) * const,
            'dy' : lambda x, y, z : (math.sin(z) - b * y) * const,
            'dz' : lambda x, y, z : (math.sin(x) - b * z) * const
        }
        
        super().__init__(diff, inits)

inits = randPoints(10)
thomasInst = thomas(1, inits, 2)
thomasInst.show()

#Finance attractor.
class finance(attractor):
    def __init__(self, a, b, c, inits, const):
        diff = {
            'dx' : lambda x, y, z : ((1 / b - a) * x + z + x * y) * const,
            'dy' : lambda x, y, z : (-b*y - x**2) * const,
            'dz' : lambda x, y, z : (-x - c * z) * const
        }
        
        super().__init__(diff, inits)

inits = randPoints(10)
financeInst = finance(0.001, 0.2, 1.1, inits, 0.025)
financeInst.show()

#noseHoover attractor.
class noseHoover(attractor):
    def __init__(self, a, inits, const):
        diff = {
            'dx' : lambda x, y, z : (y) * const,
            'dy' : lambda x, y, z : (-x + y * z) * const,
            'dz' : lambda x, y, z : (a - y**2) * const
        }
        
        super().__init__(diff, inits)

inits = randPoints(50)
noseHooverInst = noseHoover(1.5, inits, 0.1)
noseHooverInst.show()

#Wang-Sun attractor.
class wangSun(attractor):
    def __init__(self, a, b, c, d, e, f, inits, const):
        diff = {
            'dx' : lambda x, y, z : (x * a + c * y * z) * const,
            'dy' : lambda x, y, z : (b * x + d * y - x * z) * const,
            'dz' : lambda x, y, z : (e * z + f * x * y) * const
        }
        
        super().__init__(diff, inits)

inits = randPoints(10)
wangSunInst = wangSun(0.2, -0.01, 1, -0.4, -1, -1, inits, 0.0185)
wangSunInst.show()

#Halvorsen attractor.
class halvorsen(attractor):
    def __init__(self, a, inits, const):
        diff = {
            'dx' : lambda x, y, z : (-a * x - 4 * y - 4 * z - y**2) * const,
            'dy' : lambda x, y, z : (-a * y - 4 * z - 4 * x - z**2) * const,
            'dz' : lambda x, y, z : (-a * z - 4 * x - 4 * y - x**2) * const
        }
        
        super().__init__(diff, inits)

inits = randPoints(50)
halvorsenInst = halvorsen(1.4, inits, 0.005)
halvorsenInst.show()