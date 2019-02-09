import pygame
import sys
import random

class FlappyBird:
    def __init__(self):
        self.screen         = pygame.display.set_mode((400, 600))
        self.birdSprite     = pygame.image.load("assets/sprites/yellowbird-upflap.png")
        self.candySprite    = pygame.image.load("assets/sprites/Candy.png")
        self.background     = pygame.image.load("assets/sprites/background.png").convert()
        self.topPipe        = pygame.image.load("assets/sprites/rsz_top.png").convert_alpha()
        self.bottomPipe     = pygame.image.load("assets/sprites/rsz_bottom.png").convert_alpha()
        self.birdX,self.birdY   = (100,350)
        self.candyX,self.candyY = (200,150)
        self.bird           = pygame.Rect(self.birdX, self.birdY, 47, 47)
        #self.candy          = pygame.Rect(self.candyX,self.candyY,45,45)
        self.gap            = 170
        self.wallX          = 400
        self.counter        = 0
        self.offset         = random.randint(-110, 110)
        self.jump           = 0
        self.jumpSpeed      = 12
        self.gravity        = 4
        self.terminal       = False
        self.reward         = 0

    def updatePipes(self):
        self.wallX -= 2.5
        if self.wallX <= -48:
            self.wallX = 400
            self.counter += 1
            self.offset = random.randint(-110, 110)

    def updateCandy(self):
        self.candyX -= 2.5
        candy = pygame.Rect(self.candyX,self.candyY,45,45)
        if candy.colliderect(self.bird):
            self.candyX = random.randint(80,400)
            self.candyY = random.randint(80,550)
            self.reward = 2

        if  self.candyX <= -45:
            self.candyX = random.randint(80,400)
            self.candyY = random.randint(80,550)
    
    def updateBird(self):
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2

        self.bird[1] = self.birdY
        bottomRect = pygame.Rect(self.wallX,300 + self.gap - self.offset + 10,self.bottomPipe.get_width() - 10,
                            self.bottomPipe.get_height())
        topRect = pygame.Rect(self.wallX,0 - self.gap - self.offset - 10,self.topPipe.get_width() - 10,
                            self.topPipe.get_height())

        if bottomRect.colliderect(self.bird) or topRect.colliderect(self.bird):
            self.terminal = True


        # if bird goes outside the window
        if not 0 < self.bird[1] < 600:  
            self.bird[1] = 100
            self.birdY = 350
            self.terminal = False
            self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 5
    
    def run(self):
        pygame.font.init()
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 50)
        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONUP:
                    self.jump = 19
                    self.gravity = 4
                    self.jumpSpeed = 12

            if self.terminal:
                self.__init__()

            self.screen.fill((255,255,255))
            self.screen.blit(font.render(str(self.counter),-1,(255, 255, 255)),(180, 50))
            self.screen.blit(self.background, (0,0))
            self.screen.blit(self.bottomPipe, (self.wallX,300+self.gap-self.offset))
            self.screen.blit(self.topPipe, (self.wallX,0-self.gap-self.offset))
            self.screen.blit(self.birdSprite, (70, self.birdY))
            self.screen.blit(self.candySprite,(self.candyX,self.candyY))
                
            self.updateCandy()    
            self.updatePipes()
            self.updateBird()
            pygame.display.update()

if __name__ == "__main__":
    FlappyBird().run()