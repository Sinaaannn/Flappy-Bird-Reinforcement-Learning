import pygame
import sys
import random

class FlappyBird:
    def __init__(self):
        self.screen         = pygame.display.set_mode((400, 600))
        self.birdSprite     = pygame.image.load("assets/sprites/yellowbird-upflap.png").convert_alpha()
        self.candySprite    = pygame.image.load("assets/sprites/Candy.png").convert_alpha()
        self.background     = pygame.image.load("assets/sprites/background.png").convert()
        self.topPipe        = pygame.image.load("assets/sprites/rsz_top.png").convert_alpha()
        self.bottomPipe     = pygame.image.load("assets/sprites/rsz_bottom.png").convert_alpha()
        self.topPipe2       = pygame.image.load("assets/sprites/rsz_top.png").convert_alpha()
        self.bottomPipe2    = pygame.image.load("assets/sprites/rsz_bottom.png").convert_alpha()
        self.birdX,self.birdY   = (100,350)
        self.candyX,self.candyY = (200,150)
        self.bird           = pygame.Rect(self.birdX, self.birdY, 36, 26)
        self.gap            = 170
        self.wallX          = 400
        self.secondWallX    = 640
        self.score          = 0
        self.offset         = random.randint(-110, 110)
        self.offset2        = random.randint(-100, 100)
        self.jump           = 0
        self.jumpSpeed      = 12
        self.gravity        = 4
        self.terminal       = False
        self.reward         = 0.1

    def draw_score_text(self,surf,text,size,x,y):

        font_name = pygame.font.match_font("arial",bold=1)
        font = pygame.font.Font(font_name,size)
        text_surface = font.render(text,True,(0,0,0))
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x,y)
        surf.blit(text_surface,text_rect)

    def updatePipes(self):
        self.wallX -= 2.5
        self.secondWallX -= 2.5

        if self.wallX == 10:
            self.score += 1
            reward = 2

        if self.secondWallX == 10:
            self.score += 1
            reward = 2

        if self.wallX <= -48:
            self.wallX = 450
            self.offset = random.randint(-110, 110)
            
        if self.secondWallX <= -48:
            self.secondWallX = 450
            self.offset2 = random.randint(-100, 100)
            

    def updateCandy(self):
        self.candyX -= 2.5
        candy = pygame.Rect(self.candyX,self.candyY,40,40)
        if candy.colliderect(self.bird):
            self.candyX = random.randint(80,400)
            self.candyY = random.randint(80,550)
            reward = 4

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

        bottomRect = pygame.Rect(self.wallX,300 + self.gap - self.offset + 5,60,
                            self.bottomPipe.get_height())
        topRect = pygame.Rect(self.wallX,0 - self.gap - self.offset - 7,60,
                            self.topPipe.get_height())

        bottomRect2 = pygame.Rect(self.secondWallX,300 + self.gap - self.offset2 + 5,60,
                            self.bottomPipe2.get_height())
        topRect2 = pygame.Rect(self.secondWallX,0 - self.gap - self.offset2 - 7,60,
                            self.topPipe2.get_height())

        if bottomRect.colliderect(self.bird) or topRect.colliderect(self.bird):
            self.terminal = True

        if bottomRect2.colliderect(self.bird) or topRect2.colliderect(self.bird):
            self.terminal = True

        # if bird goes outside the window
        if not 0 <= self.bird[1] <= 600:  
            self.bird[1] = 100
            self.birdY = 350
            self.terminal = True
            self.score = 0
            self.wallX = 450
            self.secondWallX = 450
            self.offset = random.randint(-110, 110)
            self.offset2 = random.randint(-100, 100)
            self.gravity = 5


    def checkInput(self,action_input):
        # 1 for jump
        # 0 for doing nothing
        if action_input[0] == 1:
            self.jump = 19
            self.gravity = 4
            self.jumpSpeed = 12
    
    def run(self,action_input): # It's gonna take action as input

        reward = 0.1
        terminal = False

        pygame.font.init()
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 50)
        
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.checkInput(action_input)

        if self.terminal:
            terminal = True
            self.__init__()
            reward = -2

        self.screen.fill((255,255,255))
        self.screen.blit(self.background, (0,0))
        self.screen.blit(self.bottomPipe, (self.wallX,300+self.gap-self.offset))
        self.screen.blit(self.bottomPipe2, (self.secondWallX,300+self.gap-self.offset2))
        self.screen.blit(self.topPipe, (self.wallX,0-self.gap-self.offset))
        self.screen.blit(self.topPipe2, (self.secondWallX,0-self.gap-self.offset2))
        self.screen.blit(self.birdSprite, (70, self.birdY))
        self.screen.blit(self.candySprite,(self.candyX,self.candyY))
            
        #self.updateCandy()
        self.candyX -= 2.5
        candy = pygame.Rect(self.candyX,self.candyY,40,40)
        if candy.colliderect(self.bird):
            self.candyX = random.randint(80,400)
            self.candyY = random.randint(80,550)
            reward = 4
        if  self.candyX <= -45:
            self.candyX = random.randint(80,400)
            self.candyY = random.randint(80,550)
        #self.updatePipes()
        self.wallX -= 2.5
        self.secondWallX -= 2.5
        if self.wallX == 10:
            self.score += 1
            reward = 2

        if self.secondWallX == 10:
            self.score += 1
            reward = 2

        if self.wallX <= -48:
            self.wallX = 450
            self.offset = random.randint(-110, 110)
            
        if self.secondWallX <= -48:
            self.secondWallX = 450
            self.offset2 = random.randint(-100, 100)
        self.updateBird()
        self.draw_score_text(self.screen,str(self.score),30,200,80)

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        return image_data, reward, terminal