import sys, math
import numpy as np
import random
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import gym
from gym import spaces

#from gym.envs.box2d.car_dynamics import Car

from envs.car_dynamics1 import Car, Block
#from car_dynamics1 import Car

from gym.utils import colorize, seeding, EzPickle

import pyglet
from pyglet import gl

STATE_W = 96   # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 800
WINDOW_H = 600

SCALE       = 5.0        # Track scale
TRACK_RAD   = 200/SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD   = 500/SCALE # Game over boundary
FPS         = 50         # Frames per second
ZOOM        = 2        # Camera zoom
ZOOM_FOLLOW = True       # Set to False for fixed view (don't use zoom)

NUM_OBJ = 15

TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40/SCALE
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]
GRID_COLOR = [0.2, 0.9, 0.2]
BLOCK_COLOR = [0.1, 0.1, 0.1]


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        self._contact(contact, True)
    def EndContact(self, contact):
        self._contact(contact, False)
    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData

        #print("there's a friction detect called!!!\n")

        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj  = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj  = u1
        if not tile:
            return

        if tile.block:
            tile.color[0] = 0 # ROAD_COLOR[0]
            tile.color[1] = 0 # ROAD_COLOR[1]
            tile.color[2] = 0 # ROAD_COLOR[2]
        else:
            tile.color[0] = 1 # ROAD_COLOR[0]
            tile.color[1] = 0 # ROAD_COLOR[1]
            tile.color[2] = 0 # ROAD_COLOR[2]

        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            
            obj.tiles.add(tile)
            # print tile.road_friction, "ADD", len(obj.tiles)
            # if tile.road_visited:
            #     self.env.reward -= 1
            if not tile.road_visited:
                tile.road_visited = True
                #self.env.reward += 1000.0/len(self.env.track)
                self.env.reward += 5
                self.env.tile_visited_count += 1
                i=tile.pos[0]
                j=tile.pos[1]
                self.env.simple_grid[i][j]=1

            if tile.block:
                self.env.reward -= 20

        else:
            obj.tiles.remove(tile)
            #if len(obj.tiles) ==0:
            #    self.env.reward -=10
            #print (tile.road_friction, "DEL", len(obj.tiles) )#-- should delete to zero when on grass (this works)

class CarRacing1(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, verbose=1):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.grid = None
        self.simple_grid = np.zeros((20,20))
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.fd_tile = fixtureDef(
                shape = polygonShape(vertices=
                    [(0, 0),(1, 0),(1, -1),(0, -1)]))
        self.gd_tile = fixtureDef(
                shape = polygonShape(vertices=
                    [(0, 0),(1, 0),(1, -1),(0, -1)]))
        #self.action_space = spaces.Box( np.array([-3,-3]), np.array([+3,+3]), dtype=np.float32)  # steer, gas, brake
        self.action_space = spaces.Discrete(6)

        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
        # self.observation_space = spaces.Box(low=-100, high=100, shape=(1, 403), dtype=np.float32)
        # self.observation_space = [ spaces.Box(low=0, high=2, shape=(20, 20), dtype=np.uint8),
        #                                         spaces.Box(np.array([-100,-100,-500]), np.array([100,100,500]), dtype=np.float32) ]

        self.total_timestep = 0
        self.block = []
        self.block_poly=[]
        self.time_to_die = 0
        #self.testblock = Block(self.world, 0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        
        if not self.grid:
            return
        for t in self.grid:
            self.world.DestroyBody(t)
        for t in self.block:
            self.world.DestroyBody(t)
        self.road = []
        self.grid= []
        self.grid_poly=[]
        self.simple_grid = []
        self.car.destroy()
        self.testblock.destroy()
    
    def _create_gridmap(self):
        self.grid= []
        self.grid_poly=[]
        k=PLAYFIELD/30
        #self.testblock = Block(self.world, NUM_OBJ)
        blockx=[]
        blocky=[]
        
        obj_id = random.sample(range(1, 399), NUM_OBJ+1)
        # generate the grid map without any object on it
        for x in range(-30, 30, 3):
            for y in range(-30, 30, 3):
                vertices = [(k*x+3*k*0.01, k*y+3*k*0.01),
                    (k*x+3*k*0.01, k*y+3*k*0.99),
                    (k*x+3*k*0.99, k*y+3*k*0.99),
                    (k*x+3*k*0.99, k*y+3*k*0.01)]

                self.gd_tile.shape.vertices = vertices
                t = self.world.CreateStaticBody(fixtures=self.gd_tile)
                t.userData = t

                c=0
                # make the grid with diff color 
                # c1 = 0.1*(x%6)
                # c2 = 0.1*(y%6)
                # if c1:
                #     if c2: c=0.5
                # else:
                #     if not c2: c=0.5

                t.color = [GRID_COLOR[0] + c, GRID_COLOR[1] + c, GRID_COLOR[2] + c]
                t.road_visited = False
                t.road_friction = 1.0
                t.block = False
                t.start_point = False
                t.fixtures[0].sensor = True
                t.pos = [19 - int(y/3+10), int(x/3+10)]
                self.grid.append(t)
                
                # block color
                if len(self.grid)in obj_id[0:-1]:
                    t.color = [1.0, 0.0, 0.0]
                    t.block = True
                    #i = int(x/3+10)
                    #j = 19 - int(y/3+10)
                    i=t.pos[0]
                    j=t.pos[1]
                    blockx.append(k*x+3*k*0.5)
                    blocky.append(k*y+3*k*0.5)
                    self.simple_grid[i][j]=2

                if len(self.grid)==obj_id[-1]:
                    t.color = [1.0, 0.6, 0.2]
                    t.start_point = True
                    self.start_grid = (k*x+3*k/2, k*y+3*k/2)
                    i=t.pos[0]
                    j=t.pos[1]
                    self.simple_grid[i][j]=1

                self.grid_poly.append(([
                    (k*x+3*k*0.01, k*y+3*k*0.01),
                    (k*x+3*k*0.01, k*y+3*k*0.99),
                    (k*x+3*k*0.99, k*y+3*k*0.99),
                    (k*x+3*k*0.99, k*y+3*k*0.01)
                ],t.color))

        self.testblock = Block(self.world, NUM_OBJ, blockx, blocky)
        return True


    def reset(self):
        #print('reset')
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.grid = None
        self.grid_poly = []
        self.start_grid=[]
        self.simple_grid = np.zeros((20,20))
        self.total_timestep = 0


        while True:
            #success = self._create_track()
            test = self._create_gridmap()
            if test:

                break
            if self.verbose == 1:
                print("retry to generate track (normal if there are not many of this messages)")
        #self.car = Car(self.world, *self.track[0][1:4])
        self.car = Car(self.world, 0 , self.start_grid[0],self.start_grid[1])
        #self.testblock = Block(self.world, NUM_OBJ)


        return self.step(None)[0]

    def step(self, action):
        # if self.reward < -10000:
        #     step_reward = 0
        #     done = True
        #     return self.state, step_reward, done, {}

        self.total_timestep = self.total_timestep + 1

        # if self.total_timestep == 500:
        #     self.reward += 50
        # if self.total_timestep == 1000:
        #     self.reward += 100
        # if self.total_timestep == 1500:
        #     self.reward += 150
        #     done = 3
        #     return self.state, self.reward, done, {}

        # if action is not None:
        #     self.car.steer(-action[0])
        #     self.car.gas(action[1])
        #     #self.car.brake(action[2])

        #     self.reward += 0.3*np.abs(action[1])-0.4*np.abs(action[0])
        # print("action: ", action)
        if action is not None:
            # forward
            if action == 0:
                self.car.gas(10)
                self.car.steer(0)
                #self.reward += 10
            # backward
            elif action == 1:
                self.car.gas(-10)
                self.car.steer(0)
                #self.reward += 10
            # forward + left
            elif action == 2:
                self.car.gas(0.3)
                self.car.steer(1)
                #self.reward -= 5
            # forward + right
            elif action == 3:
                self.car.gas(0.3)
                self.car.steer(-1)
                #self.reward -= 5
            # backward + left
            elif action == 4:
                self.car.gas(-0.3)
                self.car.steer(1)
                #self.reward -= 5
            # backward + right
            elif action == 5:
                self.car.gas(-0.3)
                self.car.steer(-1)
                #self.reward -= 5

        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS
        #print("simple grid:\n", self.simple_grid)
        #print('pos and angle:', self.car.hull.position, -self.car.hull.angle)
        self.state = self.render("state_pixels")
        import matplotlib.pyplot as plt
        plt.imshow(self.state)
        plt.savefig("test.jpeg")
        # angle = (-self.car.hull.angle)%(2*np.pi)
        # carinfo = np.array([self.car.hull.position.x, self.car.hull.position.y, angle]).flatten()
        # #print(carinfo)
        # #print(self.simple_grid.size)
        # #self.state = np.array(self.simple_grid, dtype = np.float32)        
        # state = np.array(self.simple_grid, dtype = np.float32).flatten()
        # #state = np.append(state,np.zeros((1,20))).reshape((21,20))
        # state = np.append(state, carinfo)

        # state[20][0] = carinfo[0]
        # state[20][1] = carinfo[1]
        # state[20][2] = carinfo[2]
        #print(state)
        # self.state = state
        #self.state = [self.simple_grid, self.car.hull.position, -self.car.hull.angle]


        step_reward = 0
        done = False
        if action is not None: # First step without action, called from reset()
            # self.reward -= 0.1 #0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            # self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            if self.tile_visited_count>=(20*20-NUM_OBJ)*0.25: #len(self.track):
                step_reward += 1000
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward -= 1000

        return self.state, step_reward, done, {}

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        zoom=2
        # zoom = 0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE*min(self.t, 1)   # Animate zoom first second
        # zoom_state  = ZOOM*SCALE*STATE_W/WINDOW_W
        # zoom_video  = ZOOM*SCALE*VIDEO_W/WINDOW_W
        
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)

        # self.transform.set_translation(
        #     WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) - scroll_y*zoom*math.sin(angle)),
        #     WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)) )
        self.transform.set_translation(WINDOW_W/2,WINDOW_H/2)


        # self.transform.set_rotation(angle)
        self.transform.set_rotation(0)
        self.car.draw(self.viewer, mode!="state_pixels")
        self.testblock.draw(self.viewer)

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode=='rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == 'human':
            win.flip()
            return self.viewer.isopen

        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(1, 1, 1, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)

        # # create the tiles on map, no longer needed
        # gl.glColor4f(0.2, 0.2, 0.2, 1.0)
        # k = PLAYFIELD/20.0
        # for x in range(-20, 20, 4):
        #     for y in range(-20, 20, 4):
        #         gl.glVertex3f(k*x+4*k*0.05, k*y+4*k*0.05, 0)
        #         gl.glVertex3f(k*x+4*k*0.05, k*y+4*k*0.95, 0)
        #         gl.glVertex3f(k*x+4*k*0.95, k*y+4*k*0.95, 0)
        #         gl.glVertex3f(k*x+4*k*0.95, k*y+4*k*0.05, 0)        

        # for poly, color in self.road_poly:
        #     gl.glColor4f(color[0], color[1], color[2], 1)
        #     for p in poly:
        #         gl.glVertex3f(p[0], p[1], 0)

        for poly, color in self.grid_poly:
            gl.glColor4f(color[0], color[1], color[2], 0.7)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)

        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W/40.0
        h = H/40.0
        gl.glColor4f(0,0,0,1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5*h, 0)
        gl.glVertex3f(0, 5*h, 0)
        gl.glVertex3f(0, 0, 0)
        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h, 0)
            gl.glVertex3f((place+0)*s, h, 0)
        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, 4*h , 0)
            gl.glVertex3f((place+val)*s, 4*h, 0)
            gl.glVertex3f((place+val)*s, 2*h, 0)
            gl.glVertex3f((place+0)*s, 2*h, 0)
        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02*true_speed, (1,1,1))
        vertical_ind(7, 0.01*self.car.wheels[0].omega, (0.0,0,1)) # ABS sensors
        vertical_ind(8, 0.01*self.car.wheels[1].omega, (0.0,0,1))
        #vertical_ind(9, 0.01*self.car.wheels[2].omega, (0.2,0,1))
        #vertical_ind(10,0.01*self.car.wheels[3].omega, (0.2,0,1))
        horiz_ind(20, -10.0*self.car.wheels[0].joint.angle, (0,1,0))
        horiz_ind(30, -0.8*self.car.hull.angularVelocity, (1,0,0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


if __name__=="__main__":
    from pyglet.window import key
    a = np.array( [0.0, 0.0] )
    da = np.array([0])
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  da[0]= +2
        if k==key.RIGHT: da[0]= +3
        if k==key.UP:    da[0]= +0
        if k==key.DOWN:  da[0]= +1


    def key_release(k, mod):

        if k==key.LEFT:  da[0]=0
        if k==key.RIGHT: da[0]=0
        if k==key.UP:    da[0]=0
        if k==key.DOWN:  da[0]=0

    env = CarRacing1()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor
        env = Monitor(env, '/tmp/video-test', force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            #print('=========DA before send: ', da[0])
            s, r, done, info = env.step(da[0])
            total_reward += r
            # if total_reward < -200:
            #     done = True
            if steps % 200 == 0 or done:
                #print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                # import matplotlib.pyplot as plt
                # plt.imshow(s)
                # plt.savefig("test.jpeg")
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()