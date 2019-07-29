import sys, math
import numpy as np
import random
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import gym
from gym import spaces

from envs.car_dynamics1 import Car, Block


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
FPS         = 60         # Frames per second
ZOOM        = 1        # Camera zoom
ZOOM_FOLLOW = True       # Set to False for fixed view (don't use zoom)
X_MAX = int(WINDOW_W/2+PLAYFIELD*ZOOM)
X_MIN = int(WINDOW_W/2-PLAYFIELD*ZOOM)
Y_MAX = int(WINDOW_H/2+PLAYFIELD*ZOOM)
Y_MIN = int(WINDOW_H/2-PLAYFIELD*ZOOM)


TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40/SCALE
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]
GRID_COLOR = [0.2, 0.9, 0.2]
BLOCK_COLOR = [0.1, 0.1, 0.1]

NUM_OBJ=15

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

        if u1 and "block" in u1.__dict__:
            tile = u1
            obj  = u2
        if u2 and "block" in u2.__dict__:
            tile = u2
            obj  = u1
        if not tile:
            return

        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            if tile.block==True and not obj.no_touch:
                self.env.reward -= 200

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
        self.px_tile = fixtureDef(
                shape = polygonShape(vertices=
                    [(0, 0),(1, 0),(1, -1),(0, -1)]))
        #self.action_space = spaces.Box( np.array([-3,-3,0]), np.array([+3,+3,+1]), dtype=np.float32)  # steer, gas, brake
        self.action_space = spaces.Discrete(6)
        #self.observation_space = spaces.Box(low=0, high=1, shape=(int(PLAYFIELD*ZOOM*2*PLAYFIELD*ZOOM*2)+1,3), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(96,96,3), dtype=np.float32)
        #self.observation_space = spaces.Box(low=0, high=1, shape=(int(PLAYFIELD*ZOOM*2),int(PLAYFIELD*ZOOM*2),3), dtype=np.float32)
        self.pixcel_map = np.zeros((WINDOW_H, WINDOW_W, 4), dtype=np.uint8)
        self.exp_area = []
        self.last_count = 0
        self.block = []
        self.block_poly=[]
        self.time_to_die = 0
        #self.testblock = Block(self.world, 0 , 0,0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        
        for t in self.block:
            self.world.DestroyBody(t)
        self.road = []
        self.grid= []
        self.grid_poly=[]
        self.block = []
        self.block_poly=[]
        self.pixcel_map = []
        if self.car:
            self.car.destroy()
        #self.testblock.destroy()

    def _create_pixcel_block(self):

        #x = np.random.uniform(-PLAYFIELD+15, PLAYFIELD-15, NUM_OBJ+1)
        #y = np.random.uniform(-PLAYFIELD+15, PLAYFIELD-15, NUM_OBJ+1)
        x = [70, 25, -1, -42, -50, -66, -60, 76, 38, -59, 38, 50, 88, -88, 32, 0]
        y = [39, -60, 58, 20, -50, -66, -40, 38, -27, 38, 67, -22, -16, 19, 27, 0]
        for b in range(NUM_OBJ): 
            vertices = [(x[b]+5, y[b]+5),
                    (x[b]+5, y[b]-5),
                    (x[b]-5, y[b]-5),
                    (x[b]-5, y[b]+5)]
            self.gd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.gd_tile)
            t.userData = t
            t.color = BLOCK_COLOR
            t.road_visited = False
            t.road_friction = 1.0
            t.block = True
            t.start_point = False
            t.fixtures[0].sensor = True
             
            self.block.append(t)
            self.block_poly.append([vertices, t.color])

            bbxx = int((x[b]+5)*ZOOM + WINDOW_W/2)
            bbxn = int((x[b]-5)*ZOOM + WINDOW_W/2)
            bbyx = int((y[b]+5)*ZOOM + WINDOW_H/2)
            bbyn = int((y[b]-5)*ZOOM + WINDOW_H/2)
            for i in range(bbxn, bbxx):
                for j in range(bbyn, bbyx):
                    self.pixcel_map[j][i]=[255,99,99,99]


        self.start_grid =(x[-1], y[-1])
        return True

    def reset(self):
        self._destroy()
        self.state = []
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.grid = None
        self.grid_poly = []
        self.start_grid=[]
        self.exp_area = []
        self.pixcel_map = np.zeros((WINDOW_H, WINDOW_W, 4), dtype=np.uint8)
        self.time_to_die =0
        self.last_count = 0
        while True:
            test = self._create_pixcel_block()
            if test:

                break
            if self.verbose == 1:
                print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, 0 , 0,0)
        #self.testblock = Block(self.world, 0 , self.start_grid[0]+50,self.start_grid[1]+50)
        self.time_to_die = 0
        return self.step(None)[0]

    def step(self, action):
        self.time_to_die += 1

        # if self.reward < -500:
        #     step_reward = 0
        #     done = True
        #     return self.state, step_reward, done, {}
        #self.reward += 0.05 #0.1
        if action is not None:
        # forward
            if action == 0:
                self.car.gas(10.5)
                self.car.steer(0)
                #self.reward += 1
            # backward
            elif action == 1:
                self.car.gas(-10.5)
                self.car.steer(0)
                #self.reward += 1
            # forward + left
            elif action == 2:
                self.car.gas(0.5)
                self.car.steer(1)
                #self.reward -= 5
            # forward + right
            elif action == 3:
                self.car.gas(0.5)
                self.car.steer(-1)
                #self.reward -= 5
            # backward + left
            elif action == 4:
                self.car.gas(-0.5)
                self.car.steer(1)
                #self.reward -= 5
            # backward + right
            elif action == 5:
                self.car.gas(-0.5)
                self.car.steer(-1)
                #self.reward -= 5

        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        x, y = self.car.hull.position

        px = (x+PLAYFIELD)/(2*PLAYFIELD)
        py = (y+PLAYFIELD)/(2*PLAYFIELD)
        pangle = (self.car.hull.angle%(2*math.pi))/(2*math.pi)

        # if abs(x) >= PLAYFIELD*0.8 and abs(y) >= PLAYFIELD*0.8:
        #     self.reward -= 1
        # else:
        #     self.reward -= 0.1
        # if abs(x) >= PLAYFIELD*0.9 or abs(y) >= PLAYFIELD*0.9:
        #     self.reward -= 10
        x=(x*ZOOM+WINDOW_W/2)
        y=(y*ZOOM+WINDOW_H/2)

        carinfo = [px,py,pangle]
        x = min(int(x),598)
        x = max(int(x),1)
        y = min(int(y),598)
        y = max(int(y),1)
        if self.pixcel_map[y][x][1] != 99:
            self.pixcel_map[y][x] = [255,255,0,0]
            self.pixcel_map[y][x+1] = [255,255,0,0]
            self.pixcel_map[y][x-1] = [255,255,0,0]
            self.pixcel_map[y+1][x] = [255,255,0,0]
            self.pixcel_map[y-1][x] = [255,255,0,0]
        
        temp_map = self.pixcel_map
        temp_map[Y_MIN-10:Y_MAX+10,X_MIN-10:X_MIN,:] = [255,99,99,99]
        temp_map[Y_MIN-10:Y_MAX+10,X_MAX:X_MAX+10,:] = [255,99,99,99]
        temp_map[Y_MIN-10:Y_MIN,X_MIN-10:X_MAX+10,:] = [255,99,99,99]
        temp_map[Y_MAX:Y_MAX+10,X_MIN-10:X_MAX+10,:] = [255,99,99,99]
        state = temp_map[y-48:y+48,x-48:x+48,0:4]

        state = state[:,:,::-1]/255
        self.state = state[:,:,0:3]
        # self.state = self.state.reshape(int(PLAYFIELD*ZOOM*2*PLAYFIELD*ZOOM*2),3)
        # self.state = np.vstack([carinfo,self.state])

        #self.state = self.render("state_pixels")

        step_reward = 0
        done = False
        map_area = self.pixcel_map[Y_MIN:Y_MAX,X_MIN:X_MAX,0]
        count = np.count_nonzero(map_area)
        #2=top left, 1 to left bottom, 4=right bottom
        #print(map_area[0][0],map_area[599][0],map_area[599][599],map_area[0][599])
        #print(count/360000*100,"%")
        # if (count - self.last_count) <= 10:
        #     self.reward -= 0.1
        self.reward += np.abs(count - self.last_count)/10
        
            
        #self.reward += count/10000
        self.last_count = count

        if self.time_to_die >= 2000:
            #print("time to die!")
            #self.reward += 500
            self.time_to_die = 0
            done = True

        if action is not None: # First step without action, called from reset()
            self.reward += 0.1 #0.1
            step_reward = self.reward - self.prev_reward
            # if step_reward < 1:
            #     step_reward -= 0.5
            # if step_reward > 1:
            #     step_reward = step_reward * step_reward
            self.prev_reward = self.reward
            #step_reward += count/10
            # if step_reward < -100:
            #     step_reward -= 1000
            #     done = True

            # if self.tile_visited_count==6:
            #     step_reward += 1000
            #     done = True
            if count>=(40000-NUM_OBJ*20*20)*0.5:
                step_reward += 1000
                done = True
            x, y = self.car.hull.position
            # if abs(x) > PLAYFIELD*0.9 or abs(y) > PLAYFIELD*0.9:
            #     step_reward -= 100
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward -= 100

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

        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(ZOOM, ZOOM)

        self.transform.set_translation(WINDOW_W/2,WINDOW_H/2)

        self.transform.set_rotation(0)
        self.car.draw(self.viewer, mode!="state_pixels")
        #self.testblock.draw(self.viewer)

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

        self.render_background(VP_W, VP_H)
        self.render_pixcel_map(VP_W, VP_H)
        self.render_road(VP_W, VP_H)

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
        #arr = arr[int(VP_H/2-PLAYFIELD*ZOOM):int(VP_H/2+PLAYFIELD*ZOOM),int(VP_W/2-PLAYFIELD*ZOOM):int(VP_W/2+PLAYFIELD*ZOOM),:]
        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # render background, not necessory=======================================================================
    def render_background(self, VP_W, VP_H):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glEnd()

    def render_pixcel_map(self, VP_W, VP_H):
        #from gym.envs.classic_control import rendering

        # this will get the color buffer from opengl rendering process
        # image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()

        # this record the position and angle of the middle vertex of the triangle
        # self.exp_area.append((self.car.detect.transform.position.x, self.car.detect.transform.position.y, self.car.detect.transform.angle))
        # i=-1
        self.exp_area=[(self.car.detect.transform.position.x, self.car.detect.transform.position.y, self.car.detect.transform.angle)]
        i=0
        x1=self.exp_area[i][0]
        y1=self.exp_area[i][1]
        x2=self.exp_area[i][0]+8 * math.cos(self.exp_area[i][2])-16 * math.sin(self.exp_area[i][2])
        y2=self.exp_area[i][1]+16 * math.cos(self.exp_area[i][2])+8 * math.sin(self.exp_area[i][2])
        x3=self.exp_area[i][0]-8 * math.cos(self.exp_area[i][2])-16 * math.sin(self.exp_area[i][2])
        y3=self.exp_area[i][1]+16 * math.cos(self.exp_area[i][2])-8 * math.sin(self.exp_area[i][2])
        v=np.array([[x1,y1],[x2,y2],[x3,y3]])
        v=v*ZOOM
        a=np.array([WINDOW_W/2, WINDOW_H/2]*3).reshape((3,2)).astype(dtype=np.uint32)
        v=np.add(v,a).astype(dtype=np.uint32)

        def area(x1, y1, x2, y2, x3, y3): 
            return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
  
        
        # A function to check whether point P(x, y) 
        # lies inside the triangle formed by  
        # A(x1, y1), B(x2, y2) and C(x3, y3)  
        def isInside(x1, y1, x2, y2, x3, y3, x, y): 
            x1, y1, x2, y2, x3, y3, x, y = float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x), float(y)
            # Calculate area of triangle ABC 
            A = area (x1, y1, x2, y2, x3, y3) 
        
            # Calculate area of triangle PBC  
            A1 = area (x, y, x2, y2, x3, y3) 
            
            # Calculate area of triangle PAC  
            A2 = area (x1, y1, x, y, x3, y3) 
            
            # Calculate area of triangle PAB  
            A3 = area (x1, y1, x2, y2, x, y) 
            
            # Check if sum of A1, A2 and A3  
            # is same as A 
            if(A == A1 + A2 + A3): 
                return True
            else: 
                return False
            

        x_max, y_max = np.max(v, axis=0)
        x_max = int(min(x_max, VP_W/2+PLAYFIELD*ZOOM))
        y_max = int(min(y_max, VP_H/2+PLAYFIELD*ZOOM))
        x_min, y_min = np.min(v, axis=0)
        x_min = int(max(x_min, VP_W/2-PLAYFIELD*ZOOM))
        y_min = int(max(y_min, VP_H/2-PLAYFIELD*ZOOM))

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if self.pixcel_map[y][x][1] == 99:
                    continue
                elif isInside(v[0][0],v[0][1],v[1][0],v[1][1],v[2][0],v[2][1],x,y):
                    self.pixcel_map[y][x] = [255,0,0,255]
        #self.pixcel_map[v[0][1]][v[0][0]] = [0,255,0,255]
        gl.glDrawPixels(VP_W, VP_H, gl.GL_RGBA, gl.GL_UNSIGNED_INT_8_8_8_8 , np.ascontiguousarray(self.pixcel_map).ctypes)

    def render_road(self, VP_W, VP_H):

        
        gl.glBegin(gl.GL_QUADS)
        
        for poly, color in self.block_poly:
            gl.glColor4f(color[0], color[1], color[2], 0.5)
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
            s, r, done, info = env.step(da[0])
            total_reward += r
            # if total_reward < -200:
            #     done = True
            if steps % 200 == 0 or done:
                #print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                import matplotlib.pyplot as plt
                plt.imshow(s)
                plt.savefig("test.jpeg")
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()