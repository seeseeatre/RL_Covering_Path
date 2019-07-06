import sys, math
import numpy as np
import random
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import gym
from gym import spaces

#from gym.envs.box2d.car_dynamics import Car

from envs.car_dynamics1 import Car
#from car_dynamics1 import Car

from gym.utils import colorize, seeding, EzPickle

import pyglet
from pyglet import gl

# Easiest continuous control task to learn from pixels, a top-down racing environment.
# Discreet control is reasonable in this environment as well, on/off discretisation is
# fine.
#
# State consists of STATE_W x STATE_H pixels.
#
# Reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles in track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
#
# Game is solved when agent consistently gets 900+ points. Track is random every episode.
#
# Episode finishes when all tiles are visited. Car also can go outside of PLAYFIELD, that
# is far off the track, then it will get -100 and die.
#
# Some indicators shown at the bottom of the window and the state RGB buffer. From
# left to right: true speed, four ABS sensors, steering wheel position, gyroscope.
#
# To play yourself (it's rather fast for humans), type:
#
# python gym/envs/box2d/car_racing.py
#
# Remember it's powerful rear-wheel drive car, don't press accelerator and turn at the
# same time.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
STATE_W = 96   # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE       = 6.0        # Track scale
TRACK_RAD   = 200/SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD   = 500/SCALE # Game over boundary
FPS         = 60         # Frames per second
ZOOM        = 2.7        # Camera zoom
ZOOM_FOLLOW = True       # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40/SCALE
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]
GRID_COLOR = [0.2, 0.9, 0.2]

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
            tile.color[0] = 1 # ROAD_COLOR[0]
            tile.color[1] = 0 # ROAD_COLOR[1]
            tile.color[2] += 0.2 # ROAD_COLOR[2]
        else:
            tile.color[0] = 0 # ROAD_COLOR[0]
            tile.color[1] = 0 # ROAD_COLOR[1]
            tile.color[2] = 1 # ROAD_COLOR[2]

        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            
            obj.tiles.add(tile)
            # print tile.road_friction, "ADD", len(obj.tiles)
            # if tile.road_visited:
            #     self.env.reward -= 5
            if not tile.road_visited:
                tile.road_visited = True
                #self.env.reward += 1000.0/len(self.env.track)
                self.env.reward += 10
                self.env.tile_visited_count += 1
            # else:
            #     self.env.reward -= 5
            if tile.block and not obj.no_touch:
                self.env.reward -= 200

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
        self.action_space = spaces.Box( np.array([-3,-3,0]), np.array([+3,+3,+1]), dtype=np.float32)  # steer, gas, brake
        print(self.action_space)
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

        self.exp_area = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        
        if not self.grid:
            return
        for t in self.grid:
            self.world.DestroyBody(t)
        #print('destroy')
        self.road = []
        self.grid= []
        self.grid_poly=[]
        self.car.destroy()
    
    def _create_gridmap(self):
        self.grid= []
        self.grid_poly=[]
        k=PLAYFIELD/30


        NUM_OBJ=50
        obj_id = random.sample(range(1, 399), NUM_OBJ+1)
        #print(obj_id)
        # generate the grid map without any object on it
        for x in range(-30, 30, 3):
            for y in range(-30, 30, 3):
                vertices = [(k*x+3*k*0.05, k*y+3*k*0.05),
                    (k*x+3*k*0.05, k*y+3*k*0.95),
                    (k*x+3*k*0.95, k*y+3*k*0.95),
                    (k*x+3*k*0.95, k*y+3*k*0.05)]

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
                self.grid.append(t)
                
                # block color
                if len(self.grid)in obj_id[0:-1]:
                    t.color = [1.0, 0.0, 0.0]
                    t.block = True

                if len(self.grid)==obj_id[-1]:
                    t.color = [1.0, 0.6, 0.2]
                    t.start_point = True
                    self.start_grid = (k*x+3*k/2, k*y+3*k/2)

                self.grid_poly.append(([
                    (k*x+3*k*0.05, k*y+3*k*0.05),
                    (k*x+3*k*0.05, k*y+3*k*0.95),
                    (k*x+3*k*0.95, k*y+3*k*0.95),
                    (k*x+3*k*0.95, k*y+3*k*0.05)
                ],t.color))
         

        # generate the objects which the robot will die if step on
        # for obj in obj_id:
        #     self.grid[obj].color = [0.8, 0.2, 0.2]

        
        # generate a start point in the map for the robot to start it's quest


        return True

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2*math.pi*c/CHECKPOINTS + self.np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
            if c==0:
                alpha = 0
                rad = 1.5*TRACK_RAD
            if c==CHECKPOINTS-1:
                alpha = 2*math.pi*c/CHECKPOINTS
                self.start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
                rad = 1.5*TRACK_RAD
            checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )

        # print "\n".join(str(h) for h in checkpoints)
        # self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5*TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2*math.pi
            while True: # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2*math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
            while beta - alpha >  1.5*math.pi:
                 beta -= 2*math.pi
            while beta - alpha < -1.5*math.pi:
                 beta += 2*math.pi
            prev_beta = beta
            proj *= SCALE
            if proj >  0.3:
                 beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
            if proj < -0.3:
                 beta += min(TRACK_TURN_RATE, abs(0.001*proj))
            x += p1x*TRACK_DETAIL_STEP
            y += p1y*TRACK_DETAIL_STEP
            track.append( (alpha,prev_beta*0.5 + beta*0.5,x,y) )
            if laps > 4:
                 break
            no_freeze -= 1
            if no_freeze==0:
                 break
        # print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i==0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i-1][0] <= self.start_alpha
            if pass_through_start and i2==-1:
                i2 = i
            elif pass_through_start and i1==-1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
        assert i1!=-1
        assert i2!=-1

        track = track[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square( first_perp_x*(track[0][2] - track[-1][2]) ) +
            np.square( first_perp_y*(track[0][3] - track[-1][3]) ))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False]*len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i-neg-0][1]
                beta2 = track[i-neg-1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE*0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i-neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i-1]
            road1_l = (x1 - TRACK_WIDTH*math.cos(beta1), y1 - TRACK_WIDTH*math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH*math.cos(beta1), y1 + TRACK_WIDTH*math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH*math.cos(beta2), y2 - TRACK_WIDTH*math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH*math.cos(beta2), y2 + TRACK_WIDTH*math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01*(i%3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(( [road1_l, road1_r, road2_r, road2_l], t.color ))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side* TRACK_WIDTH        *math.cos(beta1), y1 + side* TRACK_WIDTH        *math.sin(beta1))
                b1_r = (x1 + side*(TRACK_WIDTH+BORDER)*math.cos(beta1), y1 + side*(TRACK_WIDTH+BORDER)*math.sin(beta1))
                b2_l = (x2 + side* TRACK_WIDTH        *math.cos(beta2), y2 + side* TRACK_WIDTH        *math.sin(beta2))
                b2_r = (x2 + side*(TRACK_WIDTH+BORDER)*math.cos(beta2), y2 + side*(TRACK_WIDTH+BORDER)*math.sin(beta2))
                self.road_poly.append(( [b1_l, b1_r, b2_r, b2_l], (1,1,1) if i%2==0 else (1,0,0) ))
        self.track = track
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
        self.exp_area = []

        while True:
            #success = self._create_track()
            test = self._create_gridmap()
            if test:

                break
            if self.verbose == 1:
                print("retry to generate track (normal if there are not many of this messages)")
        #self.car = Car(self.world, *self.track[0][1:4])
        self.car = Car(self.world, 0 , self.start_grid[0],self.start_grid[1])


        return self.step(None)[0]

    def step(self, action):
        # if self.reward < -500:
        #     step_reward = 0
        #     done = True
        #     return self.state, step_reward, done, {}

        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

            # if action[1] !=0:
            #     self.reward += 0.3*np.abs(action[1])

        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        self.state = self.render("state_pixels")

        step_reward = 0
        done = False
        if action is not None: # First step without action, called from reset()
            #self.reward -= 0.1 #0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            # self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count==300: #len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                self.reward = -1000
                step_reward = -1000

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

        zoom=3.65
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
        self.render_pixcel_map()

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

    def render_pixcel_map(self):
        from gym.envs.classic_control import rendering
        image_data = pyglet.image.get_buffer_manager().get_color_buffer()
        
        #arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        #arr = arr.reshape(VP_H, VP_W, 4)

        # render background, not necessory
        # gl.glBegin(gl.GL_QUADS)
        # gl.glColor4f(0.1, 0.8, 0.4, 1.0)
        # gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        # gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        # gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        # gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        # gl.glEnd()
        #testing render a numpy array:
        # pixcel_map = np.zeros((96, 96, 4), dtype=np.uint32)

        # for y in range(96):
        #     for x in range(96):
        #         pixcel_map[x][y][0] = 100 # np.random.randint(0,255)
        #         pixcel_map[x][y][1] = 50 # np.random.randint(0,255)
        #         pixcel_map[x][y][2] = 200 # np.random.randint(0,255)
        #         pixcel_map[x][y][3] = 250 #np.random.randint(0,255)

        # gl.glDrawPixels(96, 96, gl.GL_RGBA, gl.GL_UNSIGNED_INT , np.ascontiguousarray(pixcel_map).ctypes)

        #self.viewer.draw_line((0,0), (50,50))
        vertices = []
        self.exp_area.append((self.car.detect.transform.position.x, self.car.detect.transform.position.y, self.car.detect.transform.angle))

#This doesn't works, I don't know why=================================================================================================
        
        # for i in range(len(self.exp_area)):
        #     x1=self.exp_area[i][0]
        #     y1=self.exp_area[i][1]
        #     x2=self.exp_area[i][0]+8 * math.cos(self.exp_area[i][2])-16 * math.sin(self.exp_area[i][2])
        #     y2=self.exp_area[i][1]+16 * math.cos(self.exp_area[i][2])+8 * math.sin(self.exp_area[i][2])
        #     x3=self.exp_area[i][0]-8 * math.cos(self.exp_area[i][2])-16 * math.sin(self.exp_area[i][2])
        #     y3=self.exp_area[i][1]+16 * math.cos(self.exp_area[i][2])-8 * math.sin(self.exp_area[i][2])
        #     v=[(x1,y1),(x2,y2),(x3,y3)]
        #     #v=[(gl.GLfloat(x1),gl.GLfloat(y1)),(gl.GLfloat(x2),gl.GLfloat(y2)),(gl.GLfloat(x3),gl.GLfloat(y3))]
        #     vertices.append(v)
        # vertices = np.array(vertices)
        # VAO=gl.GLuint()
        # gl.glGenVertexArrays(1, VAO)
        # gl.glBindVertexArray(VAO)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VAO)
        # #gl.glBufferData(gl.GL_ARRAY_BUFFER, 3, vertices , gl.GL_STATIC_DRAW)
        # gl.glVertexPointer(3, gl.GL_FLOAT, 0, vertices.ctypes.data)
        
        # gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

#This works, but slows down after a while=================================================================================================
        # for i in range(max(0,len(self.exp_area)-500),len(self.exp_area)):
        #     x1=self.exp_area[i][0]
        #     y1=self.exp_area[i][1]
        #     x2=self.exp_area[i][0]+8 * math.cos(self.exp_area[i][2])-16 * math.sin(self.exp_area[i][2])
        #     y2=self.exp_area[i][1]+16 * math.cos(self.exp_area[i][2])+8 * math.sin(self.exp_area[i][2])
        #     x3=self.exp_area[i][0]-8 * math.cos(self.exp_area[i][2])-16 * math.sin(self.exp_area[i][2])
        #     y3=self.exp_area[i][1]+16 * math.cos(self.exp_area[i][2])-8 * math.sin(self.exp_area[i][2])

        #     v=[(x1,y1),(x2,y2),(x3,y3)]

        #     self.viewer.draw_polygon(v,False,color=(0.9, 0.0, 0.0))
#=================================================================================================
#This works, but slows down after a while=================================================================================================
        gl.glBegin(gl.GL_TRIANGLES)
        gl.glColor4f(0.9, 0.0, 0.0, 1.0)
        for i in range(max(0,len(self.exp_area)-500),len(self.exp_area)):
        #i=-1
            gl.glVertex2f(self.exp_area[i][0], self.exp_area[i][1])
            x= self.exp_area[i][0]+8 * math.cos(self.exp_area[i][2])-16 * math.sin(self.exp_area[i][2])
            y= self.exp_area[i][1]+16 * math.cos(self.exp_area[i][2])+8 * math.sin(self.exp_area[i][2])
            gl.glVertex2f(x,y)
            x= self.exp_area[i][0]-8 * math.cos(self.exp_area[i][2])-16 * math.sin(self.exp_area[i][2])
            y= self.exp_area[i][1]+16 * math.cos(self.exp_area[i][2])-8 * math.sin(self.exp_area[i][2])
            gl.glVertex2f(x,y)

        gl.glEnd()
#=================================================================================================


# #=======the middle=========================================================================
#         gl.glBegin(gl.GL_LINE_STRIP)
#         gl.glColor4f(0.9, 0.0, 0.0, 1.0)
#         for i in range(max(0,len(self.exp_area)-200),len(self.exp_area)):
#             gl.glVertex2f(self.exp_area[i][0], self.exp_area[i][1])
#         gl.glEnd()
# #=======the right=========================================================================
#         gl.glBegin(gl.GL_LINE_STRIP)
#         gl.glColor4f(0.0, 0.9, 0.0, 1.0)
#         for i in range(max(0,len(self.exp_area)-200),len(self.exp_area)):
#             x= self.exp_area[i][0]+8 * math.cos(self.exp_area[i][2])-16 * math.sin(self.exp_area[i][2])
#             y= self.exp_area[i][1]+16 * math.cos(self.exp_area[i][2])+8 * math.sin(self.exp_area[i][2])
#             gl.glVertex2f(x,y)
#         gl.glEnd()
# #=======the left=========================================================================
#         gl.glBegin(gl.GL_LINE_STRIP)
#         gl.glColor4f(0.0, 0.0, 0.9, 1.0)
#         for i in range(max(0,len(self.exp_area)-200),len(self.exp_area)):
#             x= self.exp_area[i][0]-8 * math.cos(self.exp_area[i][2])-16 * math.sin(self.exp_area[i][2])
#             y= self.exp_area[i][1]+16 * math.cos(self.exp_area[i][2])-8 * math.sin(self.exp_area[i][2])
#             gl.glVertex2f(x,y)
#         gl.glEnd()
# #================================================================================



        

    def render_road(self):

        
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.1, 0.3, 0.4, 1.0)
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
    a = np.array( [0.0, 0.0, 0.0] )
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -0.3
        if k==key.RIGHT: a[0] = +0.3
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[1] = -0.8   # set 1.0 for wheels to block to zero rotation
        if k==key.SPACE:  a[2] = +0.8
    def key_release(k, mod):
        if k==key.LEFT:   a[0] = 0
        if k==key.RIGHT:  a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[1] = 0
        if k==key.SPACE:  a[2] = 0
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
            s, r, done, info = env.step(a)
            total_reward += r
            # if total_reward < -200:
            #     done = True
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                #import matplotlib.pyplot as plt
                #plt.imshow(s)
                #plt.savefig("test.jpeg")
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()