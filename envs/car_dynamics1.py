import numpy as np
import math
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)

# Top-down car dynamics simulation.
#
# Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
# This simulation is a bit more detailed, with wheels rotation.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

SIZE = 0.04
ENGINE_POWER            = 100000000*SIZE*SIZE
WHEEL_MOMENT_OF_INERTIA = 4000*SIZE*SIZE
FRICTION_LIMIT          = 5000000*SIZE*SIZE     # friction ~= mass ~= size^2 (calculated implicitly using density)
WHEEL_R  = 27
WHEEL_W  = 14

WHEELPOS = [
    #(-55,+80), (+55,+80),
    (-55,0), (+55,0)
    ]
# the hull
HULL_POLY =[
    (-20,80),(-60,0),(-40,-40),
    (40,-40),(60,0),(20,80)
    ]

VIDUAL_DETECT = [
    (0,0),
    (-2,4),
    (2,4)
    ]

SQUARE_BLOCK = [
    (-5,-5),
    (-5,5),
    (5,5),
    (5,-5)
]

WALL_POSE = [(0,-105),(-105,0),(0,105),(105,0)]

WHEEL_COLOR = (0.2,0.0,0.3)
WHEEL_WHITE = (0.3,0.3,0.3)
MUD_COLOR   = (0.4,0.4,0.0)
BLOCK_COLOR = (0.2,0.2,0.2)

class Car:
    def __init__(self, world, init_angle, init_x, init_y):
        self.world = world
        #======================================================
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            angle = init_angle,
            fixtures = [ 
                 fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY ]), density=1.0)#,
                #  fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in VIDUAL_DETECT ]), density=0.0)
                ]
            )
        self.hull.color = (0.0,0.0,1.0)
        self.hull.no_touch = False
        self.hull.userData = self.hull
        #======================================================
        #======================================================
        # self.detect = self.world.CreateDynamicBody(
        #     position = (init_x, init_y),
        #     angle = self.hull.angle,
        #     fixtures = [ 
        #         fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in VIDUAL_DETECT ]), density=0.001)
        #        ]
        #     )
        # rjd = revoluteJointDef(
        #         bodyA=self.detect,
        #         bodyB=self.hull,
        #         localAnchorA=(0,-3),
        #         localAnchorB=(0,0),
        #         enableMotor=True,
        #         enableLimit=True,
        #         maxMotorTorque=0,
        #         motorSpeed = 0,
        #         lowerAngle = 0,
        #         upperAngle = 0,
        #         )
        # self.detect.joint = self.world.CreateJoint(rjd)
        # self.detect.no_touch = True
        # self.detect.color = (0.0,1.0,0.0)
        #======================================================
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W,+WHEEL_R), (+WHEEL_W,+WHEEL_R),
            (+WHEEL_W,-WHEEL_R), (-WHEEL_W,-WHEEL_R)
            ]
        for wx,wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position = (init_x+wx*SIZE, init_y+wy*SIZE),
                angle = init_angle,
                fixtures = fixtureDef(
                    shape=polygonShape(vertices=[ (x*front_k*SIZE,y*front_k*SIZE) for x,y in WHEEL_POLY ]),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0)
                    )
            w.wheel_rad = front_k*WHEEL_R*SIZE
            w.color = WHEEL_COLOR
            w.gas   = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            w.no_touch = False
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx*SIZE,wy*SIZE),
                localAnchorB=(0,0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180*900*SIZE*SIZE,
                motorSpeed = 0,
                lowerAngle = 0,
                upperAngle = 0,
                )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        #======================================================
        self.drawlist =  self.wheels + [self.hull] #+ [self.detect] 
        self.particles = []

    def gas(self, gas):
        'control: rear wheel drive'
        gas = np.clip(gas, -2, 2)
        for w in self.wheels:
            diff = gas - w.gas
            #if   gas > 0 and diff > +0.1: diff = +0.1  # gradually increase, but stop immediately
            if gas > 0: diff = min(diff,0.2)
            #elif gas < 0 and diff < -0.1: diff = -0.1  # no longer stops immediately
            elif gas < 0: diff = max(diff,-0.2)
            w.gas += diff
            #w.gas = gas
            #print(w.gas,'\n')
                

    def brake(self, b):
        'control: brake b=0..1, more than 0.9 blocks wheels to zero rotation'
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        'control: steer s=-1..1, it takes time to rotate steering wheel from side to side, s is target position'
        # the following code works, but can only turn when the car is accelrating.
        # self.wheels[0].steer = s
        # self.wheels[1].steer = s
        # s = np.clip(s, -1, 1)            
        # self.wheels[0].gas = -s*0.5
        # self.wheels[1].gas = s*0.5
        s = np.clip(s, -0.3, 0.3)
        if s != 0:
            self.wheels[0].gas = -s
            self.wheels[1].gas = s
            self.wheels[0].steer = s
            self.wheels[1].steer = s
        else:
            self.wheels[0].steer = 0
            self.wheels[1].steer = 0


        #print(self.wheels[0].gas,self.wheels[1].gas)

    def step(self, dt):
        #print(self.wheels[0].gas,self.wheels[1].gas)
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir*min(50.0*val, 3.0)

            # Position => friction_limit
            grass = True
            friction_limit = FRICTION_LIMIT*0.6  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(friction_limit, FRICTION_LIMIT*tile.road_friction)
                grass = False

            # Force
            forw = w.GetWorldVector( (0,1) )
            side = w.GetWorldVector( (1,0) )
            v = w.linearVelocity
            vf = forw[0]*v[0] + forw[1]*v[1]  # forward speed
            vs = side[0]*v[0] + side[1]*v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega
            w.omega += dt*ENGINE_POWER*w.gas/WHEEL_MOMENT_OF_INERTIA/(abs(w.omega)+5.0)  # small coef not to divide by zero
            # self.fuel_spent += dt*ENGINE_POWER*w.gas
            
            if w.gas == 0:
                w.omega = 0

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15    # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE*w.brake
                if abs(val) > abs(w.omega): val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir*val
            else:
                BRAKE_FORCE = 3    # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE*0.6
                if abs(val) > abs(w.omega): val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir*val

            w.phase += w.omega*dt

            vr = w.omega*w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr        # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.
            f_force *= 205000*SIZE*SIZE  # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            p_force *= 205000*SIZE*SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt*f_force*w.wheel_rad/WHEEL_MOMENT_OF_INERTIA

            w.ApplyForceToCenter( (
                p_force*side[0] + f_force*forw[0],
                p_force*side[1] + f_force*forw[1]), True )

    def draw(self, viewer, draw_particles=True):
        if draw_particles:
            for p in self.particles:
                viewer.draw_polyline(p.poly, color=p.color, linewidth=5)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=obj.color)
                if "phase" not in obj.__dict__: continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1>0 and s2>0: continue
                if s1>0: c1 = np.sign(c1)
                if s2>0: c2 = np.sign(c2)
                white_poly = [
                    (-WHEEL_W*SIZE, +WHEEL_R*c1*SIZE), (+WHEEL_W*SIZE, +WHEEL_R*c1*SIZE),
                    (+WHEEL_W*SIZE, +WHEEL_R*c2*SIZE), (-WHEEL_W*SIZE, +WHEEL_R*c2*SIZE)
                    ]
                viewer.draw_polygon([trans*v for v in white_poly], color=WHEEL_WHITE)

    def _create_particle(self, point1, point2, grass):
        class Particle:
            pass
        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0],point1[1]), (point2[0],point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        #self.world.DestroyBody(self.detect)
        #self.detect = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []


class Block:
    def __init__(self, world, NUM_OBJ, blockx=[], blocky=[]):
        self.world = world
        self.drawlist = []
        for w in range (4):
            if w%2 == 0:
                self.wallBlock = self.world.CreateBody(
                    position = WALL_POSE[w],
                    angle = 0,
                    fixtures = [ 
                        fixtureDef(shape = polygonShape(vertices=[ (x*21,y) for x,y in SQUARE_BLOCK ]), density=1.0)
                        ]
                )
            else:
                self.wallBlock = self.world.CreateBody(
                    position = WALL_POSE[w],
                    angle = 0,
                    fixtures = [ 
                        fixtureDef(shape = polygonShape(vertices=[ (x,y*21) for x,y in SQUARE_BLOCK ]), density=1.0)
                        ]
                )
            self.wallBlock.color = BLOCK_COLOR
            self.wallBlock.userData = self.wallBlock
            #==========================================
            
            self.drawlist.append(self.wallBlock)

        #x = [70, 25, -1, -42, -50, -66, -60, 76, 38, -59, 38, 50, 88, -88, 32, 0]
        #y = [39, -60, 58, 20, -50, -66, -40, 38, -27, 38, 67, -22, -16, 19, 27, 0]
        x=blockx
        y=blocky
        for b in range(NUM_OBJ):
        #===============define a object============
            self.shapeBlock = self.world.CreateBody(
                position = (x[b], y[b]) if len(x)>0 else (int(np.random.random_integers(-90,90)), int(np.random.random_integers(-90,90))), #(int(np.random.random_integers(-90,90)), int(np.random.random_integers(-90,90))),  #
                angle = 0, #np.random.random()*6,
                fixtures = [ 
                    fixtureDef(shape = polygonShape(vertices=[ (x,y) for x,y in SQUARE_BLOCK ]), density=1.0)
                    ]
            )
            self.shapeBlock.color = BLOCK_COLOR
            #self.shapeBlock.color2 = (0.4,0.4,0.4)
            #==========================================
            self.shapeBlock.userData = self.shapeBlock
            self.drawlist.append(self.shapeBlock)
            

    def draw(self, viewer):
        from gym.envs.classic_control import rendering

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=obj.color)
                # trans = f.body.transform
                # t = rendering.Transform(translation=trans*f.shape.pos)
                # viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                # viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)


    def destroy(self):
        for b in range(len(self.drawlist)):
            self.world.DestroyBody(self.drawlist[b])
        self.drawlist = []
        # self.world.DestroyBody(self.shapeBlock)
        # self.shapeBlock = None
