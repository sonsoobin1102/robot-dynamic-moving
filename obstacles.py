from enum import Enum
import math

class ChairState(Enum):
    IDLE = 0
    MOVING_OUT = 1
    ACTIVE = 2
    RETRACTING = 3
    COOLDOWN = 4

class PopOutChair:
    def __init__(self, x, y, vx, vy, trigger_dist=4.8, move_duration=3.5):
        self.init_x = x
        self.init_y = y
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.trigger_dist = trigger_dist
        self.move_duration = move_duration
        self.state = ChairState.IDLE
        self.timer = 0.0
        self.cooldown_timer = 0.0

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.state = ChairState.IDLE
        self.timer = 0.0
        self.cooldown_timer = 0.0

    def polite_retract(self):
        if self.state in [ChairState.MOVING_OUT, ChairState.ACTIVE]:
            self.state = ChairState.RETRACTING
            return True
        return False
    
    def update(self, robot_x, robot_y, robot_yaw, dt):
        dist = math.hypot(self.x - robot_x, self.y - robot_y)
        
        if self.state == ChairState.IDLE:
            if dist <= self.trigger_dist:
                # 시야각(FOV) 체크: 로봇이 의자 쪽을 보고 있을 때만(+-100도) 반응
                angle_to_chair = math.atan2(self.y - robot_y, self.x - robot_x)
                angle_diff = angle_to_chair - robot_yaw
                while angle_diff > math.pi: angle_diff -= 2 * math.pi
                while angle_diff < -math.pi: angle_diff += 2 * math.pi
                
                if abs(angle_diff) < (100.0 * math.pi / 180.0):
                    self.state = ChairState.MOVING_OUT
                    self.timer = 0.0

        elif self.state == ChairState.MOVING_OUT:
            if self.timer < self.move_duration:
                self.x += self.vx * dt
                self.y += self.vy * dt
                self.timer += dt
            else:
                self.state = ChairState.ACTIVE

        elif self.state == ChairState.ACTIVE:
            pass 

        elif self.state == ChairState.RETRACTING:
            retract_speed = math.hypot(self.vx, self.vy) 
            dx = self.init_x - self.x
            dy = self.init_y - self.y
            d_remain = math.hypot(dx, dy)
            if d_remain < 0.05:
                self.x = self.init_x
                self.y = self.init_y
                self.state = ChairState.COOLDOWN
                self.cooldown_timer = 20.0 
            else:
                self.x += (dx / d_remain) * retract_speed * dt
                self.y += (dy / d_remain) * retract_speed * dt

        elif self.state == ChairState.COOLDOWN:
            self.cooldown_timer -= dt
            if self.cooldown_timer <= 0:
                self.state = ChairState.IDLE
