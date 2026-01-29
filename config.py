from enum import Enum
import math

class RobotType(Enum):
    circle = 0
    rectangle = 1

class RobotState(Enum):
    SERVING = 0
    TURNING_AROUND = 1
    RETURNING = 2
    BACKWARD = 3
    WAITING = 4     
    BLOCKED = 5      
    RETREATING = 6
    RECOVERY = 7
    ASKING_WAY = 8 

class Config:
    def __init__(self):
        self.max_speed = 0.7
        self.min_speed = -0.2 # 후진 허용
        self.max_yaw_rate = 100.0 * math.pi / 180.0
        self.max_accel = 1.0
        self.max_delta_yaw_rate = 100.0 * math.pi / 180.0
        
        self.v_resolution = 0.02           
        self.yaw_rate_resolution = 1.0 * math.pi / 180.0 
        self.dt = 0.1
        self.predict_time = 1.0            
        
        self.to_goal_cost_gain = 0.5   # 경로보다 안전 우선
        self.speed_cost_gain = 3.0
        self.obstacle_cost_gain = 1.0  # 장애물 회피 성향 강화
        
        self.robot_stuck_flag_cons = 0.001
        self.robot_width = 0.48  
        self.robot_length = 0.52 
        self.safety_margin = 0.02 
        
        self.robot_radius = 0.35
        
        self.room_width = 22.0
        self.room_height = 20.0
        self.goal_tolerance = 0.40 
        
        self.grid_resolution = 0.1
        self.inflation_radius = 0.60   # 좁은 길 통과와 모서리 회전의 타협점
        
        self.cost_lethal = 254
        self.cost_factor = 3.0
        self.lookahead_dist = 1.5      # 모서리 미리 감지 거리 증가
        
        self.detour_tolerance = 10.0   # 복도 한 칸 길이 정도의 우회 허용
