import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Module Imports
from config import Config, RobotState
from amcl import SimulatedAMCL
from obstacles import PopOutChair, ChairState
from dwa import CostmapAStar, calc_target_index, dwa_control, motion, calc_path_length

class PCBangSimulator:
    def __init__(self):
        self.config = Config()
        self.start_pos = np.array([-3.0, -8.0])
        self.init_robot_state = np.array([-3.0, -8.0, math.pi / 2.0, 0.0, 0.0])
        self.x = self.init_robot_state.copy()
        self.amcl = SimulatedAMCL()
        self.seat_size = 0.8
        self.all_seats = self._generate_seat_layout()
        self.static_obs_list = [] 
        self._init_environment()
        self.state = RobotState.SERVING 
        self.interaction_msg = ""
        self.wait_timer = 0.0
        self.replanning_cooldown = 0.0 
        self.near_goal_timer = 0.0
        self.stuck_timer = 0.0
        self.msg_display_timer = 0.0
        self.progress_timer = 0.0
        self.best_dist_to_goal = float("inf")
        self.blocked_timer = 0.0
        self.has_retreated_once = False
        self.retreat_timer = 0.0
        
        # [NEW] 망설임 타이머 추가
        self.hesitate_timer = 0.0
        
        self.fig, self.ax = plt.subplots(figsize=(10, 9))
        self.trajectory = np.array([self.x])
        self.finished = False
        self.ax_btn = plt.axes([0.8, 0.01, 0.1, 0.05])
        self.btn = Button(self.ax_btn, 'Restart', color='lightgreen', hovercolor='0.975')
        self.btn.on_clicked(self.restart)
        self.astar = CostmapAStar(self.walls[:, 0], self.walls[:, 1], self.config)
        self.global_path_x, self.global_path_y = [], []
        self._set_new_goal()

    def _generate_seat_layout(self):
        seats = [] 
        seats.extend([{'x': x, 'y': 9.5, 'dir': 'down'} for x in np.linspace(-9, 9, 12)])
        rows_y = np.linspace(-3.0, 6.5, 7)
        for y in rows_y:
            seats.append({'x': -10.5, 'y': y, 'dir': 'right'})
            seats.append({'x': 10.5, 'y': y, 'dir': 'left'})
            seats.append({'x': -6.0, 'y': y, 'dir': 'left'})
            seats.append({'x': -5.0, 'y': y, 'dir': 'right'})
            seats.append({'x': -0.5, 'y': y, 'dir': 'left'})
            seats.append({'x': 0.5, 'y': y, 'dir': 'right'})
            seats.append({'x': 5.0, 'y': y, 'dir': 'left'})
            seats.append({'x': 6.0, 'y': y, 'dir': 'right'})
        return seats

    def _create_solid_block_optimized(self, x_min, x_max, y_min, y_max, step=0.1):
        xv, yv = np.meshgrid(np.arange(x_min, x_max + 0.001, step), 
                             np.arange(y_min, y_max + 0.001, step))
        return np.column_stack((xv.flatten(), yv.flatten())).tolist()

    def _init_environment(self):
        w = self.config.room_width / 2
        h = self.config.room_height / 2
        sparse_step = 0.1 
        perimeter = [[-w, -h, -w, h], [-w, h, w, h], [w, h, w, -h], [w, -h, -w, -h]]
        counter_block = self._create_solid_block_optimized(-w, -6.0, -h, -6.0, step=sparse_step)
        restroom_block = self._create_solid_block_optimized(3.0, w, -h, -6.0, step=sparse_step)
        wall_pts = []
        for p in perimeter: wall_pts.extend(self._line_to_points(p[0], p[1], p[2], p[3], step=sparse_step))
        wall_pts.extend(counter_block)
        wall_pts.extend(restroom_block)
        partition_y_start = -3.0
        partition_y_end = 6.5 
        partitions = []
        for center_x in [-5.5, 0.0, 5.5]:
            partitions.extend(self._line_to_points(center_x, partition_y_start, center_x, partition_y_end, step=sparse_step))
        wall_pts.extend(partitions) 
        self.walls = np.array(wall_pts)
        self.pop_chairs = []
        self.static_seat_obs = [] 
        self.static_obs_list = [] 
        self.valid_targets = []
        dynamic_prob = 0.25 
        for seat in self.all_seats:
            self.static_obs_list.append(seat) 
            is_safe_zone = (seat['y'] < -5.0) 
            if not is_safe_zone and random.random() < dynamic_prob:
                vx, vy = 0.0, 0.0
                if seat['dir'] == 'right': vx = 0.4
                elif seat['dir'] == 'left': vx = -0.4
                elif seat['dir'] == 'down': vy = -0.4
                self.pop_chairs.append(PopOutChair(x=seat['x'], y=seat['y'], vx=vx, vy=vy, trigger_dist=4.8, move_duration=3.5))
                block = self._create_solid_block_optimized(
                    seat['x'] - self.seat_size/2, seat['x'] + self.seat_size/2, 
                    seat['y'] - self.seat_size/2, seat['y'] + self.seat_size/2, step=0.1)
                self.static_seat_obs.extend(block)
            else:
                self.valid_targets.append(seat)
                block = self._create_solid_block_optimized(
                    seat['x'] - self.seat_size/2, seat['x'] + self.seat_size/2, 
                    seat['y'] - self.seat_size/2, seat['y'] + self.seat_size/2, step=0.1)
                self.static_seat_obs.extend(block)
        self.walls = np.vstack((self.walls, np.array(self.static_seat_obs)))

    def _line_to_points(self, x1, y1, x2, y2, step=0.1):
        points = []
        dist = math.hypot(x2-x1, y2-y1)
        n = max(int(dist / step), 1)
        for i in range(n+1):
            t = i / n
            points.append([x1 + (x2-x1)*t, y1 + (y2-y1)*t])
        return points

    def _set_new_goal(self):
        target = random.choice(self.valid_targets)
        self.target_seat = target
        
        gx, gy = 0.0, 0.0
        
        if target['y'] > 8.0: 
            gx = target['x']
            gy = 8.5 
        else: 
            gy = target['y']
            if target['x'] < -5.5:     
                gx = -8.25
            elif target['x'] < 0.0:    
                gx = -2.75
            elif target['x'] < 5.5:    
                gx = 2.75
            else:                      
                gx = 8.25

        self.goal = np.array([gx, gy])
        self.global_path_x, self.global_path_y = self.astar.planning(self.x[0], self.x[1], gx, gy)
        self.best_dist_to_goal = float("inf")

    def restart(self, event):
        print("\n[System] Restarting with V13.0...")
        self.ax.cla()
        self.x = self.init_robot_state.copy()
        self.trajectory = np.array([self.x])
        self.finished = False
        self.state = RobotState.SERVING 
        self.interaction_msg = ""
        self.replanning_cooldown = 0.0
        self.wait_timer = 0.0
        self.near_goal_timer = 0.0
        self.stuck_timer = 0.0
        self.msg_display_timer = 0.0
        self.progress_timer = 0.0
        self.best_dist_to_goal = float("inf")
        self.blocked_timer = 0.0
        self.has_retreated_once = False
        self.retreat_timer = 0.0
        self.hesitate_timer = 0.0 # 초기화
        self._init_environment()
        self.astar = CostmapAStar(self.walls[:, 0], self.walls[:, 1], self.config)
        self._set_new_goal()
        self.draw()

    def update(self):
        estimated_pose = self.amcl.get_estimated_pose(self.x)
        dwa_obs = []
        if len(self.walls) > 0:
             dwa_obs.extend(self.walls[::3]) 
        current_dynamic_obs = []
        
        if self.msg_display_timer > 0:
            self.msg_display_timer -= self.config.dt
        elif self.state != RobotState.BLOCKED and self.state != RobotState.ASKING_WAY and self.state != RobotState.TURNING_AROUND:
            self.interaction_msg = ""

        for pc in self.pop_chairs:
            pc.update(self.x[0], self.x[1], self.x[2], self.config.dt)
            
            if pc.state != ChairState.IDLE:
                obs_pts = [[pc.x, pc.y], [pc.x + 0.3, pc.y], [pc.x - 0.3, pc.y],
                           [pc.x, pc.y + 0.3], [pc.x, pc.y - 0.3]]
                base_pts = [[pc.init_x, pc.init_y], [pc.init_x + 0.2, pc.init_y], [pc.init_x - 0.2, pc.init_y]]
                dwa_obs.extend(obs_pts + base_pts)
                current_dynamic_obs.extend(obs_pts + base_pts)
            


        dwa_obs = np.array(dwa_obs)

        if not self.finished:
            if self.replanning_cooldown > 0:
                self.replanning_cooldown -= self.config.dt

            if self.state == RobotState.TURNING_AROUND:
                self.interaction_msg = "TURNING AROUND..."
                target_angle = math.atan2(self.start_pos[1] - self.x[1], self.start_pos[0] - self.x[0])
                angle_diff = target_angle - self.x[2]
                while angle_diff > math.pi: angle_diff -= 2*math.pi
                while angle_diff < -math.pi: angle_diff += 2*math.pi

                if abs(angle_diff) < 0.1:
                    print("LOG: Turn Complete! Planning Return Path...")
                    self.state = RobotState.RETURNING
                    self.goal = self.start_pos

                    self.astar.current_costmap = [row[:] for row in self.astar.static_costmap]
                    
                    self.global_path_x, self.global_path_y = self.astar.planning(self.x[0], self.x[1], self.goal[0], self.goal[1])
                    self.best_dist_to_goal = float("inf")
                else:
                    u = [0.0, 2.0 * angle_diff]
                    u[1] = max(min(u[1], self.config.max_yaw_rate), -self.config.max_yaw_rate)
                    self.x = motion(self.x, u, self.config.dt)
                    self.trajectory = np.vstack((self.trajectory, self.x))

            elif self.state == RobotState.SERVING or self.state == RobotState.RETURNING:
                path_blocked = False
                if len(self.global_path_x) > 0:
                    ind = calc_target_index(estimated_pose, self.global_path_x, self.global_path_y, self.config)
                    check_len = min(len(self.global_path_x), ind + 30) 
                    for i in range(ind, check_len):
                        px, py = self.global_path_x[i], self.global_path_y[i]
                        for pc in self.pop_chairs:
                            if pc.state in [ChairState.MOVING_OUT, ChairState.ACTIVE]:
                                if math.hypot(pc.x - px, pc.y - py) < 0.8: 
                                    path_blocked = True
                                    break
                        if path_blocked: break
                
                if path_blocked and self.replanning_cooldown <= 0:
                    print("LOG: Obstacle detected! Comparing DISTANCE (Detour vs Ask)...")
                    
                    current_idx = calc_target_index(estimated_pose, self.global_path_x, self.global_path_y, self.config)
                    current_remain_dist = calc_path_length(self.global_path_x[current_idx:], self.global_path_y[current_idx:])
                    
                    new_rx, new_ry = self.astar.update_dynamic_obs_and_plan(
                        estimated_pose[0], estimated_pose[1], self.goal[0], self.goal[1], np.array(current_dynamic_obs))
                    new_detour_dist = calc_path_length(new_rx, new_ry)

                    # 1. 우회로가 있고, 그리 멀지 않으면 -> "즉시 우회" (망설임 없이)
                    if len(new_rx) > 0 and new_detour_dist <= (current_remain_dist + self.config.detour_tolerance):
                        print(f"LOG: Short Detour found! Replanning immediately.")
                        self.interaction_msg = "OBSTACLE! REPLANNING..."
                        self.global_path_x, self.global_path_y = new_rx, new_ry
                        self.replanning_cooldown = 2.0
                        self.state = RobotState.SERVING
                        self.hesitate_timer = 0.0 # 타이머 초기화
                    
                    # 2. 우회로가 없거나 너무 멀면 -> "2초간 망설이다가 요청"
                    else:
                        self.hesitate_timer += self.config.dt
                        self.interaction_msg = "PATH BLOCKED..." # 망설이는 동안 표시될 메시지
                        
                        # 제자리 정지 (Freeze)
                        self.x = motion(self.x, [0.0, 0.0], self.config.dt)
                        self.trajectory = np.vstack((self.trajectory, self.x))
                        
                        if self.hesitate_timer > 2.0:
                            print(f"LOG: 2s Hesitation over. ASKING for way.")
                            self.state = RobotState.ASKING_WAY
                            self.blocked_timer = 0.0
                            self.hesitate_timer = 0.0
                            self.interaction_msg = "REQUESTING: PLEASE MOVE!"
                        
                        return # DWA 실행 건너뛰기
                else:
                    self.hesitate_timer = 0.0 # 길이 뚫리면 타이머 리셋

                if len(self.global_path_x) == 0:
                     self.global_path_x, self.global_path_y = self.astar.planning(self.x[0], self.x[1], self.goal[0], self.goal[1])

                if len(self.global_path_x) > 0:
                    target_ind = calc_target_index(estimated_pose, self.global_path_x, self.global_path_y, self.config)
                    local_goal = [self.global_path_x[target_ind], self.global_path_y[target_ind]]
                    dist_to_final = math.hypot(estimated_pose[0] - self.goal[0], estimated_pose[1] - self.goal[1])
                    
                    angle_to_goal = math.atan2(local_goal[1] - estimated_pose[1], local_goal[0] - estimated_pose[0])
                    angle_diff = angle_to_goal - estimated_pose[2]
                    while angle_diff > math.pi: angle_diff -= 2*math.pi
                    while angle_diff < -math.pi: angle_diff += 2*math.pi
                    
                    is_sharp_turn = abs(angle_diff) > (30.0 * math.pi / 180.0)
                    # current_config = copy.copy(self.config) # DWA handles copy
                    if is_sharp_turn:
                        # Modified copy inside DWA call or pass modified config
                        # Since DWA takes config, let's create a temporary modified one if needed.
                        # However, previous code modified a copy. Let's do that before call.
                        import copy
                        temp_config = copy.copy(self.config)
                        temp_config.max_speed = 0.05
                    else:
                        temp_config = self.config
                    
                    u = [0.0, 0.0]
                    if self.stuck_timer > 1.0:
                        u = [0.0, 0.5 * np.sign(angle_diff)] 
                        self.interaction_msg = "RECOVERING..."
                    else:
                        u, pred_traj = dwa_control(estimated_pose, temp_config, local_goal, dwa_obs, dist_to_final, False)

                    if dist_to_final < self.best_dist_to_goal - 0.1:
                        self.best_dist_to_goal = dist_to_final
                        self.progress_timer = 0.0
                    else:
                        self.progress_timer += self.config.dt
                    
                    if self.progress_timer > 15.0 and dist_to_final > 0.6:
                        print("LOG: Progress stuck. Retreating...")
                        self.state = RobotState.RETREATING
                        self.retreat_timer = 0.0
                        self.progress_timer = 0.0

                    if abs(u[0]) < 0.01 and abs(u[1]) < 0.1:
                        self.stuck_timer += self.config.dt
                    else:
                        self.stuck_timer = 0.0
                        
                    if self.stuck_timer > 5.0: 
                         self.state = RobotState.RETREATING
                         self.retreat_timer = 0.0
                    
                    self.x = motion(self.x, u, self.config.dt)
                    self.trajectory = np.vstack((self.trajectory, self.x))
                
                dist_to_final = math.hypot(estimated_pose[0] - self.goal[0], estimated_pose[1] - self.goal[1])
                if dist_to_final < 0.6:
                    self.near_goal_timer += self.config.dt
                else:
                    self.near_goal_timer = 0.0
                
                if dist_to_final <= self.config.goal_tolerance or self.near_goal_timer > 1.5:
                    print("LOG: Goal Reached!")
                    if self.state == RobotState.SERVING:
                        print("LOG: SERVING COMPLETE! NOW TURNING AROUND...")
                        self.state = RobotState.TURNING_AROUND 
                        self.near_goal_timer = 0.0
                        self.best_dist_to_goal = float("inf") 
                        self.has_retreated_once = False
                    elif self.state == RobotState.RETURNING:
                        self.finished = True

            elif self.state == RobotState.RETREATING:
                self.interaction_msg = "TACTICAL RETREAT..."

                self.x = motion(self.x, [-0.3, -0.5], self.config.dt) 

                self.retreat_timer += self.config.dt
                if self.retreat_timer > 1.5: 
                    print("LOG: Retreat Finished. Retrying Path...")
                    self.state = RobotState.SERVING
                    self.stuck_timer = 0.0
                    self.retreat_timer = 0.0
                    self.global_path_x, self.global_path_y = [], []

            elif self.state == RobotState.ASKING_WAY:
                # 1. 제자리 정지 및 메시지 설정
                self.x = motion(self.x, [0.0, 0.0], self.config.dt)
                self.interaction_msg = "REQUESTING: PLEASE MOVE!"

                # 2. 타이머를 먼저 증가시킵니다. (이 값이 올라가야 물결이 커집니다)
                self.blocked_timer += self.config.dt

                # 3. [핵심 수정] 5초가 지났는지 먼저 확인합니다.
                # 5초 전에는 이 if문 안으로 들어가지 않으므로, 의자가 움직이지 않고 물결만 계속 나옵니다.
                if self.blocked_timer >= 5.0:
                    current_idx = calc_target_index(estimated_pose, self.global_path_x, self.global_path_y, self.config)
                    # 경로 확인용 세그먼트 생성
                    path_segment_x = self.global_path_x[current_idx : current_idx + 30]
                    path_segment_y = self.global_path_y[current_idx : current_idx + 30]

                    # 의자 감지 및 이동 요청 로직
                    for pc in self.pop_chairs:
                        # 로봇과 너무 먼 의자는 패스
                        if math.hypot(pc.x - estimated_pose[0], pc.y - estimated_pose[1]) > 4.0: 
                            continue
                        
                        is_blocking = False
                        if len(path_segment_x) > 0:
                            for px, py in zip(path_segment_x, path_segment_y):
                                if math.hypot(pc.x - px, pc.y - py) < 1.0:
                                    is_blocking = True
                                    break
                        
                        # 경로를 막고 있는 의자가 확인되면 비키라고 요청
                        if is_blocking:
                            if pc.polite_retract(): # 여기서 의자가 움직임!
                                self.state = RobotState.SERVING
                                self.blocked_timer = 0.0
                                return 

                # 4. 7초가 지나도 해결이 안 되면 강제로 우회로를 찾습니다.
                if self.blocked_timer > 7.0:
                    print("LOG: Timeout. Calculating DETOUR...")
                    new_rx, new_ry = self.astar.update_dynamic_obs_and_plan(
                        estimated_pose[0], estimated_pose[1], self.goal[0], self.goal[1], np.array(current_dynamic_obs))
                    
                    if len(new_rx) > 0:
                        self.global_path_x, self.global_path_y = new_rx, new_ry
                        self.state = RobotState.SERVING 
                        self.replanning_cooldown = 2.0
                    else:
                        # 우회로도 없으면 다시 대기 (타이머 리셋)
                        self.blocked_timer = 0.0

                self.x = motion(self.x, [0.0, 0.0], self.config.dt)
                self.interaction_msg = "REQUESTING: PLEASE MOVE!"

    def draw(self):
        self.ax.cla()
        self.ax.set_aspect('equal', adjustable='box')
        w = self.config.room_width / 2
        h = self.config.room_height / 2
        self.ax.plot([-w, w, w, -w, -w], [-h, -h, h, h, -h], "-k", linewidth=3)
        counter_rect = patches.Rectangle((-w, -h), (-6.0 - -w), (-6.0 - -h), 
                                         color='lightgray', alpha=0.5, ec='black', lw=2)
        self.ax.add_patch(counter_rect)
        self.ax.text(-8.5, -8, "카운터", fontsize=9, ha='center', color='black')
        rest_rect = patches.Rectangle((3.0, -h), (w - 3.0), (-6.0 - -h), 
                                      color='lightgray', alpha=0.5, ec='black', lw=2)
        self.ax.add_patch(rest_rect)
        self.ax.text(7, -8, "화장실", fontsize=9, ha='center', color='black')
        partition_x = [-6.0, 0.0, 6.0]
        partition_x = [-5.5, 0.0, 5.5]
        for px in partition_x: self.ax.plot([px, px], [-3.0, 6.5], "-", color='gray', linewidth=2, alpha=0.5)
        for s in self.static_obs_list: 
            rect = patches.Rectangle((s['x'] - self.seat_size/2, s['y'] - self.seat_size/2), self.seat_size, self.seat_size, linewidth=1, edgecolor='k', facecolor='k')
            self.ax.add_patch(rect)
        self.ax.plot(self.goal[0], self.goal[1], "*b", markersize=15)

        for pc in self.pop_chairs:
            base_rect = patches.Rectangle((pc.init_x - self.seat_size/2, pc.init_y - self.seat_size/2), self.seat_size, self.seat_size, linewidth=1, edgecolor='gray', facecolor='white', linestyle=':')
            self.ax.add_patch(base_rect)
            if pc.state == ChairState.COOLDOWN:
                color = 'limegreen'
            else:
                color = 'm' 
            self.ax.plot(pc.x, pc.y, "o", color=color, markersize=12)
            if pc.state in [ChairState.MOVING_OUT, ChairState.ACTIVE]:
                circle = plt.Circle((pc.x, pc.y), 0.5, color='r', fill=False, linewidth=1, linestyle='--')
                self.ax.add_patch(circle)
        
        if "REQUESTING" in self.interaction_msg:
            # [수정 1] 속도를 2.0 -> 0.5로 낮춤 (아주 천천히 퍼짐)
            wave_speed = 0.5 
            
            # [수정 2] 원의 개수를 4 -> 5개로 늘려 더 풍성하게 함
            for i in range(5): 
                # [수정 3] 최대 반경을 4.0 -> 12.0으로 대폭 늘림 (% 12.0)
                # 간격도 i * 1.0 -> i * 2.0으로 넓힘
                r = (self.blocked_timer * wave_speed + i * 2.0) % 12.0
                
                # [수정 4] 반경 12.0에 맞춰 투명도가 서서히 변하도록 공식 수정
                # 제곱(**1.5)을 사용하여 물결이 멀리 가도 선명함이 좀 더 유지되게 함
                alpha = max(0, 0.6 * (1.0 - (r / 12.0)**1.5)) 
                
                signal = plt.Circle((self.x[0], self.x[1]), r, color='cyan', 
                                    fill=False, linestyle='-', linewidth=2, alpha=alpha)
                self.ax.add_patch(signal)

        if len(self.global_path_x) > 0:
            self.ax.plot(self.global_path_x, self.global_path_y, "-b", linewidth=1.0, alpha=0.5, label="Global A*")
        self.ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], "-r", linewidth=1.5, label="Driven Path")
        self._plot_robot(self.x[0], self.x[1], self.x[2])

        current_v = self.x[3]
        current_omega = self.x[4]
        current_yaw = self.x[2]

        if abs(current_v) > 0.001:
            # 1. 화살표 길이 계산 (속도에 비례)
            # visual_time: 화살표가 '몇 초 뒤의 위치'를 가리킬지 설정 (예: 1.0초 뒤)
            # scale: 화면에 잘 보이게 하기 위한 단순 확대 배수
            visual_time = 1.0 
            scale_factor = 1.0 
            
            # 2. 곡선 궤적 계산 (물리학 기반)
            # 각속도(omega)가 거의 없으면 직선으로 계산 (Divide by zero 방지)
            if abs(current_omega) < 0.001:
                dx = current_v * visual_time * math.cos(current_yaw)
                dy = current_v * visual_time * math.sin(current_yaw)
                curvature_rad = 0.0 # 직선
            else:
                # 회전 반경(Radius) = v / omega
                r = current_v / current_omega
                # 예측되는 각도 변화량
                d_yaw = current_omega * visual_time
                
                # 원 운동 공식에 따른 변위 계산
                dx = r * math.sin(current_yaw + d_yaw) - r * math.sin(current_yaw)
                dy = -r * math.cos(current_yaw + d_yaw) + r * math.cos(current_yaw)
                
                # Matplotlib FancyArrowPatch의 휨 정도(rad) 설정
                # 각속도가 클수록 더 많이 휘어지게 설정 (0.2는 시각적 조절 계수)
                curvature_rad = 0.3 * d_yaw 

            # 시각적 확대 적용
            target_x = self.x[0] + dx * scale_factor
            target_y = self.x[1] + dy * scale_factor

            # 3. FancyArrowPatch로 곡선 화살표 그리기
            # connectionstyle="arc3,rad=..."가 화살표를 휘게 만듭니다.
            curve_arrow = patches.FancyArrowPatch(
                (self.x[0], self.x[1]),  # 시작점 (로봇 위치)
                (target_x, target_y),    # 끝점 (예측 위치)
                connectionstyle=f"arc3,rad={curvature_rad}", # 휘어짐 적용
                mutation_scale=8,       # 화살표 머리 크기
                color='red',
                alpha=0.7,               # 약간 투명하게
                zorder=10,
                linewidth=2
            )
            self.ax.add_patch(curve_arrow)

        if self.finished:
            msg = "서빙 완료!"
            col = 'blue'
        elif "REQUESTING" in self.interaction_msg: 
            msg = self.interaction_msg
            col = 'red'
        elif "RECOVERING" in self.interaction_msg: 
            msg = "길 찾는 중.."
            col = 'magenta'
        elif self.state == RobotState.RETURNING:
            msg = "돌아가는 중.."
            col = 'orange'
        elif self.state == RobotState.RETREATING:
            msg = "일시적 후퇴"
            col = 'purple'
        elif self.state == RobotState.TURNING_AROUND:
            msg = "서빙 완료 후 돌아가기.."
            col = 'green'
        else:
            msg = "서빙 중.."
            col = 'green'

        self.ax.text(0, -9, msg, fontsize=15, color=col, ha='center', fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f"PC방 자율주행 로봇 시뮬레이션")
        
        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-11, 11)
        self.ax.axis('off')

    def _plot_robot(self, x, y, yaw):
        visual_width = self.config.robot_width
        visual_length = self.config.robot_length
        outline = np.array([[-visual_length / 2, visual_length / 2, (visual_length / 2), -visual_length / 2, -visual_length / 2], [visual_width / 2, visual_width / 2, - visual_width / 2, -visual_width / 2, visual_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        self.ax.plot(np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), "-k")

    def run(self):
        plt.ion()
        while plt.fignum_exists(self.fig.number):
            self.update()
            self.draw()
            plt.pause(0.001)
