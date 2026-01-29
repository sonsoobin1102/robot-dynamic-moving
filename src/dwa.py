import math
import numpy as np
import heapq
from collections import deque
import copy

class CostmapAStar:
    def __init__(self, ox, oy, config):
        self.resolution = config.grid_resolution
        self.rr = config.robot_radius
        self.config = config
        margin = 2.0
        self.min_x = round(-config.room_width/2 - margin)
        self.min_y = round(-config.room_height/2 - margin)
        self.max_x = round(config.room_width/2 + margin)
        self.max_y = round(config.room_height/2 + margin)
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.motion = self.get_motion_model()
        
        self.static_ox = ox
        self.static_oy = oy
        self.current_ox = np.copy(ox)
        self.current_oy = np.copy(oy)
        self.static_costmap = self.calc_costmap(ox, oy)
        self.current_costmap = [row[:] for row in self.static_costmap]

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

    def update_dynamic_obs_and_plan(self, sx, sy, gx, gy, dynamic_obs_list):
        self.current_costmap = [row[:] for row in self.static_costmap]
        
        discomfort_dist = 1.0  # 양쪽 의자 겹침 효과를 위해 1.0m로 설정
        lethal_dist = 0.40     # 충돌 반경은 0.35m
        
        discomfort_grid = int(discomfort_dist / self.resolution)
        lethal_grid = int(lethal_dist / self.resolution)
        
        for obs in dynamic_obs_list: 
            cx = self.calc_xy_index(obs[0], self.min_x)
            cy = self.calc_xy_index(obs[1], self.min_y)
            
            for i in range(-discomfort_grid, discomfort_grid + 1):
                for j in range(-discomfort_grid, discomfort_grid + 1):
                    nx, ny = cx + i, cy + j
                    dist = math.hypot(i, j) * self.resolution
                    if dist > discomfort_dist: continue

                    if self.is_valid_index(nx, ny):
                        if self.current_costmap[nx][ny] < self.config.cost_lethal:
                            self.current_costmap[nx][ny] = max(self.current_costmap[nx][ny], 230)

            for i in range(-lethal_grid, lethal_grid + 1):
                for j in range(-lethal_grid, lethal_grid + 1):
                    nx, ny = cx + i, cy + j
                    dist = math.hypot(i, j) * self.resolution
                    if dist > lethal_dist: continue

                    if self.is_valid_index(nx, ny):
                        self.current_costmap[nx][ny] = self.config.cost_lethal
                        
        return self.planning(sx, sy, gx, gy)

    def calc_costmap(self, ox, oy):
        cmap = [[0 for _ in range(self.y_width)] for _ in range(self.x_width)]
        
        for x, y in zip(ox, oy):
            ix = self.calc_xy_index(x, self.min_x)
            iy = self.calc_xy_index(y, self.min_y)
            if self.is_valid_index(ix, iy):
                cmap[ix][iy] = self.config.cost_lethal
        
        inflation_cells = int(self.config.inflation_radius / self.resolution)
        
        for x, y in zip(ox, oy):
            ix = self.calc_xy_index(x, self.min_x)
            iy = self.calc_xy_index(y, self.min_y)
            
            for i in range(-inflation_cells, inflation_cells + 1):
                for j in range(-inflation_cells, inflation_cells + 1):
                    nx, ny = ix + i, iy + j
                    if not self.is_valid_index(nx, ny): continue
                    if cmap[nx][ny] == self.config.cost_lethal: continue 
                    
                    dist = math.hypot(i, j) * self.resolution
                    
                    if dist > self.config.inflation_radius: continue
                    
                    cost = int(250 * math.exp(-self.config.cost_factor * dist))
                    if cost > cmap[nx][ny]:
                        cmap[nx][ny] = cost
        return cmap

    def find_nearest_walkable_node(self, gx, gy):
        ix = self.calc_xy_index(gx, self.min_x)
        iy = self.calc_xy_index(gy, self.min_y)
        ix = max(0, min(ix, self.x_width - 1))
        iy = max(0, min(iy, self.y_width - 1))
        if self.current_costmap[ix][iy] < self.config.cost_lethal: return gx, gy 
        queue = deque([(ix, iy)])
        visited = set([(ix, iy)])
        directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        while queue:
            cx, cy = queue.popleft()
            if self.current_costmap[cx][cy] < self.config.cost_lethal:
                nx = self.calc_grid_position(cx, self.min_x)
                ny = self.calc_grid_position(cy, self.min_y)
                return nx, ny
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.x_width and 0 <= ny < self.y_width:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            if len(visited) > 5000: break
        return gx, gy 

    def planning(self, sx, sy, gx, gy):
        sx, sy = self.find_nearest_walkable_node(sx, sy)
        gx, gy = self.find_nearest_walkable_node(gx, gy)
        start_node = self.Node(self.calc_xy_index(sx, self.min_x), self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x), self.calc_xy_index(gy, self.min_y), 0.0, -1)
        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        pq = []
        heapq.heappush(pq, (self.calc_heuristic(start_node, goal_node), self.calc_grid_index(start_node)))
        while pq:
            _, c_id = heapq.heappop(pq)
            if c_id not in open_set: continue
            current = open_set[c_id]
            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                return self.calc_final_path(goal_node, closed_set)
            del open_set[c_id]
            closed_set[c_id] = current
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0], current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)
                if not self.is_valid_index(node.x, node.y): continue
                cell_cost = self.current_costmap[node.x][node.y]
                if cell_cost >= self.config.cost_lethal: continue 
                if n_id in closed_set: continue
                move_cost = self.motion[i][2] + (cell_cost * 2.0)
                new_cost = current.cost + move_cost
                if n_id not in open_set:
                    open_set[n_id] = node
                    node.cost = new_cost
                    priority = new_cost + self.calc_heuristic(goal_node, node)
                    heapq.heappush(pq, (priority, n_id))
                else:
                    if open_set[n_id].cost > new_cost:
                        open_set[n_id] = node
                        node.cost = new_cost
                        priority = new_cost + self.calc_heuristic(goal_node, node)
                        heapq.heappush(pq, (priority, n_id))
        return [], []

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set.get(parent_index)
            if n is None: break
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        rx.reverse()
        ry.reverse()
        return rx, ry

    def calc_heuristic(self, n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def calc_grid_position(self, index, min_pos):
        return index * self.resolution + min_pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return node.y * self.x_width + node.x
    
    def is_valid_index(self, x, y):
        return 0 <= x < self.x_width and 0 <= y < self.y_width

    @staticmethod
    def get_motion_model():
        return [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]

def calc_target_index(state, cx, cy, config):
    dx = [state[0] - icx for icx in cx]
    dy = [state[1] - icy for icy in cy]
    d = np.hypot(dx, dy)
    ind = np.argmin(d)
    L = 0.0
    while config.lookahead_dist > L and (ind + 1) < len(cx):
        dx = cx[ind + 1] - cx[ind]
        dy = cy[ind + 1] - cy[ind]
        L += math.hypot(dx, dy)
        ind += 1
    return ind

def dwa_control(x, config, local_goal, ob, dist_to_final_goal, stuck_mode=False):
    current_config = copy.copy(config)
    
    if stuck_mode:
        current_config.min_speed = -0.2 
        current_config.speed_cost_gain = 0.1 
        
    dw = calc_dynamic_window(x, current_config)
    u, trajectory = calc_control_and_trajectory(x, dw, current_config, local_goal, ob)
    return u, trajectory

def calc_dynamic_window(x, config):
    vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]
    vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]
    dw = [max(vs[0], vd[0]), min(vs[1], vd[1]),
          max(vs[2], vd[2]), min(vs[3], vd[3])]
    return dw

def predict_trajectory(x_init, v, y, config):
    x = np.array(x_init)
    traj = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt
    return traj

def calc_control_and_trajectory(x, dw, config, goal, ob):
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            traj = predict_trajectory(x_init, v, y, config)
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(traj, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - traj[-1, 3])
            
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost_rect(traj, ob, config)
            
            final_cost = to_goal_cost + speed_cost + ob_cost

            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = traj
    return best_u, best_trajectory

def check_collision_rect(x, y, yaw, obstacles, config):
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    
    nearby_obs_mask = (np.abs(obstacles[:,0] - x) < 1.0) & (np.abs(obstacles[:,1] - y) < 1.0)
    nearby_obs = obstacles[nearby_obs_mask]
    
    if len(nearby_obs) == 0: return False
    
    dx = nearby_obs[:, 0] - x
    dy = nearby_obs[:, 1] - y
    
    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw
    
    collision_x = np.abs(local_x) < (config.robot_length / 2.0 + config.safety_margin)
    collision_y = np.abs(local_y) < (config.robot_width / 2.0 + config.safety_margin)
    
    return np.any(collision_x & collision_y)

def calc_obstacle_cost_rect(traj, ob, config):
    if len(ob) == 0: return 0.0
    
    for i in range(len(traj)):
        x, y, yaw = traj[i, 0], traj[i, 1], traj[i, 2]
        if check_collision_rect(x, y, yaw, ob, config):
            return float("inf") 
    
    min_r = float("inf")
    last_x, last_y = traj[-1, 0], traj[-1, 1]
    
    dists = (ob[:, 0] - last_x)**2 + (ob[:, 1] - last_y)**2
    min_dist = np.sqrt(np.min(dists))
    
    return 1.0 / (min_dist + 1e-5)

def calc_to_goal_cost(traj, goal):
    dx = goal[0] - traj[-1, 0]
    dy = goal[1] - traj[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - traj[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
    return cost

def motion(x, u, dt):
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def calc_path_length(px, py):
    length = 0.0
    if len(px) < 2: return 0.0
    for i in range(len(px) - 1):
        length += math.hypot(px[i+1] - px[i], py[i+1] - py[i])
    return length
