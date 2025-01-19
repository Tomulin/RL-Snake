import numpy as np
import pygame
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import rcParams

# 设置字体为支持中文的字体
rcParams['font.family'] = 'SimHei'  # 或者 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 游戏参数
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 50
SNAKE_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)
BACKGROUND_COLOR = (0, 0, 0)
OBSTACLE_COLOR = (128, 128, 128)

# Q-learning参数
ALPHA_MAX = 0.4  # 初始学习率
ALPHA_MIN = 0.05  # 最小学习率
GAMMA = 0.95  # 折扣因子
EPSILON_MAX = 0.3  # 初始探索率
EPSILON_MIN = 0.01  # 最小探索率
TAU = 1800  # 退火速度参数

# 初始化pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("贪吃蛇 Q-Learning 优化版")

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(WIDTH // 2, HEIGHT // 2)]
        self.direction = (0, -GRID_SIZE)  # 初始方向向上
        self.food = self.place_food()  # 先放置食物，确保食物位置已知
        self.obstacles = self.place_obstacles()  # 再放置障碍物，确保不与食物重叠
        self.score = 0
        self.steps = 0  # 用于计数存活时间
        self.game_start_time = pygame.time.get_ticks()  # 记录游戏开始时间
        return self.get_state()

    def place_food(self):
        while True:
            x = random.randint(0, (WIDTH - GRID_SIZE) // GRID_SIZE) * GRID_SIZE
            y = random.randint(0, (HEIGHT - GRID_SIZE) // GRID_SIZE) * GRID_SIZE
            if (x, y) not in self.snake and (x, y) not in getattr(self, 'obstacles', set()):
                return (x, y)

    def place_obstacles(self):
        obstacles = set()
        num_obstacles = 10  # 设置障碍物数量
        while len(obstacles) < num_obstacles:
            x = random.randint(0, (WIDTH - GRID_SIZE) // GRID_SIZE) * GRID_SIZE
            y = random.randint(0, (HEIGHT - GRID_SIZE) // GRID_SIZE) * GRID_SIZE
            if (x, y) not in self.snake and (x, y) != self.food:
                obstacles.add((x, y))
        return obstacles

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        state = (
            (food_x - head_x) // GRID_SIZE,
            (food_y - head_y) // GRID_SIZE,
            self.direction,
            self.is_near_body(head_x, head_y),  # 检测蛇头前方是否有身体
            self.is_near_obstacle(head_x, head_y)  # 检测蛇头前方是否有障碍物
        )
        return state

    def is_near_body(self, head_x, head_y):
        # 返回一个三元素元组，表示蛇头前方、左侧、右侧是否有身体
        directions = [
            (self.direction[0], self.direction[1]),  # 前方
            (-self.direction[1], self.direction[0]),  # 左侧
            (self.direction[1], -self.direction[0])   # 右侧
        ]
        return tuple((head_x + d[0], head_y + d[1]) in self.snake for d in directions)

    def is_near_obstacle(self, head_x, head_y):
        # 检测蛇头前方、左侧、右侧是否有障碍物
        directions = [
            (self.direction[0], self.direction[1]),  # 前方
            (-self.direction[1], self.direction[0]),  # 左侧
            (self.direction[1], -self.direction[0])   # 右侧
        ]
        return tuple((head_x + d[0], head_y + d[1]) in self.obstacles for d in directions)

    def step(self, action):
        opposite_direction = {
            (0, -GRID_SIZE): (0, GRID_SIZE),
            (0, GRID_SIZE): (0, -GRID_SIZE),
            (-GRID_SIZE, 0): (GRID_SIZE, 0),
            (GRID_SIZE, 0): (-GRID_SIZE, 0)
        }

        new_direction = self.direction
        if action == 0:  # 上
            proposed_direction = (0, -GRID_SIZE)
        elif action == 1:  # 下
            proposed_direction = (0, GRID_SIZE)
        elif action == 2:  # 左
            proposed_direction = (-GRID_SIZE, 0)
        elif action == 3:  # 右
            proposed_direction = (GRID_SIZE, 0)
        else:
            proposed_direction = self.direction

        if proposed_direction != opposite_direction[self.direction]:
            new_direction = proposed_direction

        self.direction = new_direction
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        new_head = (new_head[0] % WIDTH, new_head[1] % HEIGHT)

        # 碰到障碍物或蛇身
        if new_head in self.snake or new_head in self.obstacles:
            game_time = (pygame.time.get_ticks() - self.game_start_time) / 1000  # 游戏时长（秒）
            return self.reset(), -200, True, game_time  # 撞到自己或障碍物，游戏结束，减分

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.food = self.place_food()
            return self.get_state(), 100, False, 0  # 吃到食物，奖励
        else:
            self.snake.pop()
            self.steps += 1  # 记录存活步数
            reward = -1  # 未吃到食物，轻微惩罚
            if self.steps % 10 == 0:
                reward += 5  # 每存活10步，给小额奖励
            return self.get_state(), reward, False, 0

    def draw(self):
        screen.fill(BACKGROUND_COLOR)
        for segment in self.snake:
            pygame.draw.rect(screen, SNAKE_COLOR, (segment[0], segment[1], GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, FOOD_COLOR, (self.food[0], self.food[1], GRID_SIZE, GRID_SIZE))
        for obstacle in self.obstacles:
            pygame.draw.rect(screen, OBSTACLE_COLOR, (obstacle[0], obstacle[1], GRID_SIZE, GRID_SIZE))
        pygame.display.flip()

class QLearningAgent:
    def __init__(self):
        # 使用defaultdict来动态分配内存
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.alpha = ALPHA_MAX
        self.epsilon = EPSILON_MAX
        self.avg_q_values = []
        self.max_q_values = []
        self.avg_game_durations = []

    def update_epsilon(self, episode):
        self.epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * np.exp(-episode / TAU)

    def update_alpha(self, episode):
        self.alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * np.exp(-episode / TAU)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # 探索
        return np.argmax(self.q_table[state])  # 利用

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + GAMMA * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])

    def update_avg_q_value(self):
        all_q_values = [value for values in self.q_table.values() for value in values]
        avg_q_value = np.mean(all_q_values) if all_q_values else 0
        self.avg_q_values.append(avg_q_value)

    def update_max_q_value(self):
        all_q_values = [value for values in self.q_table.values() for value in values]
        max_q_value = np.max(all_q_values) if all_q_values else 0
        self.max_q_values.append(max_q_value)

    def update_avg_game_duration(self, game_time):
        self.avg_game_durations.append(game_time)

def main():
    game = SnakeGame()
    agent = QLearningAgent()
    total_episodes = 10000
    record_interval = 1000

    for episode in range(total_episodes):
        state = game.reset()
        done = False

        total_game_time = 0  # 记录当前时间切片的总游戏时长
        episodes_in_slice = 0  # 当前时间切片的游戏轮数

        agent.update_epsilon(episode)
        agent.update_alpha(episode)

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, game_time = game.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

            total_game_time += game_time
            episodes_in_slice += 1

        if (episode + 1) % record_interval == 0:
            avg_game_time = total_game_time / episodes_in_slice if episodes_in_slice > 0 else 0
            agent.update_avg_game_duration(avg_game_time)

            agent.update_avg_q_value()
            agent.update_max_q_value()
            print(f"第 {episode + 1} 轮训练结束: 平均Q值: {agent.avg_q_values[-1]:.4f}, 最大Q值: {agent.max_q_values[-1]:.4f}, "
                  f"平均游戏时长: {avg_game_time:.2f} 秒")
            print(f"当前学习率: {agent.alpha:.4f}, 当前探索率: {agent.epsilon:.4f}")


    # 绘图
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(agent.avg_q_values, label='平均 Q 值', color='blue')
    plt.plot(agent.max_q_values, label='最大 Q 值', color='red')
    plt.xlabel('训练轮数')
    plt.ylabel('Q 值')
    plt.title('Q 值随训练轮数变化')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(agent.avg_game_durations, label='平均游戏时长', color='green')
    plt.xlabel('训练轮数')
    plt.ylabel('游戏时长 (秒)')
    plt.title('游戏时长随训练轮数变化')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 游戏展示部分
    # 为了更好地展示学习到的策略，将epsilon设为0，完全利用
    agent.epsilon = 0

    state = game.reset()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        if done:
            break

        action = agent.get_action(state)
        next_state, reward, done, game_time = game.step(action)
        game.draw()
        state = next_state
        pygame.time.delay(100)

        if done:
            print(f"游戏结束！总时长: {game_time:.2f} 秒")

    # 输出最终的学习率和探索率
    print(f"最终学习率: {agent.alpha:.4f}")
    print(f"最终探索率: {agent.epsilon:.4f}")

    pygame.quit()

if __name__ == "__main__":
    main()
