from random import randint
import pygame
from constants import COLORS_RGB, COLORS_WEIGHT, ORIGINAL_COLORS


class Spot:
    """
    网格点类，表示迷宫中的一个单元格
    Grid point class, representing a cell in the maze

    Attributes:
        i (int): x坐标 - x coordinate
        j (int): y坐标 - y coordinate
        color (tuple): RGB颜色值 - RGB color value
        style (int): 绘制样式（0表示填充，大于0表示边框宽度）- Drawing style (0 for filled, >0 for border width)
    """

    def __init__(self, x: int, y: int, color: tuple, style: int = 0):
        """
        初始化网格点
        Initialize grid point

        Args:
            x (int): x坐标 - x coordinate
            y (int): y坐标 - y coordinate
            color (tuple): RGB颜色值 - RGB color value
            style (int, optional): 绘制样式，默认为0（填充）- Drawing style, default is 0 (filled)
        """
        self.i = x
        self.j = y
        self.color = color
        self.style = style

    def show(self, screen, grid_length: float, color: tuple = None, style: int = None):
        """
        在屏幕上显示网格点
        Display grid point on screen

        Args:
            screen: Pygame显示表面 - Pygame display surface
            grid_length (float): 网格边长 - Grid side length
            color (tuple, optional): 覆盖颜色，默认为None（使用自身颜色）
                                     Override color, default None (use self color)
            style (int, optional): 覆盖样式，默认为None（使用自身样式）
                                   Override style, default None (use self style)
        """
        # 使用指定颜色或默认颜色 - Use specified color or default color
        draw_color = color if color else self.color
        # 使用指定样式或默认样式 - Use specified style or default style
        draw_style = style if style is not None else self.style

        # 绘制矩形表示网格点 - Draw rectangle to represent grid point
        pygame.draw.rect(
            screen, draw_color,
            (self.i * grid_length, self.j * grid_length, grid_length, grid_length),
            draw_style
        )


class MazeGenerator:
    """
    迷宫生成和管理类
    Maze generation and management class

    负责创建随机迷宫、管理网格状态和可视化显示
    Responsible for creating random mazes, managing grid states and visual display
    """

    def __init__(self, grid_num: int, length: float, seed: int = 42):
        """
        初始化迷宫生成器
        Initialize maze generator

        Args:
            grid_num (int): 网格数量（每行每列的格子数）- Number of grids (cells per row/column)
            length (float): 界面总像素大小 - Total pixel size of the interface
        """
        self.grid_num = grid_num
        self.length = length
        self.grid_length = length / grid_num  # 计算单个网格的像素长度 - Calculate pixel length of single grid
        self.grids = None  # 网格点二维数组 - 2D array of grid points
        self.weight_list = None  # 权重二维数组 - 2D array of weights
        self.maze = None  # 迷宫数据二维数组（0=可通行，1=障碍物）- 2D maze data (0=passable, 1=obstacle)
        self.seed = seed  # 保存随机种子 - Save random seed
        self.generate_new_maze()  # 生成初始迷宫 - Generate initial maze

    def generate_new_maze(self):
        """
        生成新的随机迷宫
        Generate new random maze

        创建网格点数组、权重数组和迷宫数据数组，并随机初始化每个网格的颜色和属性
        Create grid points array, weights array and maze data array,
        randomly initialize each grid's color and properties
        """
        # 初始化二维数组 - Initialize 2D arrays
        self.grids = [[None] * self.grid_num for _ in range(self.grid_num)]
        self.weight_list = [[0] * self.grid_num for _ in range(self.grid_num)]
        self.maze = [[0] * self.grid_num for _ in range(self.grid_num)]

        # 创建局部随机数生成器 - Create local random number generator
        import random
        # self.seed = 4303
        local_random = random.Random(self.seed)

        # 遍历所有网格位置 - Iterate through all grid positions
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                # 随机选择颜色名称 - Randomly select color name
                color_name = ORIGINAL_COLORS[local_random.randint(0, len(ORIGINAL_COLORS) - 1)]
                # 获取对应的RGB颜色值 - Get corresponding RGB color value
                color = COLORS_RGB[color_name]
                # 创建网格点对象 - Create grid point object
                self.grids[i][j] = Spot(i, j, color)
                # 设置权重值 - Set weight value
                self.weight_list[i][j] = COLORS_WEIGHT[color_name]
                # 设置迷宫数据（1表示障碍物，0表示可通行）- Set maze data (1=obstacle, 0=passable)
                self.maze[i][j] = 1 if color_name == 'black' else 0

    def draw_maze(self, screen, start_pos: tuple, end_pos: tuple, path: list = None):
        """
        绘制整个迷宫到屏幕
        Draw entire maze to screen

        Args:
            screen: Pygame显示表面 - Pygame display surface
            start_pos (tuple): 起点坐标(x,y) - Start position coordinates (x,y)
            end_pos (tuple): 终点坐标(x,y) - End position coordinates (x,y)
            path (list, optional): 路径点列表，默认为None - Path points list, default None
        """
        # 绘制所有网格点 - Draw all grid points
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                self.grids[i][j].show(screen, self.grid_length)

        # 绘制起点（蓝色填充）- Draw start point (blue filled)
        self.grids[start_pos[0]][start_pos[1]].show(screen, self.grid_length, COLORS_RGB['blue'], 0)
        # 绘制终点（红色填充）- Draw end point (red filled)
        self.grids[end_pos[0]][end_pos[1]].show(screen, self.grid_length, COLORS_RGB['red'], 0)

        # 如果存在路径，绘制路径点（黄色）- If path exists, draw path points (yellow)
        if path:
            # 排除起点和终点，只绘制中间路径点 - Exclude start and end points, only draw intermediate path points
            for x, y in path[1:-1]:
                self.grids[x][y].show(screen, self.grid_length, COLORS_RGB['yellow'], self.grids[x][y].style)

        # 更新显示 - Update display
        pygame.display.update()

    def highlight_cell(self, screen, x: int, y: int, color: str, immediate_update: bool = True):
        """
        高亮特定单元格
        Highlight specific cell

        用于算法可视化过程中标记当前处理的单元格
        Used to mark currently processed cell during algorithm visualization

        Args:
            screen: Pygame显示表面 - Pygame display surface
            x (int): x坐标 - x coordinate
            y (int): y坐标 - y coordinate
            color (str): 颜色名称 - Color name
            immediate_update (bool): 是否立即刷新显示 - Whether to update display immediately
        """
        # 使用指定颜色高亮显示单元格 - Highlight cell with specified color
        self.grids[x][y].show(screen, self.grid_length, COLORS_RGB[color], 0)

        rect = pygame.Rect(
            x * self.grid_length,
            y * self.grid_length,
            self.grid_length,
            self.grid_length
        )
        pygame.display.update(rect)  # 只更新指定区域

        # 立即更新显示 - Update display immediately
        if immediate_update:
            pygame.display.update()

    def get_random_valid_positions(self) -> tuple:
        """
        获取随机的有效起点和终点位置
        Get random valid start and end positions

        随机生成起点和终点，确保它们不是同一个位置且都不是障碍物
        Randomly generate start and end positions, ensure they are not the same
        and neither is an obstacle

        Returns:
            tuple: 包含起点和终点的元组 ((start_x, start_y), (end_x, end_y))
                   Tuple containing start and end positions ((start_x, start_y), (end_x, end_y))
        """
        for _ in range(10000):
            # 随机生成起点坐标 - Randomly generate start coordinates
            start = (randint(0, self.grid_num - 1), randint(0, self.grid_num - 1))
            # 随机生成终点坐标 - Randomly generate end coordinates
            end = (randint(0, self.grid_num - 1), randint(0, self.grid_num - 1))

            # 检查起点和终点是否有效 - Check if start and end are valid
            if (start != end and
                    self.grids[start[0]][start[1]].color != COLORS_RGB['black'] and
                    self.grids[end[0]][end[1]].color != COLORS_RGB['black']):
                return start, end  # 返回有效的起点和终点 - Return valid start and end positions
        return (0, 0), (0, 0)
