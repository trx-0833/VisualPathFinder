"""
包含所有颜色和网格相关的配置参数
Contains all color and grid related configuration parameters
"""

# ============================================================================
# 颜色系统配置
# Color System Configuration
# ============================================================================

# 基础颜色定义 - 用于地图生成和可视化
# Basic color definitions - for map generation and visualization
COLORS_RGB = {
    # 障碍物颜色 - Obstacle colors
    'black': (0, 0, 0),  # 不可通行的障碍物 - Impassable obstacles

    # 可通行区域颜色 - Passable area colors  
    'white': (255, 255, 255),  # 基础可通行区域 - Basic passable area
    'grey': (204, 204, 204),  # 中等通行成本区域 - Medium traversal cost area
    'orange': (254, 216, 177),  # 高通行成本区域 - High traversal cost area

    # 特殊标记颜色 - Special marker colors
    'blue': (0, 0, 255),  # 起点标记 - Start point marker
    'red': (255, 20, 60),  # 终点标记 - End point marker
    'yellow': (255, 255, 25),  # 路径标记 - Path marker
    'green': (102, 204, 153),  # 搜索过程标记 - Search process marker

    # 备用颜色 - Reserved colors
    'purple': (218, 112, 214)  # 备用颜色，可用于扩展功能 - Reserved for extended features
}

# 创建RGB值到颜色名称的反向映射
# Create reverse mapping from RGB values to color names
RGB_TO_COLORS = {v: k for k, v in COLORS_RGB.items()}

# 颜色权重配置 - 影响路径搜索算法中的移动成本
# Color weight configuration - affects movement cost in pathfinding algorithms
COLORS_WEIGHT = {
    'black': float('inf'),  # 障碍物，无法通行 - Obstacle, impassable
    'white': 1.0,  # 基础成本 - Base cost
    'grey': 2.0,  # 中等成本 - Medium cost
    'orange': 3.0,  # 较高成本 - Higher cost
    'red': 1.0,  # 终点，特殊处理 - End point, special handling
    'blue': 1.0,  # 起点，特殊处理 - Start point, special handling
    'green': 1.0,  # 搜索过程，临时状态 - Search process, temporary state
    'yellow': 1.0,  # 最终路径，特殊状态 - Final path, special state
    'purple': 1.0  # 备用权重 - Reserved weight
}

# ============================================================================
# 网格系统配置  
# Grid System Configuration
# ============================================================================

# 网格数量配置
# Grid quantity configuration
GRID_NUM = 100  # 每行/每列的网格数量 - Number of grids per row/column

# 界面尺寸配置  
# Interface dimension configuration
LENGTH = 600  # 主界面像素尺寸 - Main interface pixel dimension

# 计算得到的网格尺寸
# Calculated grid dimensions
GRID_LENGTH = LENGTH / GRID_NUM  # 单个网格的像素大小 - Pixel size of a single grid

# ============================================================================
# 算法相关配置
# Algorithm Related Configuration  
# ============================================================================

# 初始地图生成颜色分布
# Initial map generation color distribution
ORIGINAL_COLORS = [
    'black',  # 障碍物比例 - Obstacle ratio
    'white',  # 基础可通行区域比例 - Basic passable area ratio
    'grey',  # 中等成本区域比例 - Medium-cost area ratio
    'orange'  # 高成本区域比例 - High-cost area ratio
]
