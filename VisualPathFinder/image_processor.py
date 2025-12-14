"""
下列问题：
1、当GRID_NUM过小时，如GRID_NUM=30时，不能顺利导入第一张图片
(原因：GRID_NUM太小了的话，分辨率达不到要求，则显示的迷宫是全黑的，这个时候get_random_valid_positions()就会陷入查找起点和终点的死循环
解决办法：设置一个上限，比如一万次，找不到符合条件的点再返回错误信息)
"""

import cv2
import os
from constants import COLORS_RGB


class ImageProcessor:
    """
    图像处理类，用于从图片导入迷宫
    Image processing class for importing mazes from images

    提供将图片转换为迷宫网格的功能，支持常见图片格式
    Provides functionality to convert images to maze grids, supports common image formats
    """

    @staticmethod
    def process_maze_image(file_path: str, grids, maze, weight_list, grid_num: int, length: float) -> bool:
        """
        处理迷宫图片并更新网格数据
        Process maze image and update grid data

        将指定图片文件解析为迷宫网格，根据像素颜色设置网格属性和权重
        Parse specified image file into maze grid, set grid properties and weights based on pixel colors

        Args:
            file_path (str): 图片文件路径 - Image file path
            grids: 网格对象数组 - Grid objects array
            maze: 迷宫数据数组（0=可通行，1=障碍物）- Maze data array (0=passable, 1=obstacle)
            weight_list: 权重数组 - Weight array
            grid_num (int): 网格数量 - Number of grids
            length (float): 界面尺寸 - Interface dimension

        Returns:
            bool: 处理是否成功 - Whether processing was successful

        Raises:
            Exception: 图像处理过程中可能出现的各种异常 - Various exceptions that may occur during image processing
        """
        # 检查文件路径是否有效 - Check if file path is valid
        if not file_path:
            return False

        try:
            # ============================================================================
            # 文件路径处理
            # File Path Processing
            # ============================================================================

            # 获取当前文件的基准目录 - Get base directory of current file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # 将绝对路径转换为相对路径，提高可移植性 - Convert absolute path to relative path for better portability
            relative_path = os.path.relpath(file_path, base_dir)

            # ============================================================================
            # 图像读取和预处理
            # Image Reading and Preprocessing
            # ============================================================================

            # 读取图像文件 - Read image file
            image = cv2.imread(relative_path)
            # 检查图像是否成功加载 - Check if image was loaded successfully
            if image is None:
                return False

            # 转换为灰度图像，简化处理 - Convert to grayscale for simpler processing
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 二值化处理：将图像转换为黑白两色 - Binarization: convert image to black and white
            # 阈值128，大于128变为255（白色），小于等于128变为0（黑色）
            # Threshold 128, values >128 become 255 (white), values <=128 become 0 (black)
            _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
            # 调整图像尺寸以匹配界面大小 - Resize image to match interface size
            resized_image = cv2.resize(binary_image, (length, length))
            # 阈值128，大于128变为255（白色），小于等于128变为0（黑色）
            # Threshold 128, values >128 become 255 (white), values <=128 become 0 (black)
            _, binary_image_after = cv2.threshold(resized_image, 128, 255, cv2.THRESH_BINARY)
            # ============================================================================
            # 网格划分和处理
            # Grid Division and Processing
            # ============================================================================

            # 计算每个网格块的像素大小 - Calculate pixel size of each grid block
            block_size = int(length / grid_num)

            # 遍历图像中的所有网格块 - Iterate through all grid blocks in the image
            for y in range(0, binary_image_after.shape[0], block_size):
                for x in range(0, binary_image_after.shape[1], block_size):
                    # 计算网格坐标 - Calculate grid coordinates
                    j, i = int(x / block_size), int(y / block_size)
                    # 提取当前网格块对应的图像区域 - Extract image region corresponding to current grid block
                    block = binary_image_after[y:y + block_size, x:x + block_size]

                    # ============================================================================
                    # 颜色识别和网格更新
                    # Color Recognition and Grid Update
                    # ============================================================================

                    # 检查块中是否有黑色像素（值为0）- Check if block contains black pixels (value 0)
                    if (block == 0).any():  # 黑色像素 - Black pixels
                        # 设置为障碍物（黑色）- Set as obstacle (black)
                        grids[j][i].color = COLORS_RGB['black']
                        maze[j][i] = 1  # 标记为障碍物 - Mark as obstacle
                    else:  # 白色像素 - White pixels
                        # 设置为可通行区域（白色）- Set as passable area (white)
                        grids[j][i].color = COLORS_RGB['white']
                        maze[j][i] = 0  # 标记为可通行 - Mark as passable

                    # 根据网格类型设置权重 - Set weight based on grid type
                    # 障碍物权重为无穷大，可通行区域权重为1.0
                    # Obstacle weight is infinity, passable area weight is 1.0
                    weight_list[j][i] = float('inf') if maze[j][i] == 1 else 1.0

            # 处理成功，返回True - Processing successful, return True
            return True

        except Exception as e:
            # ============================================================================
            # 异常处理
            # Exception Handling
            # ============================================================================

            # 打印错误信息，便于调试 - Print error information for debugging
            print(f"图像处理错误: {e}")
            # 返回处理失败 - Return processing failure
            return False
