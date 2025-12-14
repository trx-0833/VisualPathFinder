import re
from typing import Tuple


def parse_coordinates(text: str) -> Tuple[int, int]:
    """
    从文本中解析坐标信息
    Parse coordinate information from text

    使用正则表达式从任意文本中提取前两个数字作为坐标
    Use regular expressions to extract the first two numbers as coordinates from any text

    Args:
        text (str): 包含坐标信息的文本，如 "(10, 20)" 或 "坐标: 10, 20"
                   Text containing coordinate information, e.g. "(10, 20)" or "coordinate: 10, 20"

    Returns:
        Tuple[int, int]: 解析出的坐标元组 (x, y)
                        如果无法解析出两个数字，则返回 (-1, -1)
                        Parsed coordinate tuple (x, y)
                        Returns (-1, -1) if cannot parse two numbers

    Examples:
        >>> parse_coordinates("起点: (5, 10)")
        (5, 10)
        >>> parse_coordinates("位置 15,20 结束")
        (15, 20)
        >>> parse_coordinates("无效文本")
        (-1, -1)
    """
    # 使用正则表达式提取文本中的所有数字序列
    # Use regular expression to extract all number sequences from text
    matches = re.findall(r'\d+', text)

    # 如果找到至少两个数字，返回前两个作为坐标
    # If at least two numbers found, return first two as coordinates
    if len(matches) >= 2:
        return int(matches[0]), int(matches[1])

    # 无法解析出有效坐标，返回错误标志
    # Cannot parse valid coordinates, return error flag
    return -1, -1


def is_valid_coordinate(x: int, y: int, grid_num: int) -> bool:
    """
    检查坐标是否在有效范围内
    Check if coordinates are within valid range

    验证坐标是否在网格系统的边界内 (0 <= x,y < grid_num)
    Verify if coordinates are within grid system boundaries (0 <= x,y < grid_num)

    Args:
        x (int): x坐标 - x coordinate
        y (int): y坐标 - y coordinate
        grid_num (int): 网格系统的大小（每行/每列的网格数量）
                       Grid system size (number of grids per row/column)

    Returns:
        bool: 坐标是否有效 - Whether coordinates are valid

    Examples:
        >>> is_valid_coordinate(5, 10, 25)
        True
        >>> is_valid_coordinate(-1, 10, 25)
        False
        >>> is_valid_coordinate(30, 10, 25)
        False
    """
    # 检查x和y坐标是否都在有效范围内 [0, grid_num-1]
    # Check if both x and y coordinates are within valid range [0, grid_num-1]
    return 0 <= x < grid_num and 0 <= y < grid_num
