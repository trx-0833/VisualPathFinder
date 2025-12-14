"""
以下问题：
1、Dijkstra算法如果没有通路会返回inf和长度为1的信息，需要根据此另外给出没有通路的判断
"""

import tkinter as tk
import random
import cProfile
import pstats
from pstats import SortKey
from maze_generator import MazeGenerator
from path_algorithms import PathFinder
from image_processor import ImageProcessor
from tkinter_gui import ControlPanel
from pygame_display import PygameDisplay
from constants import GRID_NUM, LENGTH, COLORS_RGB


class PathFinderApp:
    """
    主应用程序类
    Main Application Class

    协调所有组件，提供完整的路径搜索应用程序
    Coordinates all components to provide a complete path finding application
    """

    def __init__(self):
        """
        初始化主应用程序
        Initialize main application
        """
        # 创建主窗口 - Create main window
        self.root = tk.Tk()
        self.root.title("VisualPathFinder - 可视化路径搜索工具")

        # 初始化应用程序组件和变量
        # Initialize application components and variables
        self.setup_components()
        self.setup_variables()

        #
        self._is_on_search = False

        # 启动应用程序主循环
        # Start application main loop
        self.start_main_loop()

    def setup_components(self):
        """
        设置和初始化所有应用程序组件
        Setup and initialize all application components

        创建并配置GUI组件、显示模块和核心功能模块
        Create and configure GUI components, display modules and core functionality modules
        """
        # Pygame显示区域 - Pygame display area
        self.pygame_frame = tk.Frame(self.root, width=LENGTH, height=LENGTH)
        self.pygame_frame.pack(side=tk.LEFT)

        # Pygame显示模块 - Pygame display module
        self.pygame_display = PygameDisplay(self.pygame_frame)

        # 核心功能模块 - Core functionality modules
        self.maze_generator = MazeGenerator(GRID_NUM, LENGTH)  # 迷宫生成器 - Maze generator
        self.path_finder = PathFinder(GRID_NUM)  # 路径查找器 - Path finder
        self.image_processor = ImageProcessor()  # 图像处理器 - Image processor

        # 控制面板 - Control panel
        self.control_panel = ControlPanel(
            self.root,
            self.maze_generator.seed,
            on_refresh=self.on_refresh,
            on_search=self.on_search,
            on_reset=self.on_reset,
            on_define=self.on_define,
            on_help=self.on_help,
            on_maze=self.on_maze,
            on_seed=self.on_seed
        )

    def setup_variables(self):
        """
        初始化应用程序状态变量
        Initialize application state variables
        """
        # 获取随机的有效起点和终点位置
        # Get random valid start and end positions
        self.start_pos, self.end_pos = self.maze_generator.get_random_valid_positions()
        # 在控制面板中显示起点和终点位置
        # Display start and end positions in control panel
        self.control_panel.set_positions_to_entries(self.start_pos, self.end_pos)

        # 初始化路径相关变量
        # Initialize path-related variables
        self.current_path = None  # 当前路径 - Current path
        self.path_cost = 0  # 路径代价 - Path cost

        # 初始化标志变量
        # Initialize flag variables
        self.first_time = True  # 首次运行标志 - First run flag

    def start_main_loop(self):
        """
        启动应用程序主循环
        Start application main loop

        初始化Pygame循环并启动Tkinter主事件循环
        Initialize Pygame loop and start Tkinter main event loop
        """
        self.pygame_loop()  # 启动Pygame循环 - Start Pygame loop
        self.root.mainloop()  # 启动Tkinter主循环 - Start Tkinter main loop

    def pygame_loop(self):
        """
        Pygame主循环函数
        Pygame main loop function

        处理Pygame事件、更新显示，并安排下一次循环
        Process Pygame events, update display, and schedule next loop
        """
        # 处理Pygame事件，如果返回False则退出程序
        # Process Pygame events, exit program if returns False
        if not self.pygame_display.handle_events(self.on_cell_click):
            self.root.quit()
            return

        # 清空屏幕 - Clear screen
        self.pygame_display.clear_screen()

        # 如果是第一次运行，绘制迷宫
        # If first time running, draw maze
        # if self.first_time:
        #     self.draw_maze()
        #     self.first_time = False

        # 绘制迷宫 - Draw maze
        self.draw_maze()

        # 安排300毫秒后再次调用此函数，实现循环
        # Schedule next call to this function after 300ms to implement loop
        self.root.after(300, self.pygame_loop)

    def draw_maze(self):
        """
        绘制迷宫到屏幕
        Draw maze to screen

        调用迷宫生成器的绘制方法，显示当前迷宫状态
        Call maze generator's draw method to display current maze state
        """
        self.maze_generator.draw_maze(
            self.pygame_display.screen,  # Pygame显示表面 - Pygame display surface
            self.start_pos,  # 起点位置 - Start position
            self.end_pos,  # 终点位置 - End position
            self.current_path  # 当前路径（可能为None）- Current path (perhaps None)
        )

    def on_cell_click(self, x: int, y: int):
        """
        处理网格单元格点击事件
        Handle grid cell click event

        当用户在Pygame区域点击时，根据焦点状态更新起点或终点位置
        When user clicks in Pygame area, update start or end position based on focus state

        Args:
            x (int): 点击的网格x坐标 - Clicked grid x coordinate
            y (int): 点击的网格y坐标 - Clicked grid y coordinate
        """
        # 判断path的线条是否被去掉，如果path存在则清除它
        # Check if path lines should be removed, if path exists then clear it
        # if self.current_path is not None:
        #     self.on_reset()  # 重新绘制迷宫（不显示路径）- Redraw maze (without path)

        # 判断是否处于搜索状态，如果处于，则所有事件都被忽略
        # Determine whether it is in a search state; if it is, all events are ignored.
        if self._is_on_search:
            return None
        # 检查点击位置是否有效（不是障碍物且不是当前起点或终点）
        # Check if clicked position is valid (not obstacle and not current start/end)
        if (self.maze_generator.grids[x][y].color != COLORS_RGB['black'] and
                (x, y) != self.start_pos and (x, y) != self.end_pos):

            # 如果起点输入框有焦点，更新起点位置
            # If start entry has focus, update start position
            if self.control_panel.entry_1:
                # 防止鼠标在Pygame界面点击时清除路径，只在聚焦到起点输入框时清除路径
                # Prevent clearing path when clicking on Pygame interface, only clear when start entry is focused
                self.on_reset()  # 重新绘制迷宫（不显示路径）- Redraw maze (without path)
                self.start_pos = (x, y)
                # 更新起点输入框内容 - Update start entry content
                self.control_panel.entry_start.delete(0, 'end')
                self.control_panel.entry_start.insert(0, str((x, y)))

            # 如果终点输入框有焦点，更新终点位置
            # If end entry has focus, update end position
            if self.control_panel.entry_2:
                # 防止鼠标在Pygame界面点击时清除路径，只在聚焦到终点输入框时清除路径
                # Prevent clearing path when clicking on Pygame interface, only clear when end entry is focused
                self.on_reset()  # 重新绘制迷宫（不显示路径）- Redraw maze (without path)
                self.end_pos = (x, y)
                # 更新终点输入框内容 - Update end entry content
                self.control_panel.entry_end.delete(0, 'end')
                self.control_panel.entry_end.insert(0, str((x, y)))

    def on_refresh(self, _is_seed_set=True):
        """
        刷新地图回调函数
        Refresh map callback function

        生成新的随机迷宫，重置起点终点，并更新显示
        Generate new random maze, reset start/end positions, and update display
        """
        # 生成新种子 - Generate new seed
        if _is_seed_set:
            self.maze_generator.seed = random.randint(1, 100000)
        # 生成新迷宫 - Generate new maze
        self.maze_generator.generate_new_maze()
        # 更新随机数种子 - Update random seed
        self.control_panel.set_seed(self.maze_generator.seed)
        # 获取新的随机起点和终点 - Get new random start and end positions
        self.start_pos, self.end_pos = self.maze_generator.get_random_valid_positions()
        # 更新控制面板中的位置显示 - Update position display in control panel
        self.control_panel.set_positions_to_entries(self.start_pos, self.end_pos)
        # 重置路径显示 - Reset path display
        self.on_reset()
        # 重新绘制迷宫 - Redraw maze
        self.draw_maze()

    def on_search(self):
        """
        搜索路径回调函数
        Search path callback function

        执行路径搜索算法，查找从起点到终点的最短路径
        Execute path search algorithm to find the shortest path from start to end
        """
        # 禁用所有控件防止用户干扰 - Disable all controls to prevent user interference
        self.control_panel.disable_all()
        # 确认当前位置设置 - Confirm current position settings
        self.on_define()
        # 重置当前路径显示 - Reset current path display
        self.on_reset()
        # 设置当前状态为搜索状态 - Set the current status to search mode
        self._is_on_search = True
        # 从thinker下拉选框中读取method方法 - Read the method from the thinker dropdown menu
        method = self.control_panel.get_selected_algorithm()

        # 创建函数映射词典 - Create a function mapping dictionary
        search_methods = {
            'BFS': self.path_finder.bfs,
            'DFS': self.path_finder.dfs,
            'A*': self.path_finder.a_star,
            'Dijkstra': self.path_finder.dijkstra,
            'D-BFS': self.path_finder.bidirectional_bfs,
            'D-DFS': self.path_finder.bidirectional_dfs,
            'B*': self.path_finder.b_star
        }

        # 创建分析器实例并开始分析
        profiler = cProfile.Profile()
        profiler.enable()  # 开启性能分析

        # 使用指定算法查找最短路径 - Use specially algorithm to find the shortest path
        self.current_path, self.path_cost = search_methods[method](
            self.pygame_display.screen, self.start_pos, self.end_pos,
            self.maze_generator.grids, self.maze_generator.weight_list,
            self.control_panel.check_var.get(),  # 是否显示搜索细节 - Whether to show search details
            callback=self.maze_generator.highlight_cell  # 进度回调函数 - Progress callback function
        )

        # 停止分析并处理结果
        profiler.disable()  # 关闭性能分析
        # 直接打印统计结果（按累计耗时排序）
        stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats()

        if self.path_cost != float('inf') and len(self.current_path) != []:
            # 显示搜索结果对话框 - Show search result dialog
            self.control_panel.show_info_dialog(
                "搜索结果",
                f"{method}算法的搜索结果\n最小代价: {self.path_cost}, 路径长度: {len(self.current_path) - 1}"
            )
            # 更新迷宫显示（包含路径）- Update maze display (with path)
            self.draw_maze()
        else:
            # 路径不存在，显示错误信息 - Path doesn't exist, show error message
            self.control_panel.show_info_dialog(
                f"{method}算法的搜索结果",
                "起点和终点之间不存在通路",
                is_success=False
            )

        # 设置当前状态为搜索状态 - Set the current status to search mode
        self._is_on_search = False

        # 重新启用所有控件 - Re-enable all controls
        self.control_panel.enable_all()

    def on_reset(self):
        """
        重置路径显示回调函数
        Reset path display callback function

        清除当前路径并更新显示
        Clear current path and update display
        """
        self.current_path = None  # 清除路径 - Clear path
        self.draw_maze()  # 重新绘制迷宫（不显示路径）- Redraw maze (without path)

    def on_define(self):
        """
        确认位置回调函数
        Confirm positions callback function

        从输入框读取并验证起点和终点位置
        Read and validate start and end positions from input fields
        """
        # 移除焦点，确保输入框更新 - Remove focus to ensure entry updates
        self.root.focus()
        # 从输入框获取位置 - Get positions from entries
        pos1, pos2 = self.control_panel.get_positions_from_entries()

        # 如果位置有效，更新起点和终点 - If positions valid, update start and end
        if pos1 != (-1, -1) and pos2 != (-1, -1):
            # 更新起点（如果有效且不是终点）- Update start (if valid and not end)
            if (self.path_finder.is_valid_move(pos1[0], pos1[1], self.maze_generator.grids) and
                    pos1 != self.end_pos):
                self.start_pos = pos1

            # 更新终点（如果有效且不是起点）- Update end (if valid and not start)
            if (self.path_finder.is_valid_move(pos2[0], pos2[1], self.maze_generator.grids) and
                    pos2 != self.start_pos):
                self.end_pos = pos2

        # 更新输入框显示（处理可能的调整）- Update entry displays (handling possible adjustments)
        self.control_panel.set_positions_to_entries(self.start_pos, self.end_pos)

    def on_seed(self):
        """
        处理种子设置事件
        Handle seed setting event

        从输入框获取种子值并更新迷宫生成器
        Get seed value from entry field and update maze generator
        """
        # 移除焦点，确保输入框更新 - Remove focus to ensure entry updates
        self.root.focus()

        # 从输入框获取位置 - Get positions from entry
        seed = self.control_panel.get_seed_from_entry()

        if seed is not None:
            # 更新迷宫生成器的种子 - Update maze generator's seed
            self.maze_generator.seed = seed
        # 刷新地图但不重置种子（因为种子已经设置）- Refresh map without resetting seed (since seed is already set)
        self.on_refresh(False)

    def on_help(self):
        """
        显示帮助回调函数
        Show help callback function

        打开使用帮助对话框
        Open usage help dialog
        """
        self.control_panel.show_help_dialog()

    def on_maze(self):
        """
        图片迷宫回调函数
        Image maze callback function

        从图片文件导入迷宫并更新显示
        Import maze from image file and update display
        """
        # 重置路径 - Reset path
        self.on_reset()
        # 选择图片文件 - Select image file
        file_path = self.control_panel.select_image_file()
        if file_path:
            # 处理图片迷宫 - Process image maze
            success = self.image_processor.process_maze_image(
                file_path,
                self.maze_generator.grids,
                self.maze_generator.maze,
                self.maze_generator.weight_list,
                GRID_NUM,
                LENGTH
            )

            # 如果处理成功，更新显示 - If processing successful, update display
            if success:
                # 获取新的有效起点和终点 - Get new valid start and end positions
                self.start_pos, self.end_pos = self.maze_generator.get_random_valid_positions()
                # 如果获取不到起点、终点坐标，同样报错
                # If the start or end coordinates cannot be obtained, an error will also be reported.
                if self.start_pos == self.end_pos:
                    # 处理失败，显示错误信息 - Processing failed, show error message
                    self.control_panel.show_info_dialog("错误", "请尝试调大源码中GRID_NUM值,\n当前功能推荐GRID_NUM大小=LENGTH/2大小", False)
                    # 刷新地图 - refresh maze
                    self.on_refresh()
                # 更新输入框显示 - Update entry displays
                self.control_panel.set_positions_to_entries(self.start_pos, self.end_pos)
                # 重置路径 - Reset path
                self.on_reset()
                # 重新绘制迷宫 - Redraw maze
                self.draw_maze()
            else:
                # 处理失败，显示错误信息 - Processing failed, show error message
                self.control_panel.show_info_dialog("错误", "无法处理选择的图片文件", False)


if __name__ == "__main__":
    """
    应用程序入口点
    Application entry point

    当直接运行此脚本时创建并启动应用程序
    Create and start application when this script is run directly
    """
    app = PathFinderApp()  # 创建应用程序实例 - Create application instance
