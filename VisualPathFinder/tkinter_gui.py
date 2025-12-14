"""
如下问题：
1、控件的睡眠功能没有设置好，容易卡死，（确定问题出现在disable_all函数上）
2、如有必要可以增加一个设置随机种子的控件
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re
from typing import Callable


class ControlPanel:
    """
    控制面板类
    Control Panel Class

    负责管理应用程序的图形用户界面，包括按钮、输入框、对话框等控件
    Responsible for managing the application's graphical user interface,
    including buttons, input fields, dialogs and other controls
    """

    def __init__(self, parent, seed, on_refresh: Callable, on_search: Callable, on_reset: Callable,
                 on_define: Callable, on_help: Callable, on_maze: Callable, on_seed:Callable):
        """
        初始化控制面板
        Initialize control panel

        Args:
            parent: Tkinter父窗口 - Tkinter parent window
            seed: 随机数种子 - Random seed
            on_refresh (Callable): 刷新地图回调函数 - Refresh map callback function
            on_search (Callable): 搜索路径回调函数 - Search path callback function
            on_reset (Callable): 重置路径回调函数 - Reset path callback function
            on_define (Callable): 确认位置回调函数 - Confirm positions callback function
            on_help (Callable): 显示帮助回调函数 - Show help callback function
            on_maze (Callable): 图片迷宫回调函数 - Image maze callback function
        """
        self.parent = parent  # 父窗口引用 - Parent window reference
        self.seed = seed  # 生成自maze_generator的种子
        # 回调函数字典 - Callback functions dictionary
        self.callbacks = {
            'refresh': on_refresh,
            'search': on_search,
            'reset': on_reset,
            'define': on_define,
            'help': on_help,
            'maze': on_maze,
            'seed': on_seed
        }

        # 输入框焦点状态标志 - Input field focus status flags
        self.entry_1 = False  # 起点输入框焦点状态 - Start point input field focus status
        self.entry_2 = False  # 终点输入框焦点状态 - End point input field focus status
        self.check_var = tk.IntVar()  # 复选框变量 - Checkbox variable

        # 算法选择变量 - Algorithm selection variable
        self.algorithm_var = tk.StringVar(value="Dijkstra")  # 默认选择Dijkstra算法 - Default to Dijkstra algorithm

        self.setup_ui()  # 初始化用户界面 - Initialize user interface

    def setup_ui(self):
        """
        设置用户界面布局
        Setup user interface layout

        创建和排列所有GUI控件，包括输入框、按钮、标签等
        Create and arrange all GUI controls including input fields, buttons, labels, etc.
        """
        # 主框架 - Main frame
        button_win = tk.Frame(self.parent)
        button_win.pack(side=tk.LEFT)

        # 添加随机种子设置 - Random seed setup
        self.setup_random_seed(button_win)

        # 算法选择下拉框设置 - Algorithm selection dropdown setup
        self.setup_algorithm_dropdown(button_win)

        # 起点和终点输入框设置 - Start and end point input fields setup
        self.setup_position_inputs(button_win)

        # 控制按钮设置 - Control buttons setup
        self.setup_control_buttons(button_win)

        # 复选框设置 - Checkbox setup
        self.setup_checkbox(button_win)

        # 功能按钮设置 - Function buttons setup
        self.setup_function_buttons(button_win)

        # 提示信息标签设置 - Information label setup
        self.setup_info_label(button_win)

    def setup_random_seed(self, parent):
        """
        设置随机种子控件
        Set Random Seed Control
        """
        seed_frame = tk.Frame(parent)
        seed_frame.pack(pady=5)

        # 随机种子标签 - Random Seed Label
        tk.Label(seed_frame, text="随机种子").pack(side=tk.LEFT)

        # 随机种子输入框 - Random seed input box
        self.seed_var = tk.StringVar(value=str(self.seed))
        self.seed_entry = tk.Entry(seed_frame, textvariable=self.seed_var, width=10)
        self.seed_entry.pack(side=tk.LEFT, padx=5)

        # 设置种子按钮 - Set seed button
        tk.Button(seed_frame, text="设置种子", command=self.callbacks['seed']).pack(side=tk.LEFT)

    def setup_algorithm_dropdown(self, parent):
        """
        设置算法选择下拉框
        Setup algorithm selection dropdown

        Args:
            parent: 父框架 - Parent frame
        """
        # 算法选择框架 - Algorithm selection frame
        algo_frame = tk.Frame(parent)
        algo_frame.pack()

        # 算法选择标签 - Algorithm selection label
        tk.Label(algo_frame, text="搜索算法").pack(side=tk.LEFT)

        # 算法选择下拉框 - Algorithm selection combobox
        algorithm_combo = ttk.Combobox(
            algo_frame,
            textvariable=self.algorithm_var,
            values=["BFS", "DFS", "Dijkstra", "A*"],
            state="readonly",  # 只读，用户只能选择预设值 - Readonly, user can only select predefined values
            width=17
        )
        algorithm_combo.pack(side=tk.LEFT, padx=0)

        # 设置默认值 - Set default value
        algorithm_combo.set("BFS")

    def setup_position_inputs(self, parent):
        """
        设置起点和终点位置输入框
        Setup start and end position input fields

        Args:
            parent: 父框架 - Parent frame
        """
        # 起点输入框架 - Start point input frame
        start_frame = tk.Frame(parent)
        start_frame.pack()
        # 起点标签 - Start point label
        tk.Label(start_frame, text="起点位置").pack(side=tk.LEFT)
        # 起点输入框 - Start point entry field
        self.entry_start = tk.Entry(start_frame)
        self.entry_start.pack(side=tk.LEFT)
        # 绑定焦点事件 - Bind focus events
        self.entry_start.bind("<FocusIn>", lambda e: self._handle_focus_in(True, False))
        self.entry_start.bind("<FocusOut>", lambda e: self._handle_focus_out(True, False))

        # 终点输入框架 - End point input frame
        end_frame = tk.Frame(parent)
        end_frame.pack()
        # 终点标签 - End point label
        tk.Label(end_frame, text="终点位置").pack(side=tk.LEFT)
        # 终点输入框 - End point entry field
        self.entry_end = tk.Entry(end_frame)
        self.entry_end.pack(side=tk.LEFT)
        # 绑定焦点事件 - Bind focus events
        self.entry_end.bind("<FocusIn>", lambda e: self._handle_focus_in(False, True))
        self.entry_end.bind("<FocusOut>", lambda e: self._handle_focus_out(False, True))

    def setup_control_buttons(self, parent):
        """
        设置控制按钮（重置、确定）
        Setup control buttons (Reset, Confirm)

        Args:
            parent: 父框架 - Parent frame
        """
        button_frame = tk.Frame(parent)
        button_frame.pack()

        # 重置按钮 - Reset button
        tk.Button(button_frame, text='重置', command=self.callbacks['reset']).pack(side=tk.LEFT)
        # 确定按钮 - Confirm button
        tk.Button(button_frame, text='确定', command=self.callbacks['define']).pack(side=tk.LEFT)

    def setup_checkbox(self, parent):
        """
        设置搜索细节显示复选框
        Setup search details display checkbox

        Args:
            parent: 父框架 - Parent frame
        """
        # 创建复选框 - Create checkbox
        tk.Checkbutton(parent, text="显示搜索细节", variable=self.check_var).pack()

    def setup_function_buttons(self, parent):
        """
        设置功能按钮（刷新地图、搜索路线、使用帮助）
        Setup function buttons (Refresh Map, Search Path, Help)

        Args:
            parent: 父框架 - Parent frame
        """
        button_frame = tk.Frame(parent)
        button_frame.pack()

        # 刷新地图按钮 - Refresh map button
        tk.Button(button_frame, text='刷新地图', command=self.callbacks['refresh']).pack(side=tk.LEFT)
        # 搜索路线按钮 - Search path button
        tk.Button(button_frame, text='搜索路线', command=self.callbacks['search']).pack(side=tk.LEFT)
        # 使用帮助按钮 - Help button
        tk.Button(button_frame, text='使用帮助', command=self.callbacks['help']).pack(side=tk.LEFT)

        # 图片迷宫按钮框架 - Image maze button frame
        maze_frame = tk.Frame(parent)
        maze_frame.pack()
        # 图片迷宫探索按钮 - Image maze exploration button
        tk.Button(maze_frame, text='图片迷宫探索(实验性)', command=self.callbacks['maze']).pack()

    def setup_info_label(self, parent):
        """
        设置提示信息标签
        Setup information label

        Args:
            parent: 父框架 - Parent frame
        """
        info_frame = tk.Frame(parent)
        info_frame.pack()
        # 提示信息标签 - Information label
        tk.Label(info_frame, text="请务必查看使用帮助！").pack()

    def _handle_focus_in(self, is_entry1: bool, is_entry2: bool):
        """
        处理输入框焦点进入事件（内部方法）
        Handle input field focus in event (internal method)

        Args:
            is_entry1 (bool): 是否是起点输入框 - Whether it's start point entry
            is_entry2 (bool): 是否是终点输入框 - Whether it's end point entry
        """
        self.entry_1 = is_entry1
        self.entry_2 = is_entry2

    def _handle_focus_out(self, is_entry1: bool, is_entry2: bool):
        """
        处理输入框焦点离开事件（内部方法）
        Handle input field focus out event (internal method)

        Args:
            is_entry1 (bool): 是否是起点输入框 - Whether it's start point entry
            is_entry2 (bool): 是否是终点输入框 - Whether it's end point entry
        """
        # 清除对应的焦点状态 - Clear corresponding focus status
        self.entry_1 = not is_entry1 if self.entry_1 else self.entry_1
        self.entry_2 = not is_entry2 if self.entry_2 else self.entry_2

    def get_selected_algorithm(self) -> str:
        """
        获取当前选择的算法
        Get currently selected algorithm

        Returns:
            str: 算法名称（'bfs', 'dfs', 'dijkstra', 'a_star'）
                  Algorithm name ('bfs', 'dfs', 'dijkstra', 'a_star')
        """
        return self.algorithm_var.get()

    def get_positions_from_entries(self) -> tuple:
        """
        从输入框解析起点和终点位置
        Parse start and end positions from input fields

        Returns:
            tuple: 包含起点和终点坐标的元组 ((x1,y1), (x2,y2))
                   Tuple containing start and end coordinates ((x1,y1), (x2,y2))
        """

        def parse_position(text: str) -> tuple:
            """
            从文本中解析坐标位置
            Parse coordinate position from text

            Args:
                text (str): 包含坐标的文本 - Text containing coordinates

            Returns:
                tuple: 坐标元组 (x,y)，解析失败返回(-1,-1)
                       Coordinate tuple (x,y), returns (-1,-1) if parsing fails
            """
            # 使用正则表达式提取所有数字 - Use regex to extract all numbers
            matches = re.findall(r'\d+', text)
            if len(matches) >= 2:
                return int(matches[0]), int(matches[1])
            return -1, -1

        # 解析起点和终点坐标 - Parse start and end coordinates
        pos1 = parse_position(self.entry_start.get())
        pos2 = parse_position(self.entry_end.get())
        return pos1, pos2

    def set_positions_to_entries(self, start: tuple, end: tuple):
        """
        设置起点和终点位置到输入框
        Set start and end positions to input fields

        Args:
            start (tuple): 起点坐标 (x,y) - Start point coordinates (x,y)
            end (tuple): 终点坐标 (x,y) - End point coordinates (x,y)
        """
        # 清空起点输入框并插入新坐标 - Clear start entry and insert new coordinates
        self.entry_start.delete(0, 'end')
        self.entry_start.insert(0, str(start))
        # 清空终点输入框并插入新坐标 - Clear end entry and insert new coordinates
        self.entry_end.delete(0, 'end')
        self.entry_end.insert(0, str(end))

    def get_seed_from_entry(self) -> int:
        """
        从种子输入框解析种子值
        Parse seed value from seed entry field

        Returns:
            int: 种子值，解析失败返回None
                  Seed value, returns None if parsing fails
        """

        def seed_value(text: str):
            """
            从文本中解析种子值
            Parse seed value from text

            Args:
                text (str): 包含种子的文本 - Text containing seed

            Returns:
                int: 种子值，解析失败返回None
                      Seed value, returns None if parsing fails
            """
            # 使用正则表达式提取所有数字 - Use regex to extract all numbers
            matches = re.findall(r'\d+', text)
            if len(matches) >= 1:
                return int(matches[0])
            return None
        # 解析种子值 - Parse seed value
        seed = seed_value(self.seed_var.get())
        return seed

    def set_seed(self, seed: int):
        """
        更新随机种子值并刷新显示
        Update random seed value and refresh display

        Args:
            seed (int): 新的随机种子值 - New random seed value
        """
        self.seed = seed  # 更新内部种子值 - Update internal seed value
        self.seed_var.set(str(seed))  # 更新输入框显示 - Update entry field display

    def show_info_dialog(self, title: str, message: str, is_success: bool = True):
        """
        显示信息对话框
        Show information dialog

        Args:
            title (str): 对话框标题 - Dialog title
            message (str): 对话框消息 - Dialog message
            is_success (bool, optional): 是否是成功信息，默认为True
                                         Whether it's success information, default True
        """
        if is_success:
            # 显示成功信息对话框 - Show success information dialog
            messagebox.showinfo(title, message)
        else:
            # 显示错误信息对话框 - Show error information dialog
            messagebox.showerror(title, message)

    def show_help_dialog(self):
        """
        显示使用帮助对话框
        Show help dialog

        包含程序使用说明和注意事项
        Contains program usage instructions and precautions
        """
        help_text = """
        本程序较为开放，可以修改代码开头大部分参数。

        使用本程序时有一些小技巧和注意事项：
        1. 输入起点和终点时可以点击对应的输入框并直接在图中选择点
        2. 重置按钮是为了清除地图上原本显示的路径
        3. 程序会先用深度优先搜索判断两点之间是否有通路
        4. 使用下拉框选择不同的搜索算法：
           - BFS: 广度优先搜索，找到最短路径（步数最少）
           - DFS: 深度优先搜索，快速找到一条路径但不保证最短
           - Dijkstra: 迪杰斯特拉算法，找到最小代价路径
           - A*: A星算法，启发式搜索，高效找到最小代价路径
        5. 图片迷宫探索支持以图片的形式输入迷宫
        6. 在导入迷宫图片时，推荐将GRID_NUM设置为上限值(即GRID_NUM=LENGTH/2)
        """

        # 创建帮助窗口 - Create help window
        help_window = tk.Toplevel(self.parent)
        help_window.title("使用帮助")
        help_window.geometry("600x400")

        # 创建文本部件显示帮助内容 - Create text widget to display help content
        text_widget = tk.Text(help_window, wrap=tk.WORD, width=80, height=25)
        text_widget.pack(padx=10, pady=10)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)  # 设置为只读 - Set to read-only

        # 关闭按钮 - Close button
        tk.Button(help_window, text="关闭", command=help_window.destroy).pack(pady=10)

    def select_image_file(self) -> str:
        """
        选择图片文件对话框
        Select image file dialog

        Returns:
            str: 选择的文件路径，如果取消选择则返回空字符串
                  Selected file path, returns empty string if selection is canceled
        """
        return filedialog.askopenfilename(
            title='选择文件(注意路径或文件名中不要有中文)',
            filetypes=[('JPG', '*.jpg'), ('All Files', '*')]
        )

    def disable_all(self):
        """
        禁用所有控件
        Disable all controls

        通常在长时间操作时调用，防止用户干扰
        Usually called during long operations to prevent user interference
        """

        # 隐藏鼠标光标（忙碌状态） - Hide mouse cursor (busy state)
        self.parent.config(cursor="none")  # 隐藏光标

        def disable_widget(widget):
            """
            递归禁用控件
            Recursively disable controls
            """
            try:
                # 尝试禁用当前控件 - Try disabling the current control
                if hasattr(widget, 'configure') and 'state' in widget.configure():
                    current_state = widget.cget('state')
                    if current_state != 'disabled':
                        widget.config(state='disabled')
            except Exception as e:
                # 忽略不支持state属性的控件 - Ignore controls that do not support the state property
                pass

            # 递归处理所有子控件 - Recursively process all child controls
            try:
                for child in widget.winfo_children():
                    disable_widget(child)
            except:
                pass

        # 禁用所有控件 - Disable all controls
        disable_widget(self.parent)

        # 强制更新GUI - Force update GUI
        self.parent.update_idletasks()
        self.parent.update()

    def enable_all(self):
        """
        启用所有控件
        Enable all controls

        在长时间操作完成后恢复控件状态
        Restore control states after long operations are completed
        """

        # 恢复鼠标光标 - Restore mouse cursor
        self.parent.config(cursor="")

        def enable_widget(widget):
            """递归启用控件 - Recursively enable widgets"""
            try:
                # 尝试启用当前控件 - Try to enable current widget
                if hasattr(widget, 'configure') and 'state' in widget.configure():
                    widget.config(state='normal')  # 设置为正常状态 - Set to normal state
            except Exception as e:
                # 忽略不支持state属性的控件 - Ignore widgets that don't support state property
                pass

            # 递归处理所有子控件 - Recursively process all child widgets
            try:
                for child in widget.winfo_children():  # 获取所有子控件 - Get all child widgets
                    enable_widget(child)  # 递归调用启用函数 - Recursively call enable function
            except:
                pass  # 忽略获取子控件时的异常 - Ignore exceptions when getting child widgets

        # 启用所有控件 - Enable all widgets
        enable_widget(self.parent)  # 从父窗口开始递归启用 - Start recursive enabling from parent window

        # 强制更新GUI - Force GUI update
        self.parent.update_idletasks()  # 处理所有挂起的空闲任务 - Process all pending idle tasks
        self.parent.update()  # 强制刷新显示 - Force display refresh
