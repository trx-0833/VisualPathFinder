import pygame
from pygame import QUIT
from constants import GRID_NUM, LENGTH


class PygameDisplay:
    """
    Pygame显示管理类
    Pygame Display Management Class

    负责管理Pygame的初始化、事件处理和显示更新
    Responsible for managing Pygame initialization, event handling and display updates
    """

    def __init__(self, parent_frame):
        """
        初始化Pygame显示管理器
        Initialize Pygame display manager

        Args:
            parent_frame: Tkinter父框架，用于嵌入Pygame窗口
                         Tkinter parent frame for embedding Pygame window
        """
        self.parent = parent_frame  # Tkinter父框架 - Tkinter parent frame
        self.screen = None  # Pygame显示表面 - Pygame display surface
        self.setup_pygame()  # 初始化Pygame环境 - Initialize Pygame environment

    def setup_pygame(self):
        """
        设置和初始化Pygame环境
        Setup and initialize Pygame environment

        配置SDL环境变量，将Pygame窗口嵌入到Tkinter框架中
        Configure SDL environment variables to embed Pygame window into Tkinter frame
        """
        import os
        # 设置SDL窗口ID到Tkinter框架的窗口ID
        # Set SDL window ID to Tkinter frame's window ID
        os.environ['SDL_WINDOWID'] = str(self.parent.winfo_id())
        # 设置SDL视频驱动程序为Windows DIB（设备无关位图）
        # Set SDL video driver to Windows DIB (Device Independent Bitmap)
        os.environ['SDL_VIDEODRIVER'] = 'windib'

        # 初始化Pygame显示模块 - Initialize Pygame display module
        pygame.display.init()
        # 创建指定大小的显示表面 - Create display surface with specified size
        self.screen = pygame.display.set_mode((LENGTH, LENGTH))

    def handle_events(self, on_click_callback):
        """
        处理Pygame事件循环
        Handle Pygame event loop

        监听并处理Pygame事件，包括退出事件和鼠标点击事件
        Listen and process Pygame events including quit events and mouse click events

        Args:
            on_click_callback (callable): 鼠标点击回调函数，接收网格坐标参数
                                          Mouse click callback function, receives grid coordinates

        Returns:
            bool: 是否继续运行（False表示应退出程序）- Whether to continue running (False means should exit program)
        """
        # 遍历所有Pygame事件 - Iterate through all Pygame events
        for event in pygame.event.get():
            # 处理退出事件（点击窗口关闭按钮）- Handle quit event (click window close button)
            if event.type == QUIT:
                return False  # 返回False表示应退出程序 - Return False means should exit program

            # 处理鼠标左键点击事件 - Handle mouse left button click event
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # 获取鼠标点击位置的像素坐标 - Get pixel coordinates of mouse click position
                x, y = event.pos
                # 将像素坐标转换为网格坐标 - Convert pixel coordinates to grid coordinates
                grid_x, grid_y = int(x / (LENGTH / GRID_NUM)), int(y / (LENGTH / GRID_NUM))
                # 调用点击回调函数，传递网格坐标 - Call click callback function with grid coordinates
                on_click_callback(grid_x, grid_y)

        # 返回True表示继续运行 - Return True means continue running
        return True

    def clear_screen(self):
        """
        清空屏幕（填充黑色）
        Clear screen (fill with black)

        用于在重绘前清除之前的内容，避免残影
        Used to clear previous content before redrawing to avoid ghosting
        """
        # 使用黑色填充整个屏幕 - Fill entire screen with black color
        self.screen.fill((0, 0, 0))

    def update_display(self):
        """
        更新显示内容
        Update display content

        将内存中的绘制内容刷新到屏幕上显示
        Refresh drawn content from memory to screen for display
        """
        # 更新Pygame显示 - Update Pygame display
        pygame.display.update()
