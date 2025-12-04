# VisualPathFinder - 可视化路径搜索算法演示工具

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Pygame](https://img.shields.io/badge/Pygame-2.5.0-green.svg)](https://pygame.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0.74-orange.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于 Pygame 和 Tkinter 的交互式路径搜索算法可视化工具，支持多种经典搜索算法，并包含迷宫生成、图像导入和实时可视化功能。

<p align="center">
  <img src="https://img.shields.io/badge/算法-BFS%20|%20DFS%20|%20Dijkstra%20|%20A*-blueviolet" alt="支持的算法">
  <img src="https://img.shields.io/badge/交互式-图形界面-ff69b4" alt="交互式界面">
  <img src="https://img.shields.io/badge/实时-可视化-9cf" alt="实时可视化">
</p>

## ✨ 功能特性

### 🧠 算法可视化
- **广度优先搜索 (BFS)** - 找到最短步数路径
- **深度优先搜索 (DFS)** - 快速找到一条路径
- **Dijkstra算法** - 最小代价路径（考虑权重）
- **A*算法** - 启发式搜索，高效找到最优路径
- **实时过程显示** - 绿色节点表示已访问，紫色节点表示待访问

### 🗺️ 迷宫系统
- **随机迷宫生成** - 支持设置随机种子复现迷宫
- **图片迷宫导入** - 可将图片转换为迷宫（黑白二值化处理）
- **多种地形类型** - 障碍物（黑色）、普通路径（白色）、高成本区域（灰色/橙色）
- **权重系统** - 不同颜色代表不同的通行成本

### 🎮 交互功能
- **点击设置起点/终点** - 直观的鼠标操作
- **实时路径显示** - 算法执行过程中动态显示搜索过程
- **性能分析** - 集成cProfile，显示算法运行统计
- **可调参数** - 网格大小、搜索细节显示等

## 📦 安装指南

### 环境要求
- Python 3.8+
- Pygame 2.5.0
- OpenCV 4.8.0.74

### 快速安装

1. **克隆仓库**
   ```bash
   git clone https://github.com/trx-0833/VisualPathFinder.git
   cd VisualPathFinder
   ```

2. **安装依赖**
   ```bash
   pip install pygame==2.5.0 opencv-python==4.8.0.74
   ```

   或者使用requirements.txt：
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 使用指南

### 基本使用
1. **启动程序**
   ```bash
   python main.py
   ```

2. **界面介绍**
   - 左侧：迷宫显示区域（Pygame）
   - 右侧：控制面板（Tkinter）

3. **设置起点和终点**
   - 方法1：点击起点/终点输入框，然后在迷宫图中点击设置位置
   - 方法2：直接在输入框中输入坐标，格式如 `(5, 10)`

4. **选择搜索算法**
   - 使用下拉菜单选择BFS、DFS、Dijkstra或A*算法
   - 勾选"显示搜索细节"可观察算法展开过程

5. **开始搜索**
   - 点击"搜索路线"按钮开始路径搜索
   - 使用"重置"按钮清除当前路径
   - 使用"刷新地图"按钮生成新迷宫

### 高级功能

#### 图片迷宫导入
1. 点击"图片迷宫探索(实验性)"按钮
2. 选择一张图片文件（推荐JPG格式，路径不要包含中文）
3. 程序将自动将图片转换为黑白迷宫

#### 随机种子设置
1. 在"随机种子"输入框中输入种子值
2. 点击"设置种子"按钮
3. 点击"刷新地图"使用指定种子生成迷宫

## 📁 项目结构

```
VisualPathFinder/
├── main.py                 # 主应用程序入口
├── constants.py            # 常量配置（颜色、网格、算法参数）
├── pygame_display.py       # Pygame显示管理模块
├── tkinter_gui.py          # Tkinter控制面板
├── maze_generator.py       # 迷宫生成与管理
├── path_algorithms.py      # 路径搜索算法实现
├── image_processor.py      # 图像迷宫处理模块
├── helpers.py              # 工具函数（坐标解析等）
└── README.md               # 项目说明文档
```

## ⚙️ 配置说明

可在 `constants.py` 中调整以下参数：

```python
# 网格系统配置
GRID_NUM = 100      # 每行/每列的网格数量
LENGTH = 600        # 主界面像素尺寸

# 颜色与权重配置
COLORS_RGB = {
    'black': (0, 0, 0),        # 障碍物
    'white': (255, 255, 255),  # 基础可通行
    'grey': (204, 204, 204),   # 中等成本
    'orange': (254, 216, 177), # 高成本
    'blue': (0, 0, 255),       # 起点
    'red': (255, 20, 60),      # 终点
    'yellow': (255, 255, 25),  # 路径
    'green': (102, 204, 153),  # 搜索过程
    'purple': (218, 112, 214)  # 待访问
}
```

## 🖼️ 界面截图

```
+--------------------------------+--------------------------------+
|         迷宫显示区域           |        控制面板区域           |
|                                |                                |
|  [迷宫图形]                    |  随机种子: [4303] [设置种子]  |
|                                |  搜索算法: [Dijkstra▼]         |
|                                |  起点位置: [(5, 10)]          |
|                                |  终点位置: [(15, 20)]         |
|                                |   [重置] [确定]               |
|                                |   [ ] 显示搜索细节            |
|                                |   [刷新地图] [搜索路线]       |
|                                |   [使用帮助]                  |
|                                |   [图片迷宫探索(实验性)]      |
|                                |  请务必查看使用帮助！         |
+--------------------------------+--------------------------------+
```

## 🔧 故障排除

### 常见问题

1. **程序无法启动**
   ```
   检查依赖安装：python test_environment.py
   确保安装了正确版本的Pygame和OpenCV
   ```

2. **图片导入失败**
   ```
   确保图片路径不包含中文字符
   检查图片格式是否支持（推荐JPG）
   尝试调整GRID_NUM值（建议设置为LENGTH/2）
   ```

3. **搜索算法卡死**
   ```
   检查起点和终点是否可达
   尝试增加网格大小或简化迷宫
   使用"显示搜索细节"观察算法过程
   ```

4. **界面显示异常**
   ```
   Windows：确保设置了正确的SDL环境变量
   Linux：可能需要安装tkinter：sudo apt-get install python3-tk
   MacOS：可能需要安装SDL2库
   ```

### 调试模式
在代码中添加调试输出：
```python
# 在main.py中添加
import cProfile
profiler = cProfile.Profile()
profiler.enable()
# ... 算法执行 ...
profiler.disable()
profiler.print_stats(sort='cumtime')
```

## 📚 算法说明

### BFS（广度优先搜索）
- **特点**：逐层扩展，找到最短步数路径
- **适用场景**：所有边权重相等，寻找最短步数
- **时间复杂度**：O(V+E)

### DFS（深度优先搜索）
- **特点**：深入探索分支，快速找到一条路径
- **适用场景**：只需要任意路径，不要求最优
- **时间复杂度**：O(V+E)

### Dijkstra算法
- **特点**：考虑边权重，找到最小代价路径
- **适用场景**：有权图，寻找最小代价路径
- **时间复杂度**：O((V+E)logV)

### A*算法
- **特点**：启发式搜索，更高效找到最优路径
- **适用场景**：有权图，且存在有效启发函数
- **时间复杂度**：取决于启发函数质量

## 🤝 贡献指南

欢迎贡献代码或报告问题！

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢Pygame和OpenCV开源社区
- 感谢所有贡献者和用户
- 灵感来源于经典的路径搜索算法教学需求

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [提交问题](https://github.com/trx-0833/VisualPathFinder/issues)

---

<p align="center">
  <b>项目来源于我本科时期的期末作业，对其进行完善以后放到GitHub上，供诸位参考！</b>
</p>
