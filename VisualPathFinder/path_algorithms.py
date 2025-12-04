"""
检查时发现了一下问题：
1、显示搜索过程的刷新率，要在项目基本完成后重新考量
2、dijkstra算法的绿色（已搜索点）和紫色（正在搜索点）出现交替闪烁现象，应该是不被允许的(原因在于已经确定最短距离的点重复入列，导致会反复计算这些点)
3、GRID_NUM=30时，导入maze_2迷宫，dijkstra算法搜索会出现问题(错误来源于回溯路径时出错，导致回溯路径陷入死循环)
4、更新的时候改成批量更新脏矩形
"""

import heapq
from typing import List, Tuple, Set
from constants import COLORS_RGB
import pygame
import threading


class PathFinder:
    """
    路径查找算法类
    Path Finding Algorithm Class

    实现多种路径搜索算法，包括DFS路径存在性检查和Dijkstra最短路径算法
    Implements various path search algorithms including DFS path existence check and Dijkstra shortest path algorithm
    """

    def __init__(self, grid_num: int):
        """
        初始化路径查找器
        Initialize pathfinder

        Args:
            grid_num (int): 网格数量 - Number of grids
        """
        self.grid_num = grid_num

    def is_valid_move(self, x: int, y: int, grids) -> bool:
        """
        检查坐标(x,y)是否是可以移动的有效位置
        Check if coordinate (x,y) is a valid movable position

        Args:
            x (int): x坐标 - x coordinate
            y (int): y坐标 - y coordinate
            grids: 网格对象数组 - Grid objects array

        Returns:
            bool: 是否可以移动 - Whether movement is allowed
        """
        return (0 <= x < self.grid_num and 0 <= y < self.grid_num
                and grids[x][y].color != COLORS_RGB['black'])

    def spot_weight(self, x: int, y: int, weight_list: object, start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """
        获取指定坐标点的移动权重
        Get movement weight for specified coordinate point

        Args:
            x (int): x坐标 - x coordinate
            y (int): y坐标 - y coordinate
            weight_list: 权重列表 - Weight list
            start (Tuple[int, int]): 起点坐标 - Start point coordinates
            end (Tuple[int, int]): 终点坐标 - End point coordinates

        Returns:
            float: 移动权重值 - Movement weight value
        """
        # 起点和终点的权重固定为1.0，其他位置使用权重列表中的值
        # Start and end points have fixed weight of 1.0, other positions use values from weight list
        if (x, y) != start and (x, y) != end:
            return weight_list[x][y]
        return 1.0

    def bfs(self, screen, start: Tuple[int, int], end: Tuple[int, int], grids, weight_list,
            show_progress: bool = False, callback=None) -> Tuple[List[Tuple[int, int]], float]:
        """
        使用广度优先搜索(BFS)寻找从起点到终点的路径
        Use Breadth-First Search (BFS) to find a path from start to end point

        Args:
            screen: pygame的显示窗口 - Pygame display window
            start (Tuple[int, int]): 起点坐标 - Start point coordinates
            end (Tuple[int, int]): 终点坐标 - End point coordinates
            grids: 网格对象数组 - Grid objects array
            weight_list: 权重列表 - Weight list
            show_progress (bool, optional): 是否显示搜索过程 - Whether to show search progress
            callback (callable, optional): 进度回调函数 - Progress callback function

        Returns:
            Tuple[List[Tuple[int, int]], float]:
                - 路径坐标列表 - Path coordinate list
                - 路径总权重 - Total path weight

        """
        # 定义四个移动方向：上、下、左、右
        # Define four movement directions: up, down, left, right
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # 队列结构，存储待访问节点 - Queue structure, storing nodes to visit
        from collections import deque
        queue = deque([(start[0], start[1])])

        # 已访问节点集合 - Set of visited nodes
        visited = set()

        # 前驱节点字典，用于重建路径 - Predecessor dictionary, used to reconstruct path
        predecessor = {}

        # 可视化刷新率 - Visualization refresh rate
        refresh_rate = 1

        # 绿色高亮点列表
        green_highlight = set()
        # 紫色高亮点列表
        purple_highlight = set()

        # 主算法循环 - Main algorithm loop
        while queue:
            # 可视化进度处理 - Visualization progress handling
            if show_progress and callback:
                # 高亮显示已访问节点（绿色）- Highlight visited nodes (green)
                for node in visited:
                    x, y = node
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                            x, y) not in green_highlight:
                        callback(screen, x, y, 'green', False)
                        green_highlight.add((x, y))

                # 高亮显示待访问列表中的节点（紫色）- Highlight nodes in the to-visit list (purple)
                to_visit = set(queue)
                for x, y in to_visit:
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                            x, y) not in visited and (x, y) not in purple_highlight:
                        callback(screen, x, y, 'purple', False)
                        purple_highlight.add((x, y))

                # 通过强制刷新pygame界面实现减慢可视化的显示过程
                # Slow down the visualization display process by forcibly refreshing the Pygame interface
                # for _ in range(refresh_rate):
                #     pygame.display.update()

            # 从队列中取出一个节点 - Pop a node from the queue
            x, y = queue.popleft()

            # 如果到达终点，重建路径并返回 - If reached end point, reconstruct path and return
            if (x, y) == end:
                # 重建路径 - Reconstruct the path
                path = []
                current = (x, y)
                total_weight = 0

                # 回溯直到起点 - Backtrack until start point
                while current != start:
                    path.append(current)
                    prev = predecessor[current]
                    # 计算边的权重（使用前驱节点的权重）- Calculate edge weight (using predecessor's weight)
                    weight = self.spot_weight(prev[0], prev[1], weight_list, start, end)
                    total_weight += weight
                    current = prev

                # 添加起点并反转路径 - Add start point and reverse path
                path.append(start)
                path.reverse()
                return path, total_weight

            # 如果节点已访问过，跳过 - If node already visited, skip
            if (x, y) in visited:
                continue

            # 标记当前节点为已访问 - Mark current node as visited
            visited.add((x, y))

            # 检查所有相邻节点 - Check all adjacent nodes
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy

                # 如果相邻节点有效且未被访问，记录前驱并加入队列
                # If adjacent node is valid and not visited, record predecessor and add to queue
                if self.is_valid_move(new_x, new_y, grids) and (new_x, new_y) not in visited:
                    # 记录前驱节点 - Record predecessor node
                    predecessor[(new_x, new_y)] = (x, y)
                    queue.append((new_x, new_y))

        # 未找到路径 - No path found
        return [], float('inf')

    def dfs(self, screen, start: Tuple[int, int], end: Tuple[int, int], grids, weight_list,
            show_progress: bool = False, callback=None) -> Tuple[List[Tuple[int, int]], float]:
        """
        使用深度优先搜索(DFS)寻找从起点到终点的路径
        Use Depth-First Search (DFS) to find a path from start to end point

        Args:
            screen: pygame的显示窗口 - Pygame display window
            start (Tuple[int, int]): 起点坐标 - Start point coordinates
            end (Tuple[int, int]): 终点坐标 - End point coordinates
            grids: 网格对象数组 - Grid objects array
            weight_list: 权重列表 - Weight list
            show_progress (bool, optional): 是否显示搜索过程 - Whether to show search progress
            callback (callable, optional): 进度回调函数 - Progress callback function

        Returns:
            Tuple[List[Tuple[int, int]], float]:
                - 路径坐标列表 - Path coordinate list
                - 路径总权重 - Total path weight

        """
        # 定义四个移动方向：上、下、左、右
        # Define four movement directions: up, down, left, right
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # 栈结构，存储待访问节点 - Stack structure, storing nodes to visit
        stack = [(start[0], start[1])]

        # 已访问节点集合 - Set of visited nodes
        visited = set()

        # 前驱节点字典，用于重建路径 - Predecessor dictionary, used to reconstruct path
        predecessor = {}

        # 可视化刷新率 - Visualization refresh rate
        refresh_rate = 1

        # 绿色高亮点列表
        green_highlight = set()
        # 紫色高亮点列表
        purple_highlight = set()

        # 主算法循环 - Main algorithm loop
        while stack:
            # 可视化进度处理 - Visualization progress handling
            if show_progress and callback:
                # 高亮显示已访问节点（绿色）- Highlight visited nodes (green)
                for node in visited:
                    x, y = node
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                            x, y) not in green_highlight:
                        callback(screen, x, y, 'green', False)
                        green_highlight.add((x, y))

                # 高亮显示待访问列表中的节点（紫色）- Highlight nodes in the to-visit list (purple)
                for x, y in stack:
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                            x, y) not in visited and (x, y) not in purple_highlight:
                        callback(screen, x, y, 'purple', False)
                        purple_highlight.add((x, y))

                # 通过强制刷新pygame界面实现减慢可视化的显示过程
                # Slow down the visualization display process by forcibly refreshing the Pygame interface
                # for _ in range(refresh_rate):
                #     pygame.display.update()

            # 从栈中取出一个节点 - Pop a node from the stack
            x, y = stack.pop()

            # 如果到达终点，重建路径并返回 - If reached end point, reconstruct path and return
            if (x, y) == end:
                # 重建路径 - Reconstruct the path
                path = []
                current = (x, y)
                total_weight = 0

                # 回溯直到起点 - Backtrack until start point
                while current != start:
                    path.append(current)
                    prev = predecessor[current]
                    # 计算边的权重（使用前驱节点的权重）- Calculate edge weight (using predecessor's weight)
                    weight = self.spot_weight(prev[0], prev[1], weight_list, start, end)
                    total_weight += weight
                    current = prev

                # 添加起点并反转路径 - Add start point and reverse path
                path.append(start)
                path.reverse()
                return path, total_weight

            # 如果节点已访问过，跳过 - If node already visited, skip
            if (x, y) in visited:
                continue

            # 标记当前节点为已访问 - Mark current node as visited
            visited.add((x, y))

            # 检查所有相邻节点 - Check all adjacent nodes
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy

                # 如果相邻节点有效且未被访问，记录前驱并加入栈
                # If adjacent node is valid and not visited, record predecessor and add to stack
                if self.is_valid_move(new_x, new_y, grids) and (new_x, new_y) not in visited:
                    # 记录前驱节点 - Record predecessor node
                    predecessor[(new_x, new_y)] = (x, y)
                    stack.append((new_x, new_y))

        # 未找到路径 - No path found
        return [], float('inf')

    def dijkstra(self, screen, start: Tuple[int, int], end: Tuple[int, int], grids, weight_list,
                 show_progress: bool = False, callback=None) -> Tuple[List[Tuple[int, int]], float]:
        """
        Dijkstra算法寻找从起点到终点的最短路径
        Dijkstra algorithm to find the shortest path from start to end point

        Args:
            screen: pygame的显示窗口 - Pygame display window
            start (Tuple[int, int]): 起点坐标 - Start point coordinates
            end (Tuple[int, int]): 终点坐标 - End point coordinates
            grids: 网格对象数组 - Grid objects array
            weight_list: 权重列表 - Weight list
            show_progress (bool, optional): 是否显示搜索过程 - Whether to show search progress
            callback (callable, optional): 进度回调函数 - Progress callback function

        Returns:
            Tuple[List[Tuple[int, int]], float]:
                - 最短路径坐标列表 - Shortest path coordinate list
                - 路径总权重 - Total path weight

        """
        # 定义四个移动方向：右、左、下、上
        # Define four movement directions: right, left, down, up
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # 已访问节点集合 - Set of visited nodes
        visited: Set[Tuple[int, int]] = set()

        # 最小距离矩阵，初始化为无穷大 - Minimum distance matrix, initialized to infinity
        min_distance = [[float('inf')] * self.grid_num for _ in range(self.grid_num)]
        min_distance[start[0]][start[1]] = 0  # 起点距离为0 - Start point distance is 0

        # 优先队列（最小堆），存储(距离, 坐标) - Priority queue (min heap), storing (distance, coordinate)
        to_visit = [(0, start)]

        # 计算可视化刷新率 - Calculate visualization refresh rate
        refresh_rate = 1  # 刷新率应大于1 - refresh_rate should be bigger than 1
        # 大网格使用较高刷新率避免卡顿，小网格使用较低刷新率保证可视化效果
        # Large grids use higher refresh rate to avoid lag, small grids use lower refresh rate for better visualization
        # refresh_rate = 80 if self.grid_num > 800 else min(10, self.grid_num // 100)

        # rate = refresh_rate  # 刷新率计数器 - Refresh rate counter

        # 绿色高亮点列表
        green_highlight = set()
        # 紫色高亮点列表
        purple_highlight = set()

        # 主算法循环 - Main algorithm loop
        while to_visit:
            # print(to_visit)
            # 可视化进度处理 - Visualization progress handling
            if show_progress and callback:
                # 高亮显示已确定最短距离的节点（绿色）- Highlight confirmed nodes with the determined shortest distance(green)
                for node in visited:
                    x, y = node
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                            x, y) not in green_highlight:
                        callback(screen, x, y, 'green', False)  # 使用绿色表示已确定节点
                        green_highlight.add((x, y))
                # 高亮显示待访问列表中的节点 - Highlight nodes in the to-visit list
                for _, (x, y) in to_visit:
                    # 检查是否为障碍物、起点、终点或者出现在visited中的点
                    # Check if it's an obstacle, start point, end point or points in visited-list
                    """
                    这里存在一个我也处理不好的问题，就是说当一个节点的最短距离（实际上是最小权重大小）已经确定时，"to_visit"堆中还存有该点
                    应该是之前入堆时存在重复入堆的问题，但是要直接将堆中该点全部去除又会造成复杂度上升，我本人没有这样做，而是在显示时没有显示这些重复点。
                    There is a problem here that I also cannot handle well: 
                    when the shortest distance (actually the smallest weight) of a node has been determined, 
                    the "to_visit" heap still contains that node. 
                    This should be due to the issue of duplicate entries when the node was previously added to the heap. 
                    However, directly removing all instances of that node from the heap would increase the complexity. 
                    I didn't do that myself; 
                    instead, I simply did not display these duplicate nodes when showing the results.
                    """
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                            x, y) not in visited and (x, y) not in purple_highlight:
                        callback(screen, x, y, 'purple', False)  # 使用紫色高亮 - Use purple for highlighting
                        purple_highlight.add((x, y))

                # refresh_rate = rate  # 重置计数器 - Reset counter
                # 通过强制刷新pygame界面实现减慢可视化的显示过程
                # Slow down the visualization display process by forcibly refreshing the Pygame interface
                # for _ in range(refresh_rate):
                #     pygame.display.update()

                # for distance, (x, y) in to_visit:
                #     # 只打印未访问的节点（虽然理论上所有节点都是未访问的）
                #     # Only print unvisited nodes (although theoretically all nodes are unvisited)
                #     if (x, y) in visited:
                #         print(f"节点: ({x}, {y}), 距离: {distance}")
                #     else:
                #         print("None")

            # 从优先队列中取出距离最小的节点 - Pop the node with the smallest distance from priority queue
            dist, current = heapq.heappop(to_visit)

            # 如果到达终点，提前终止 - If reached end point, terminate early
            if current == end:
                break

            # 如果节点已访问过，跳过 - If node already visited, skip
            if current in visited:
                continue

            # 标记当前节点为已访问 - Mark current node as visited
            visited.add(current)

            # 检查所有相邻节点 - Check all adjacent nodes
            for dx, dy in directions:
                x, y = current[0] + dx, current[1] + dy

                # 如果相邻节点有效 - If adjacent node is valid
                if self.is_valid_move(x, y, grids):
                    # 计算当前节点到相邻节点的权重 - Calculate weight from current to adjacent node
                    weight = self.spot_weight(current[0], current[1], weight_list, start, end)

                    # 计算新的距离 - Calculate new distance
                    new_dist = min_distance[current[0]][current[1]] + weight

                    # 如果找到更短路径，更新距离并加入优先队列
                    # If found shorter path, update distance and add to priority queue
                    if new_dist < min_distance[x][y]:
                        min_distance[x][y] = new_dist
                        heapq.heappush(to_visit, (new_dist, (x, y)))

        # 检查终点是否可达 - Check if end point is reachable
        """
        一定要检查，否则:
        假如起点能到达的所有点都已经探索完毕了，但是终点和终点附近的其他点无法从起点出发抵达，
        如果未进行检查，那么后面回溯的时候就会发现，路径会在在终点和周围几个点之间来回往复，因为终点附近的点的min_distance都是float('inf')。
        也可以用记录前驱节点的方法重写这段代码。
        Be sure to check, otherwise: 
        If all the points reachable from the starting point have been explored, 
        but the endpoint and other points near the endpoint cannot be reached from the starting point, 
        then without this check, 
        you will later find during backtracking that the path will keep going back and forth between the endpoint and a 
        few nearby points, because the min_distance of points near the endpoint is all float('inf'). 
        You can also rewrite this part of the code using a method that records predecessor nodes.
        """
        if min_distance[end[0]][end[1]] == float('inf'):
            return [], float('inf')

        # 重建最短路径 - Reconstruct the shortest path
        path = []
        x, y = end  # 从终点开始回溯 - Start backtracking from end point

        # 回溯直到起点 - Backtrack until start point
        while (x, y) != start:
            path.append((x, y))

            # 检查四个方向，找到前驱节点 - Check four directions to find predecessor
            found = False
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # 如果相邻节点有效 - If adjacent node is valid
                if self.is_valid_move(nx, ny, grids):
                    # 计算相邻节点的权重 - Calculate adjacent node's weight
                    weight = self.spot_weight(nx, ny, weight_list, start, end)

                    # 如果满足距离关系，说明这是路径上的前一个节点
                    # If distance relationship is satisfied, this is the previous node on the path
                    if min_distance[nx][ny] == min_distance[x][y] - weight:
                        x, y = nx, ny
                        found = True
                        break

            # 如果找不到有效前驱，终止循环 - If no valid predecessor found, terminate loop
            if not found:
                break

        # 添加起点并反转路径 - Add start point and reverse path
        path.append(start)
        path.reverse()

        # 返回路径和总权重 - Return path and total weight
        return path, min_distance[end[0]][end[1]]

    def a_star(self, screen, start: Tuple[int, int], end: Tuple[int, int], grids, weight_list,
               show_progress: bool = False, callback=None) -> Tuple[List[Tuple[int, int]], float]:
        """
        使用A*算法寻找从起点到终点的最短路径
        Use A* algorithm to find the shortest path from start to end point

        Args:
            screen: pygame的显示窗口 - Pygame display window
            start (Tuple[int, int]): 起点坐标 - Start point coordinates
            end (Tuple[int, int]): 终点坐标 - End point coordinates
            grids: 网格对象数组 - Grid objects array
            weight_list: 权重列表 - Weight list
            show_progress (bool, optional): 是否显示搜索过程 - Whether to show search progress
            callback (callable, optional): 进度回调函数 - Progress callback function

        Returns:
            Tuple[List[Tuple[int, int]], float]:
                - 最短路径坐标列表 - Shortest path coordinate list
                - 路径总权重 - Total path weight

        """
        # 定义四个移动方向：上、下、左、右
        # Define four movement directions: up, down, left, right
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # 启发式函数：曼哈顿距离 - Heuristic function: Manhattan distance
        def heuristic(a, b):
            return (abs(a[0] - b[0]) + abs(a[1] - b[1])) * 11.4514

        # 优先队列，存储(f值, g值, 坐标) - Priority queue, storing (f_value, g_value, coordinate)
        import heapq
        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, end), 0, start[0], start[1]))

        # 已访问节点集合 - Set of visited nodes
        visited = set()

        # 前驱节点字典，用于重建路径 - Predecessor dictionary, used to reconstruct path
        predecessor = {}

        # g值字典：从起点到当前节点的实际代价 - g_value dictionary: actual cost from start to current node
        g_value = {}
        g_value[start] = 0

        # 可视化刷新率 - Visualization refresh rate
        refresh_rate = 1

        # 绿色高亮点列表
        green_highlight = set()
        # 紫色高亮点列表
        purple_highlight = set()

        # 主算法循环 - Main algorithm loop
        while open_set:
            # 可视化进度处理 - Visualization progress handling
            if show_progress and callback:
                # 高亮显示已访问节点（绿色）- Highlight visited nodes (green)
                for node in visited:
                    x, y = node
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                            x, y) not in green_highlight:
                        callback(screen, x, y, 'green', False)
                        green_highlight.add((x, y))

                # 高亮显示待访问列表中的节点（紫色）- Highlight nodes in the open set (purple)
                for _, _, x, y in open_set:
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                            x, y) not in visited and (x, y) not in purple_highlight:
                        callback(screen, x, y, 'purple', False)
                        purple_highlight.add((x, y))

                # 通过强制刷新pygame界面实现减慢可视化的显示过程
                # Slow down the visualization display process by forcibly refreshing the Pygame interface
                # for _ in range(refresh_rate):
                #     pygame.display.update()

            # 从优先队列中取出f值最小的节点 - Pop the node with the smallest f-value from priority queue
            current_f, current_g, x, y = heapq.heappop(open_set)
            current = (x, y)

            # 如果到达终点，重建路径并返回 - If reached end point, reconstruct path and return
            if current == end:
                # 重建路径 - Reconstruct the path
                path = []
                current_node = current
                total_weight = g_value[current_node]

                # 回溯直到起点 - Backtrack until start point
                while current_node != start:
                    path.append(current_node)
                    current_node = predecessor[current_node]

                # 添加起点并反转路径 - Add start point and reverse path
                path.append(start)
                path.reverse()
                return path, total_weight

            # 如果节点已访问过，跳过 - If node already visited, skip
            if current in visited:
                continue

            # 标记当前节点为已访问 - Mark current node as visited
            visited.add(current)

            # 检查所有相邻节点 - Check all adjacent nodes
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                neighbor = (new_x, new_y)

                # 如果相邻节点有效且未被访问 - If adjacent node is valid and not visited
                if self.is_valid_move(new_x, new_y, grids) and neighbor not in visited:
                    # 计算从起点到邻居节点的实际代价 - Calculate actual cost from start to neighbor
                    weight = self.spot_weight(new_x, new_y, weight_list, start, end)
                    tentative_g = g_value[current] + weight

                    # 如果找到更优路径，更新代价并加入优先队列
                    # If found better path, update cost and add to priority queue
                    if neighbor not in g_value or tentative_g < g_value[neighbor]:
                        # 更新前驱节点和g值 - Update predecessor and g-value
                        predecessor[neighbor] = current
                        g_value[neighbor] = tentative_g

                        # 计算f值 = g值 + 启发式估计值 - Calculate f-value = g-value + heuristic estimate
                        f_value = tentative_g + heuristic(neighbor, end)

                        # 将邻居加入优先队列 - Add neighbor to priority queue
                        heapq.heappush(open_set, (f_value, tentative_g, new_x, new_y))

        # 未找到路径 - No path found
        return [], float('inf')
