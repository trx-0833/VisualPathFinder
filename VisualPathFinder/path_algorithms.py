"""
检查时发现了一下问题：
1、显示搜索过程的刷新率，要在项目基本完成后重新考量
2、dijkstra算法的绿色（已搜索点）和紫色（正在搜索点）出现交替闪烁现象，应该是不被允许的(原因在于已经确定最短距离的点重复入列，导致会反复计算这些点)
3、GRID_NUM=30时，导入maze_2迷宫，dijkstra算法搜索会出现问题(错误来源于回溯路径时出错，导致回溯路径陷入死循环)
4、更新的时候改成批量更新脏矩形
"""

import heapq
from typing import List, Tuple, Set, Dict, Deque
from collections import deque
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
                    if grids[x][y].color != COLORS_RGB['black'] and node != start and node != end and node not in green_highlight:
                        callback(screen, x, y, 'green', False)
                        green_highlight.add(node)

                # 高亮显示待访问列表中的节点（紫色）- Highlight nodes in the to-visit list (purple)
                to_visit = set(queue)
                for node in to_visit:
                    x, y = node
                    if grids[x][y].color != COLORS_RGB['black'] and node != start and node != end and node not in visited and node not in purple_highlight:
                        callback(screen, x, y, 'purple', False)
                        purple_highlight.add(node)

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

    def bidirectional_bfs(self, screen, start: Tuple[int, int], end: Tuple[int, int], grids, weight_list,
                          show_progress: bool = False, callback=None) -> Tuple[List[Tuple[int, int]], float]:
        """
        双向广度优先搜索 (D-BFS) - 从起点和终点同时进行BFS搜索
        Bidirectional Breadth-First Search (D-BFS) - BFS search from both start and end simultaneously

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
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # 从前向搜索（从起点开始）
        forward_queue = deque([start])
        forward_visited = {start: None}  # 存储节点和前驱节点
        forward_dist = {start: 0}  # 从起点到各点的距离

        # 从后向搜索（从终点开始）
        backward_queue = deque([end])
        backward_visited = {end: None}  # 存储节点和前驱节点
        backward_dist = {end: 0}  # 从终点到各点的距离

        # 相遇点
        meeting_point = None

        # 可视化相关变量
        green_highlight = set()
        purple_highlight = set()

        # 双向搜索循环
        while forward_queue and backward_queue:
            # 可视化进度处理
            if show_progress and callback:
                # 高亮显示前向搜索已访问节点（绿色）
                for node in forward_visited:
                    x, y = node
                    if grids[x][y].color != COLORS_RGB[
                         'black'] and node != start and node != end and node not in green_highlight:
                        callback(screen, x, y, 'green', False)
                        green_highlight.add(node)

                # 高亮显示后向搜索已访问节点（绿色）
                for node in backward_visited:
                    x, y = node
                    if grids[x][y].color != COLORS_RGB[
                         'black'] and node != start and node != end and node not in green_highlight:
                        callback(screen, x, y, 'green', False)
                        green_highlight.add(node)

                # 高亮显示前向搜索待访问节点（紫色）
                for x, y in forward_queue:
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                     x, y) not in forward_visited and (x, y) not in purple_highlight:
                        callback(screen, x, y, 'purple', False)
                        purple_highlight.add((x, y))

                # 高亮显示后向搜索待访问节点（紫色）
                for x, y in backward_queue:
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                    x, y) not in backward_visited and (x, y) not in purple_highlight:
                        callback(screen, x, y, 'purple', False)
                        purple_highlight.add((x, y))

            # 从前向队列扩展一层
            forward_size = len(forward_queue)
            for _ in range(forward_size):
                x, y = forward_queue.popleft()

                # 检查是否在后向搜索中访问过（相遇）
                if (x, y) in backward_visited:
                    meeting_point = (x, y)
                    break

                # 扩展邻居节点
                for dx, dy in directions:
                    new_x, new_y = x + dx, y + dy
                    new_pos = (new_x, new_y)

                    if self.is_valid_move(new_x, new_y, grids) and new_pos not in forward_visited:
                        forward_visited[new_pos] = (x, y)
                        forward_dist[new_pos] = forward_dist[(x, y)] + self.spot_weight(x, y, weight_list, start, end)
                        forward_queue.append(new_pos)

            if meeting_point:
                break

            # 从后向队列扩展一层
            backward_size = len(backward_queue)
            for _ in range(backward_size):
                x, y = backward_queue.popleft()

                # 检查是否在前向搜索中访问过（相遇）
                if (x, y) in forward_visited:
                    meeting_point = (x, y)
                    break

                # 扩展邻居节点
                for dx, dy in directions:
                    new_x, new_y = x + dx, y + dy
                    new_pos = (new_x, new_y)

                    if self.is_valid_move(new_x, new_y, grids) and new_pos not in backward_visited:
                        backward_visited[new_pos] = (x, y)
                        backward_dist[new_pos] = backward_dist[(x, y)] + self.spot_weight(x, y, weight_list, start, end)
                        backward_queue.append(new_pos)

            if meeting_point:
                break

        # 如果没有相遇点，返回无路径
        if not meeting_point:
            return [], float('inf')

        # 重建路径：从前向路径 + 后向路径（反向）
        # 前向部分：从起点到相遇点
        forward_path = []
        current = meeting_point
        while current != start:
            forward_path.append(current)
            current = forward_visited[current]
        forward_path.append(start)
        forward_path.reverse()

        # 后向部分：从相遇点到终点（不包括相遇点）
        backward_path = []
        current = meeting_point
        while current != end:
            current = backward_visited[current]
            if current != end:
                backward_path.append(current)

        # 合并路径
        full_path = forward_path + backward_path

        # 计算总权重
        total_weight = forward_dist[meeting_point] + backward_dist[meeting_point]

        return full_path, total_weight

    def bidirectional_dfs(self, screen, start: Tuple[int, int], end: Tuple[int, int], grids, weight_list,
                          show_progress: bool = False, callback=None) -> Tuple[List[Tuple[int, int]], float]:
        """
        双向深度优先搜索 (D-DFS) - 从起点和终点同时进行DFS搜索
        Bidirectional Depth-First Search (D-DFS) - DFS search from both start and end simultaneously

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
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # 检查起点和终点是否有效
        if not self.is_valid_move(start[0], start[1], grids):
            return [], float('inf')
        if not self.is_valid_move(end[0], end[1], grids):
            return [], float('inf')

        # 从前向搜索（从起点开始）
        forward_stack = [start]
        forward_parent = {start: None}  # 存储节点和前驱节点
        forward_dist = {start: 0}  # 从起点到各点的距离

        # 从后向搜索（从终点开始）
        backward_stack = [end]
        backward_parent = {end: None}  # 存储节点和前驱节点
        backward_dist = {end: 0}  # 从终点到各点的距离

        # 相遇点
        meeting_point = None

        # 可视化相关变量
        green_highlight = set()
        purple_highlight = set()

        # 交替执行前向和后向搜索
        while forward_stack and backward_stack:
            # 可视化进度处理
            if show_progress and callback:
                # 高亮显示前向搜索已访问节点（绿色）
                for node in forward_parent:
                    x, y = node
                    if grids[x][y].color != COLORS_RGB[
                         'black'] and node != start and node != end and node not in green_highlight:
                        callback(screen, x, y, 'green', False)
                        green_highlight.add(node)

                # 高亮显示后向搜索已访问节点（绿色）
                for node in backward_parent:
                    x, y = node
                    if grids[x][y].color != COLORS_RGB[
                         'black'] and node != start and node != end and node not in green_highlight:
                        callback(screen, x, y, 'green', False)
                        green_highlight.add(node)

                # 高亮显示前向搜索待访问节点（紫色）
                for x, y in forward_stack:
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                     x, y) not in forward_parent and (x, y) not in purple_highlight:
                        callback(screen, x, y, 'purple', False)
                        purple_highlight.add((x, y))

                # 高亮显示后向搜索待访问节点（紫色）
                for x, y in backward_stack:
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                    x, y) not in backward_parent and (x, y) not in purple_highlight:
                        callback(screen, x, y, 'purple', False)
                        purple_highlight.add((x, y))

            # 前向搜索一步
            if forward_stack:
                x, y = forward_stack.pop()

                # 检查是否在后向搜索中访问过（相遇）
                if (x, y) in backward_parent:
                    meeting_point = (x, y)
                    break

                # 扩展邻居节点
                for dx, dy in directions:
                    new_x, new_y = x + dx, y + dy
                    new_pos = (new_x, new_y)

                    if self.is_valid_move(new_x, new_y, grids) and new_pos not in forward_parent:
                        forward_parent[new_pos] = (x, y)
                        forward_dist[new_pos] = forward_dist[(x, y)] + self.spot_weight(new_x, new_y, weight_list,
                                                                                        start, end)
                        forward_stack.append(new_pos)

            # 后向搜索一步
            if backward_stack:
                x, y = backward_stack.pop()

                # 检查是否在前向搜索中访问过（相遇）
                if (x, y) in forward_parent:
                    meeting_point = (x, y)
                    break

                # 扩展邻居节点
                for dx, dy in directions:
                    new_x, new_y = x + dx, y + dy
                    new_pos = (new_x, new_y)

                    if self.is_valid_move(new_x, new_y, grids) and new_pos not in backward_parent:
                        backward_parent[new_pos] = (x, y)
                        backward_dist[new_pos] = backward_dist[(x, y)] + self.spot_weight(new_x, new_y, weight_list,
                                                                                          start, end)
                        backward_stack.append(new_pos)

        # 如果没有相遇点，返回无路径
        if not meeting_point:
            return [], float('inf')

        # 重建路径
        # 从前向相遇点到起点
        forward_path = []
        current = meeting_point
        while current is not None:
            forward_path.append(current)
            current = forward_parent[current]
        forward_path.reverse()  # 反转得到从起点到相遇点的路径

        # 从相遇点的后向父节点到终点
        backward_path = []
        current = backward_parent[meeting_point]  # 从相遇点的下一个节点开始
        while current is not None:
            backward_path.append(current)
            current = backward_parent[current]

        # 合并路径（注意：相遇点在forward_path中已经包含）
        full_path = forward_path + backward_path

        # 计算总权重（注意：相遇点的权重不应该重复计算）
        # 前向距离包含相遇点权重，后向距离也包含相遇点权重，所以需要减去一次相遇点权重
        meeting_weight = self.spot_weight(meeting_point[0], meeting_point[1], weight_list, start, end)
        total_weight = forward_dist[meeting_point] + backward_dist[meeting_point] - meeting_weight

        return full_path, total_weight

    def b_star(self, screen, start: Tuple[int, int], end: Tuple[int, int], grids, weight_list,
               show_progress: bool = False, callback=None) -> Tuple[List[Tuple[int, int]], float]:
        """
        B*算法 - 结合最佳优先搜索和A*的改进算法
        B* Algorithm - Improved algorithm combining Best-First Search and A*

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
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # 启发式函数：曼哈顿距离
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # 双启发式：欧几里得距离
        def heuristic_euclidean(a, b):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

        # 优先队列，存储(f值, g值, h值, 坐标)
        open_set = []
        g_start = 0
        h_start = heuristic(start, end)
        f_start = g_start + h_start
        heapq.heappush(open_set, (f_start, g_start, h_start, start[0], start[1]))

        # 已访问节点集合
        visited = set()

        # 前驱节点字典
        predecessor = {}

        # g值字典：从起点到当前节点的实际代价
        g_value = {start: 0}

        # 可视化相关变量
        green_highlight = set()
        purple_highlight = set()

        # 主算法循环
        while open_set:
            # 可视化进度处理
            if show_progress and callback:
                # 高亮显示已访问节点（绿色）
                for node in visited:
                    x, y = node
                    if grids[x][y].color != COLORS_RGB[
                         'black'] and node != start and node != end and node not in green_highlight:
                        callback(screen, x, y, 'green', False)
                        green_highlight.add(node)

                # 高亮显示待访问列表中的节点（紫色）
                for _, _, _, x, y in open_set:
                    if grids[x][y].color != COLORS_RGB['black'] and (x, y) != start and (x, y) != end and (
                     x, y) not in visited and (x, y) not in purple_highlight:
                        callback(screen, x, y, 'purple', False)
                        purple_highlight.add((x, y))

            # 从优先队列中取出f值最小的节点
            current_f, current_g, current_h, x, y = heapq.heappop(open_set)
            current = (x, y)

            # 如果到达终点，重建路径并返回
            if current == end:
                # 重建路径
                path = []
                current_node = current
                total_weight = g_value[current_node]

                # 回溯直到起点
                while current_node != start:
                    path.append(current_node)
                    current_node = predecessor[current_node]

                # 添加起点并反转路径
                path.append(start)
                path.reverse()
                return path, total_weight

            # 如果节点已访问过，跳过
            if current in visited:
                continue

            # 标记当前节点为已访问
            visited.add(current)

            # 检查所有相邻节点
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                neighbor = (new_x, new_y)

                # 如果相邻节点有效且未被访问
                if self.is_valid_move(new_x, new_y, grids) and neighbor not in visited:
                    # 计算从起点到邻居节点的实际代价
                    weight = self.spot_weight(new_x, new_y, weight_list, start, end)
                    tentative_g = g_value[current] + weight

                    # 如果找到更优路径，更新代价并加入优先队列
                    if neighbor not in g_value or tentative_g < g_value[neighbor]:
                        # 更新前驱节点和g值
                        predecessor[neighbor] = current
                        g_value[neighbor] = tentative_g

                        # 计算启发式值（使用曼哈顿距离和欧几里得距离的加权平均）
                        h_manhattan = heuristic(neighbor, end)
                        h_euclidean = heuristic_euclidean(neighbor, end)
                        h_value = 0.7 * h_manhattan + 0.3 * h_euclidean

                        # B*算法的f值计算：g + h + 动态调整因子
                        # 动态调整因子基于当前节点到起点的距离
                        distance_factor = 1.0 + (tentative_g / 100.0)  # 随着距离增加，逐渐重视实际代价
                        f_value = (tentative_g + h_value * 10) * distance_factor

                        # 将邻居加入优先队列
                        heapq.heappush(open_set, (f_value, tentative_g, h_value, new_x, new_y))

        # 未找到路径
        return [], float('inf')
