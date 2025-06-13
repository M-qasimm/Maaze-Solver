import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

st.set_page_config(page_title="Maze Solver", layout="centered")
st.title("🧭 Maze Solver using BFS, DFS or A*")

st.markdown("""
Use the grid below to draw your own maze:
- **0**: Walkable path  
- **1**: Wall/block  
Choose algorithm and click Solve.
""")

rows = st.number_input("Rows", 5, 15, 5)
cols = st.number_input("Cols", 5, 15, 5)

algorithm = st.selectbox("Choose Algorithm", ["BFS", "DFS", "A*"])

custom_maze = []
with st.form("maze_form"):
    for i in range(int(rows)):
        default_row = ",".join(["0"] * int(cols))
        row = st.text_input(f"Row {i+1}", default_row)
        custom_maze.append(row)
    submitted = st.form_submit_button("Solve Maze")

def bfs(maze, start, goal):
    queue = deque([start])
    visited = set()
    parent = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            x, y = current[0]+dx, current[1]+dy
            if 0<=x<len(maze) and 0<=y<len(maze[0]) and maze[x][y]==0 and (x,y) not in visited:
                visited.add((x,y))
                parent[(x,y)] = current
                queue.append((x,y))
    path = []
    if goal in parent:
        while goal:
            path.append(goal)
            goal = parent[goal]
        path.reverse()
    return path

def dfs(maze, start, goal):
    stack = [start]
    visited = set()
    parent = {start: None}
    while stack:
        current = stack.pop()
        if current == goal:
            break
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            x, y = current[0]+dx, current[1]+dy
            if 0<=x<len(maze) and 0<=y<len(maze[0]) and maze[x][y]==0 and (x,y) not in visited:
                visited.add((x,y))
                parent[(x,y)] = current
                stack.append((x,y))
    path = []
    if goal in parent:
        while goal:
            path.append(goal)
            goal = parent[goal]
        path.reverse()
    return path

import heapq
def astar(maze, start, goal):
    def h(p): return abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    open_set = [(h(start), 0, start)]
    parent = {start: None}
    cost = {start: 0}
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            break
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            x, y = current[0]+dx, current[1]+dy
            next_node = (x, y)
            if 0<=x<len(maze) and 0<=y<len(maze[0]) and maze[x][y]==0:
                new_cost = g + 1
                if next_node not in cost or new_cost < cost[next_node]:
                    cost[next_node] = new_cost
                    priority = new_cost + h(next_node)
                    heapq.heappush(open_set, (priority, new_cost, next_node))
                    parent[next_node] = current
    path = []
    if goal in parent:
        while goal:
            path.append(goal)
            goal = parent[goal]
        path.reverse()
    return path

if submitted:
    try:
        maze = [list(map(int, row.split(','))) for row in custom_maze]
        start = (0, 0)
        end = (int(rows)-1, int(cols)-1)

        if algorithm == "BFS":
            path = bfs(maze, start, end)
        elif algorithm == "DFS":
            path = dfs(maze, start, end)
        else:
            path = astar(maze, start, end)

        st.subheader("Maze")
        st.dataframe(np.array(maze))

        fig, ax = plt.subplots()
        ax.imshow(maze, cmap="gray_r")

        if path:
            px, py = zip(*path)
            ax.plot(py, px, color="red", linewidth=2)
            st.success("✅ Path found!")
        else:
            st.error("🚫 No path found")

        ax.set_title(f"Solution using {algorithm}")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error: {e}")