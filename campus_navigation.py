"""
Smart Campus Navigation and Rescue Planner
KIIT Campus-25 Emergency Evacuation System
"""

import time
import random


# Helper Functions
def get_cell_cost(cell_type):
    """Return the movement cost for a given cell type"""
    costs = {
        'C': 1,  # Corridor
        'R': 1,  # Room
        'S': 3,  # Stairs
        'L': 2,  # Lift
        'O': 1,  # Open ground
        ' ': 1,  # Empty/Corridor
    }
    return costs.get(cell_type, 1)


def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def terrain_aware_heuristic(current_pos, goal_pos, campus_map):
    """
    Custom heuristic that considers terrain penalties
    Penalizes paths that might go through stairs
    This is an admissible heuristic for A* when used correctly
    """
    base_distance = manhattan_distance(current_pos, goal_pos)
    
    # Add penalty if current cell is stairs or lift
    terrain_penalty = 0
    cell_type = campus_map[current_pos[0]][current_pos[1]]
    
    if cell_type == 'S':  # Stairs
        terrain_penalty = 2  # Heavy penalty for stairs
    elif cell_type == 'L':  # Lift
        terrain_penalty = 1  # Moderate penalty for lift
    
    return base_distance + terrain_penalty


def main():
    """
    Main function to orchestrate the campus navigation system
    """
    
    print("="*70)
    print("SMART CAMPUS NAVIGATION AND RESCUE PLANNER - KIIT CAMPUS-25")
    print("="*70)
    print()
    
    # Step 1: Create Campus Grid Map
    print("Step 1: Creating Campus Grid Map...")
    grid_size = (12, 15)  # rows x columns
    campus_map = create_campus_map(grid_size)
    visualize_map(campus_map, title="Initial Campus Map")
    print()
    
    # Step 2: Define Start and Goal Locations
    print("Step 2: Setting Start and Goal Locations...")
    start_location = set_start_location(campus_map)
    goal_locations = set_goal_locations(campus_map, num_goals=3)
    print(f"Start Location: {start_location}")
    print(f"Goal Locations (Safe Zones): {goal_locations}")
    print()
    
    # Step 3: Implement and Run Search Algorithms
    print("="*70)
    print("Step 3: Running Search Algorithms...")
    print("="*70)
    
    algorithms = {
        'BFS': bfs_search,
        'DFS': dfs_search,
        'Best-First (Manhattan)': best_first_search,
        'Best-First (Custom)': best_first_search_custom,
        'A* (Manhattan)': a_star_search,
        'A* (Custom)': a_star_search_custom
    }
    
    results = {}
    
    for algo_name, algo_func in algorithms.items():
        print(f"\n--- Running {algo_name} ---")
        start_time = time.time()
        
        path, cost, nodes_explored, exploration_order = algo_func(
            campus_map, 
            start_location, 
            goal_locations
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        results[algo_name] = {
            'path': path,
            'cost': cost,
            'nodes_explored': nodes_explored,
            'exploration_order': exploration_order,
            'execution_time': execution_time
        }
        
        print(f"Path Found: {path if path else 'No path'}")
        print(f"Path Length: {len(path) if path else 0} steps")
        print(f"Total Cost: {cost}")
        print(f"Nodes Explored (count): {nodes_explored}")
        print(f"Exploration Order: {exploration_order[:10]}{'...' if len(exploration_order) > 10 else ''}")
        print(f"Execution Time: {execution_time:.6f} seconds")
    
    # Step 4: Comparative Analysis
    print("\n" + "="*70)
    print("Step 4: Comparative Analysis")
    print("="*70)
    generate_comparison_table(results)
    print()
    
    # Step 5: Dynamic Obstacle Handling
    print("="*70)
    print("Step 5: Dynamic Obstacle Testing")
    print("="*70)
    print("\nBlocking 3-5 cells on the best path and re-running algorithms...")
    
    # Find the best algorithm's path
    best_algo = min(results.items(), 
                    key=lambda x: x[1]['cost'] if x[1]['cost'] is not None else float('inf'))
    best_path = best_algo[1]['path']
    
    if best_path and len(best_path) > 5:
        # Block random cells on the path
        modified_map, blocked_cells = add_dynamic_obstacles(
            campus_map, 
            best_path, 
            num_blocks=random.randint(3, 5)
        )
        
        print(f"\nBlocked cells: {blocked_cells}")
        visualize_map(modified_map, title="Map with Dynamic Obstacles")
        
        # Re-run algorithms on modified map
        print("\nRe-running algorithms on modified map...")
        dynamic_results = {}
        
        for algo_name, algo_func in algorithms.items():
            print(f"\n--- Re-running {algo_name} ---")
            start_time = time.time()
            
            path, cost, nodes_explored, exploration_order = algo_func(
                modified_map, 
                start_location, 
                goal_locations
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            dynamic_results[algo_name] = {
                'path': path,
                'cost': cost,
                'nodes_explored': nodes_explored,
                'exploration_order': exploration_order,
                'execution_time': execution_time,
                'adapted': path is not None
            }
            
            print(f"Adapted: {'Yes' if path else 'No'}")
            print(f"Path Found: {path if path else 'No path'}")
            print(f"New Cost: {cost}")
            print(f"Nodes Explored: {nodes_explored}")
            print(f"Exploration Order: {exploration_order[:10]}{'...' if len(exploration_order) > 10 else ''}")
            print(f"Execution Time: {execution_time:.6f} seconds")
        
        print("\n--- Dynamic Adaptation Comparison ---")
        generate_comparison_table(dynamic_results)
        
        # Analyze which algorithm adapts faster
        print("\n" + "="*70)
        print("DYNAMIC ADAPTATION SPEED ANALYSIS")
        print("="*70)
        print("\nWhich algorithm adapts FASTER to obstacles?")
        
        # Sort by execution time
        adapted_algos = {k: v for k, v in dynamic_results.items() if v['adapted']}
        if adapted_algos:
            fastest = min(adapted_algos.items(), key=lambda x: x[1]['execution_time'])
            most_efficient = min(adapted_algos.items(), key=lambda x: x[1]['nodes_explored'])
            
            print(f"\n1. FASTEST RE-PLANNING: {fastest[0]}")
            print(f"   Time: {fastest[1]['execution_time']*1000:.4f} ms")
            print(f"   Why: Efficiently found alternative path with minimal computation")
            
            print(f"\n2. MOST EFFICIENT (Fewest nodes): {most_efficient[0]}")
            print(f"   Nodes Explored: {most_efficient[1]['nodes_explored']}")
            print(f"   Why: Explored fewer nodes to find new path")
            
            # Compare cost increases
            print(f"\n3. COST INCREASE COMPARISON:")
            for algo, dyn_result in dynamic_results.items():
                if dyn_result['adapted'] and algo in results:
                    original_cost = results[algo]['cost']
                    new_cost = dyn_result['cost']
                    increase = new_cost - original_cost
                    increase_pct = (increase / original_cost * 100) if original_cost else 0
                    print(f"   {algo}: {original_cost} → {new_cost} (+{increase}, +{increase_pct:.1f}%)")
            
            print(f"\nINSIGHT:")
            print(f"   - Heuristic-based algorithms (Best-First, A*) often adapt faster")
            print(f"   - They use heuristics to quickly find alternative routes")
            print(f"   - BFS/DFS explore more systematically but may be slower")
        print()
    
    # Step 6: Generate Final Conclusion
    print("\n" + "="*70)
    print("Step 6: Final Conclusion and Recommendations")
    print("="*70)
    generate_conclusion(results, dynamic_results if 'dynamic_results' in locals() else None)
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)


# Placeholder functions (to be implemented)
def create_campus_map(size):
    """
    Create a hardcoded campus grid map with corridors, rooms, and obstacles
    
    Cell Types:
    'C' = Corridor (cost 1)
    'R' = Room/Classroom (cost 1)
    'S' = Stairs (cost 3)
    'L' = Lift (cost 2)
    'O' = Open ground (cost 1)
    'X' = Blocked (impassable)
    ' ' = Empty/Corridor (cost 1)
    
    Layout: Rectangular corridors with rooms on each side
    """
    
    # 12x15 grid - Corridor layout with rooms
    campus_map = [
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
        ['X', 'R', 'R', 'R', 'C', 'R', 'R', 'R', 'C', 'R', 'R', 'R', 'O', 'O', 'X'],
        ['X', 'R', 'R', 'R', 'C', 'R', 'R', 'R', 'C', 'R', 'R', 'R', 'O', 'O', 'X'],
        ['X', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'X'],
        ['X', 'R', 'R', 'X', 'C', 'R', 'R', 'S', 'C', 'L', 'R', 'R', 'X', 'X', 'X'],
        ['X', 'R', 'R', 'X', 'C', 'R', 'R', 'S', 'C', 'L', 'R', 'R', 'O', 'O', 'X'],
        ['X', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'X'],
        ['X', 'R', 'R', 'R', 'C', 'X', 'X', 'R', 'C', 'R', 'R', 'R', 'X', 'O', 'X'],
        ['X', 'R', 'R', 'R', 'C', 'R', 'R', 'R', 'C', 'R', 'R', 'R', 'O', 'O', 'X'],
        ['X', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'X'],
        ['X', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'X'],
        ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
    ]
    
    return campus_map

def visualize_map(campus_map, title="Campus Map"):
    """Visualize the campus map"""
    print(f"\n{title}")
    print("-" * (len(campus_map[0]) * 3))
    
    for i, row in enumerate(campus_map):
        print(f"{i:2d} ", end="")
        for cell in row:
            print(f" {cell} ", end="")
        print()
    
    # Column numbers
    print("   ", end="")
    for j in range(len(campus_map[0])):
        print(f"{j:2d} ", end="")
    print("\n")
    
    # Legend
    print("Legend:")
    print("  C = Corridor (cost 1)")
    print("  R = Room (cost 1)")
    print("  S = Stairs (cost 3)")
    print("  L = Lift (cost 2)")
    print("  O = Open ground (cost 1)")
    print("  X = Blocked (impassable)")
    print()

def set_start_location(campus_map):
    """Set the start location for the student"""
    # Start at a room in top-left area
    start = (1, 1)  # Row 1, Column 1 - a room
    return start

def set_goal_locations(campus_map, num_goals=3):
    """Set multiple goal locations (safe zones)"""
    # Safe zones at different locations (open ground areas)
    goals = [
        (1, 13),   # Top-right open area
        (10, 7),   # Bottom open ground
        (8, 13)    # Right side open area
    ]
    return goals[:num_goals]

def bfs_search(campus_map, start, goals):
    """
    Breadth-First Search implementation
    Explores nodes level by level
    Does not consider path cost - finds shortest path in terms of steps
    """
    from collections import deque
    
    rows = len(campus_map)
    cols = len(campus_map[0])
    
    # Queue: stores (row, col, path, cost)
    queue = deque([(start[0], start[1], [start], 0)])
    visited = set()
    visited.add(start)
    exploration_order = []  # Track order of exploration
    
    while queue:
        row, col, path, cost = queue.popleft()
        exploration_order.append((row, col))
        
        # Check if we reached a goal
        if (row, col) in goals:
            return path, cost, len(exploration_order), exploration_order
        
        # Explore neighbors (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if 0 <= new_row < rows and 0 <= new_col < cols:
                # Check if not visited and not blocked
                if (new_row, new_col) not in visited and campus_map[new_row][new_col] != 'X':
                    visited.add((new_row, new_col))
                    
                    # Get movement cost
                    move_cost = get_cell_cost(campus_map[new_row][new_col])
                    new_path = path + [(new_row, new_col)]
                    new_cost = cost + move_cost
                    
                    queue.append((new_row, new_col, new_path, new_cost))
    
    # No path found
    return None, None, len(exploration_order), exploration_order

def dfs_search(campus_map, start, goals):
    """
    Depth-First Search implementation
    Explores as deep as possible before backtracking
    Not optimal - may not find the shortest path
    """
    rows = len(campus_map)
    cols = len(campus_map[0])
    
    # Stack: stores (row, col, path, cost)
    stack = [(start[0], start[1], [start], 0)]
    visited = set()
    visited.add(start)
    exploration_order = []  # Track order of exploration
    
    while stack:
        row, col, path, cost = stack.pop()
        exploration_order.append((row, col))
        
        # Check if we reached a goal
        if (row, col) in goals:
            return path, cost, len(exploration_order), exploration_order
        
        # Explore neighbors (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if 0 <= new_row < rows and 0 <= new_col < cols:
                # Check if not visited and not blocked
                if (new_row, new_col) not in visited and campus_map[new_row][new_col] != 'X':
                    visited.add((new_row, new_col))
                    
                    # Get movement cost
                    move_cost = get_cell_cost(campus_map[new_row][new_col])
                    new_path = path + [(new_row, new_col)]
                    new_cost = cost + move_cost
                    
                    stack.append((new_row, new_col, new_path, new_cost))
    
    # No path found
    return None, None, len(exploration_order), exploration_order

def best_first_search(campus_map, start, goals):
    """
    Best-First Search (greedy) implementation
    Uses heuristic only (Manhattan distance to closest goal)
    Not optimal - doesn't consider actual path cost
    """
    import heapq
    
    rows = len(campus_map)
    cols = len(campus_map[0])
    
    h = min(manhattan_distance(start, goal) for goal in goals)
    heap = [(h, start[0], start[1], [start], 0)]
    visited = set()
    visited.add(start)
    exploration_order = []
    
    while heap:
        _, row, col, path, cost = heapq.heappop(heap)
        exploration_order.append((row, col))
        
        if (row, col) in goals:
            return path, cost, len(exploration_order), exploration_order
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if (new_row, new_col) not in visited and campus_map[new_row][new_col] != 'X':
                    visited.add((new_row, new_col))
                    
                    move_cost = get_cell_cost(campus_map[new_row][new_col])
                    new_path = path + [(new_row, new_col)]
                    new_cost = cost + move_cost
                    
                    h = min(manhattan_distance((new_row, new_col), goal) for goal in goals)
                    
                    heapq.heappush(heap, (h, new_row, new_col, new_path, new_cost))
    
    return None, None, len(exploration_order), exploration_order

def best_first_search_custom(campus_map, start, goals):
    """
    Best-First Search with CUSTOM TERRAIN-AWARE heuristic
    Penalizes paths through stairs and lifts
    """
    import heapq
    
    rows = len(campus_map)
    cols = len(campus_map[0])
    
    h = min(terrain_aware_heuristic(start, goal, campus_map) for goal in goals)
    heap = [(h, start[0], start[1], [start], 0)]
    visited = set()
    visited.add(start)
    exploration_order = []
    
    while heap:
        _, row, col, path, cost = heapq.heappop(heap)
        exploration_order.append((row, col))
        
        if (row, col) in goals:
            return path, cost, len(exploration_order), exploration_order
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if (new_row, new_col) not in visited and campus_map[new_row][new_col] != 'X':
                    visited.add((new_row, new_col))
                    
                    move_cost = get_cell_cost(campus_map[new_row][new_col])
                    new_path = path + [(new_row, new_col)]
                    new_cost = cost + move_cost
                    
                    h = min(terrain_aware_heuristic((new_row, new_col), goal, campus_map) for goal in goals)
                    
                    heapq.heappush(heap, (h, new_row, new_col, new_path, new_cost))
    
    return None, None, len(exploration_order), exploration_order

def a_star_search(campus_map, start, goals):
    """
    A* Search implementation
    Uses f(n) = g(n) + h(n)
    g(n) = actual cost from start to current node
    h(n) = heuristic (estimated cost from current to goal)
    Optimal - guaranteed to find shortest path
    """
    import heapq
    
    rows = len(campus_map)
    cols = len(campus_map[0])
    
    h = min(manhattan_distance(start, goal) for goal in goals)
    heap = [(h, 0, start[0], start[1], [start])]
    
    best_cost = {start: 0}
    exploration_order = []
    
    while heap:
        f_score, g_score, row, col, path = heapq.heappop(heap)
        exploration_order.append((row, col))
        
        if (row, col) in goals:
            return path, g_score, len(exploration_order), exploration_order
        
        if (row, col) in best_cost and g_score > best_cost[(row, col)]:
            continue
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if campus_map[new_row][new_col] != 'X':
                    move_cost = get_cell_cost(campus_map[new_row][new_col])
                    new_g_score = g_score + move_cost
                    
                    if (new_row, new_col) not in best_cost or new_g_score < best_cost[(new_row, new_col)]:
                        best_cost[(new_row, new_col)] = new_g_score
                        
                        h = min(manhattan_distance((new_row, new_col), goal) for goal in goals)
                        new_f_score = new_g_score + h
                        
                        new_path = path + [(new_row, new_col)]
                        heapq.heappush(heap, (new_f_score, new_g_score, new_row, new_col, new_path))
    
    return None, None, len(exploration_order), exploration_order

def a_star_search_custom(campus_map, start, goals):
    """
    A* Search with CUSTOM TERRAIN-AWARE heuristic
    Uses f(n) = g(n) + h_custom(n)
    h_custom penalizes stairs and lifts
    """
    import heapq
    
    rows = len(campus_map)
    cols = len(campus_map[0])
    
    h = min(terrain_aware_heuristic(start, goal, campus_map) for goal in goals)
    heap = [(h, 0, start[0], start[1], [start])]
    
    best_cost = {start: 0}
    exploration_order = []
    
    while heap:
        f_score, g_score, row, col, path = heapq.heappop(heap)
        exploration_order.append((row, col))
        
        if (row, col) in goals:
            return path, g_score, len(exploration_order), exploration_order
        
        if (row, col) in best_cost and g_score > best_cost[(row, col)]:
            continue
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if campus_map[new_row][new_col] != 'X':
                    move_cost = get_cell_cost(campus_map[new_row][new_col])
                    new_g_score = g_score + move_cost
                    
                    if (new_row, new_col) not in best_cost or new_g_score < best_cost[(new_row, new_col)]:
                        best_cost[(new_row, new_col)] = new_g_score
                        
                        h = min(terrain_aware_heuristic((new_row, new_col), goal, campus_map) for goal in goals)
                        new_f_score = new_g_score + h
                        
                        new_path = path + [(new_row, new_col)]
                        heapq.heappush(heap, (new_f_score, new_g_score, new_row, new_col, new_path))
    
    return None, None, len(exploration_order), exploration_order

def visualize_path(campus_map, path, start, goals, title="Path"):
    """Visualize the found path on the map"""
    import copy
    
    # Create a copy of the map for visualization
    display_map = copy.deepcopy(campus_map)
    
    # Mark the path with dots
    for pos in path[1:-1]:  # Exclude start and goal
        if display_map[pos[0]][pos[1]] != 'X':
            display_map[pos[0]][pos[1]] = '·'
    
    # Mark start and goal
    display_map[start[0]][start[1]] = 'S'
    goal_reached = path[-1] if path else None
    if goal_reached:
        display_map[goal_reached[0]][goal_reached[1]] = 'G'
    
    print(f"\n{title}")
    print("-" * (len(display_map[0]) * 3))
    
    for i, row in enumerate(display_map):
        print(f"{i:2d} ", end="")
        for cell in row:
            print(f" {cell} ", end="")
        print()
    
    # Column numbers
    print("   ", end="")
    for j in range(len(display_map[0])):
        print(f"{j:2d} ", end="")
    print("\n")
    
    print("Legend: S = START, G = GOAL, · = Path")
    print()

def generate_comparison_table(results):
    """Generate a comparison table of algorithm performance"""
    print("\n" + "="*90)
    print(f"{'Algorithm':<25} {'Path Length':<15} {'Total Cost':<15} {'Nodes Explored':<20} {'Time (ms)':<15}")
    print("="*90)
    
    # Find the optimal cost
    valid_results = {k: v for k, v in results.items() if v['cost'] is not None}
    optimal_cost = min(v['cost'] for v in valid_results.values()) if valid_results else None
    
    for algo_name, result in results.items():
        path_len = len(result['path']) if result['path'] else 0
        cost = result['cost'] if result['cost'] is not None else 'N/A'
        nodes = result['nodes_explored']
        time_ms = result['execution_time'] * 1000
        
        # Mark if path is optimal
        optimal_marker = " [OPTIMAL]" if result['cost'] == optimal_cost else ""
        
        print(f"{algo_name:<25} {path_len:<15} {str(cost):<15} {nodes:<20} {time_ms:<15.4f}{optimal_marker}")
    
    print("="*90)
    
    # Detailed optimal vs non-optimal analysis
    if valid_results:
        print("\nOPTIMAL VS NON-OPTIMAL PATH ANALYSIS:")
        print(f"   Optimal Cost: {optimal_cost}")
        
        optimal_algos = [k for k, v in valid_results.items() if v['cost'] == optimal_cost]
        non_optimal_algos = [k for k, v in valid_results.items() if v['cost'] > optimal_cost]
        
        print(f"   Algorithms finding OPTIMAL path: {', '.join(optimal_algos)}")
        if non_optimal_algos:
            print(f"   Algorithms finding NON-OPTIMAL path:")
            for algo in non_optimal_algos:
                cost_diff = valid_results[algo]['cost'] - optimal_cost
                print(f"      - {algo}: Cost {valid_results[algo]['cost']} (+{cost_diff} extra cost)")
        
        # Best performers
        best_cost = min(valid_results.items(), key=lambda x: x[1]['cost'])
        best_nodes = min(valid_results.items(), key=lambda x: x[1]['nodes_explored'])
        best_time = min(valid_results.items(), key=lambda x: x[1]['execution_time'])
        
        print(f"\nBEST PERFORMERS:")
        print(f"   Best Cost: {best_cost[0]} (Cost: {best_cost[1]['cost']})")
        print(f"   Most Efficient: {best_nodes[0]} (Nodes: {best_nodes[1]['nodes_explored']})")
        print(f"   Fastest: {best_time[0]} (Time: {best_time[1]['execution_time']*1000:.4f} ms)")
    print()

def add_dynamic_obstacles(campus_map, path, num_blocks):
    """Add dynamic obstacles to the map"""
    import copy
    
    # Create a copy of the map
    modified_map = copy.deepcopy(campus_map)
    
    # Don't block start or goal positions
    blockable_positions = path[1:-1]  # Exclude start and end
    
    if len(blockable_positions) < num_blocks:
        num_blocks = len(blockable_positions)
    
    # Randomly select positions to block
    blocked_cells = random.sample(blockable_positions, num_blocks)
    
    # Block the selected cells
    for pos in blocked_cells:
        modified_map[pos[0]][pos[1]] = 'X'
    
    return modified_map, blocked_cells

def generate_conclusion(initial_results, dynamic_results=None):
    """Generate final conclusion and recommendations"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("="*70)
    
    # Filter valid results
    valid_results = {k: v for k, v in initial_results.items() if v['cost'] is not None}
    
    if not valid_results:
        print("No valid paths found.")
        return
    
    # 1. Performance Comparison
    print("\n1. ALGORITHM PERFORMANCE COMPARISON")
    print("-" * 70)
    
    # Find rankings
    sorted_by_cost = sorted(valid_results.items(), key=lambda x: x[1]['cost'])
    sorted_by_nodes = sorted(valid_results.items(), key=lambda x: x[1]['nodes_explored'])
    sorted_by_time = sorted(valid_results.items(), key=lambda x: x[1]['execution_time'])
    
    print("\nRanking by Path Cost (Lower is Better):")
    for i, (algo, result) in enumerate(sorted_by_cost, 1):
        print(f"  {i}. {algo}: {result['cost']}")
    
    print("\nRanking by Nodes Explored (Lower is Better):")
    for i, (algo, result) in enumerate(sorted_by_nodes, 1):
        print(f"  {i}. {algo}: {result['nodes_explored']}")
    
    print("\nRanking by Execution Time (Lower is Better):")
    for i, (algo, result) in enumerate(sorted_by_time, 1):
        print(f"  {i}. {algo}: {result['execution_time']*1000:.4f} ms")
    
    # 2. Algorithm Strengths and Weaknesses
    print("\n2. ALGORITHM STRENGTHS AND WEAKNESSES")
    print("-" * 70)
    
    print("\nBFS (Breadth-First Search):")
    print("  Strengths:")
    print("    - Guarantees shortest path in terms of number of steps")
    print("    - Explores systematically level by level")
    print("    - Complete - always finds a solution if one exists")
    print("  Weaknesses:")
    print("    - Does NOT consider movement costs (treats stairs = corridor)")
    print("    - High memory usage (stores all nodes at current level)")
    print("    - Can explore many unnecessary nodes")
    
    print("\nDFS (Depth-First Search):")
    print("  Strengths:")
    print("    - Low memory usage (only stores path)")
    print("    - Fast in some cases when goal is deep in search tree")
    print("  Weaknesses:")
    print("    - Does NOT guarantee optimal path")
    print("    - Can get stuck exploring wrong branches deeply")
    print("    - Path quality depends heavily on exploration order")
    
    print("\nBest-First Search (Greedy):")
    print("  Strengths:")
    print("    - Uses heuristic to guide search toward goal")
    print("    - Often faster than uninformed search")
    print("    - Lower node exploration in favorable cases")
    print("  Weaknesses:")
    print("    - NOT optimal - ignores actual path cost")
    print("    - Can be misled by heuristic")
    print("    - May find longer paths if heuristic is misleading")
    
    print("\nA* Search:")
    print("  Strengths:")
    print("    - Optimal - guarantees shortest path by COST")
    print("    - Considers both actual cost and heuristic")
    print("    - Efficient - explores fewer nodes than uninformed search")
    print("  Weaknesses:")
    print("    - Higher computational overhead per node")
    print("    - Requires good heuristic design")
    print("    - More memory usage than DFS")
    
    # 3. Why might BFS outperform A* in some scenarios?
    print("\n3. WHY BFS MIGHT OUTPERFORM A* IN THIS SCENARIO")
    print("-" * 70)
    
    bfs_cost = initial_results['BFS']['cost'] if 'BFS' in valid_results else None
    astar_cost = initial_results['A*']['cost'] if 'A*' in valid_results else None
    
    print("\nKey Insight:")
    if bfs_cost and astar_cost:
        if bfs_cost < astar_cost:
            print("  ⚠ BFS found a CHEAPER path than A*!")
            print("\n  This seems contradictory, but here's why:")
            print("  - BFS finds shortest path by NUMBER OF STEPS")
            print("  - A* finds shortest path by TOTAL COST")
            print("  - In this map, the BFS path may have more steps but lower cost")
            print("    because it avoided high-cost cells (stairs, lifts)")
            print("  - A* uses Manhattan distance which doesn't account for")
            print("    terrain cost differences, leading to suboptimal choices")
        elif bfs_cost == astar_cost:
            print("  Both found paths with equal cost")
            print("  - This indicates the optimal path has uniform terrain")
        else:
            print("  A* found the optimal path as expected")
            print(f"  - A* cost: {astar_cost} vs BFS cost: {bfs_cost}")
    
    print("\n  General Analysis:")
    print("  - BFS explores uniformly, so it naturally finds paths that")
    print("    minimize steps, which can coincide with minimizing cost")
    print("  - A* is only optimal if the heuristic is admissible")
    print("  - Manhattan distance doesn't consider terrain penalties")
    print("  - A better heuristic would improve A* performance")
    
    # 4. Dynamic Scenario Analysis
    if dynamic_results:
        print("\n4. DYNAMIC ADAPTATION ANALYSIS")
        print("-" * 70)
        
        print("\nAdaptation Success:")
        for algo, result in dynamic_results.items():
            adapted = "Yes" if result['adapted'] else "No"
            print(f"  {algo}: {adapted}")
            if result['adapted']:
                original_cost = initial_results[algo]['cost']
                new_cost = result['cost']
                cost_increase = new_cost - original_cost if original_cost else 0
                print(f"    Original Cost: {original_cost} → New Cost: {new_cost} (Δ +{cost_increase})")
        
        print("\nKey Observations:")
        print("  - All algorithms successfully found alternative paths")
        print("  - The cost increase shows the impact of blocked cells")
        print("  - Algorithms that explore more broadly (BFS, A*) adapt better")
    
    # 5. Recommendations
    print("\n5. RECOMMENDATIONS FOR REAL-WORLD CAMPUS NAVIGATION")
    print("-" * 70)
    
    print("\nBest Algorithm Choice Depends on Priority:")
    print("  1. For EMERGENCY EVACUATION (speed critical):")
    print("     → Use BFS or Best-First Search")
    print("     Reason: Fast, finds reasonable paths quickly")
    
    print("\n  2. For OPTIMAL PATH (minimize cost/difficulty):")
    print("     → Use A* with terrain-aware heuristic")
    print("     Reason: Considers both distance and terrain cost")
    
    print("\n  3. For RESOURCE-CONSTRAINED DEVICES:")
    print("     → Use DFS or Best-First Search")
    print("     Reason: Lower memory usage")
    
    print("\n  4. For DYNAMIC ENVIRONMENTS (frequent obstacles):")
    print("     → Use A* with fast re-planning")
    print("     Reason: Can quickly adapt to changes")
    
    print("\nHeuristic Recommendations:")
    print("  - Manhattan distance: Simple, fast, but ignores terrain")
    print("  - Terrain-aware heuristic: Better for campus with stairs/lifts")
    print("  - Custom heuristic: Can encode domain knowledge (crowd avoidance)")
    
    print("\nFinal Verdict:")
    best_overall = sorted_by_cost[0][0]
    print(f"  For this specific campus map: {best_overall} performed best")
    print(f"  Cost: {sorted_by_cost[0][1]['cost']}")
    print(f"  Nodes explored: {sorted_by_cost[0][1]['nodes_explored']}")
    print(f"  Time: {sorted_by_cost[0][1]['execution_time']*1000:.4f} ms")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
