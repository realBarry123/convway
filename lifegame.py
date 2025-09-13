import copy
def _count_neighbours(universe, row, col):
    neighbours = 0
    
    for drow in [-1, 0, 1]:
        for dcol in [-1, 0, 1]:
            if drow == 0 and dcol == 0:
                continue
            if 0 <= row+drow < len(universe) and 0 <= col+dcol < len(universe[0]):
                neighbours += universe[row+drow][col+dcol]

    return neighbours

def _print_matrix(universe):
    for row in universe:
        for cell in row:
            if cell == 1:
                print("#", end="")
            else:
                print(".", end="")
        print("\n")

def update(universe):
    new_universe = copy.deepcopy(universe)
    for row in range(len(universe)):
        for col in range(len(universe[row])):
            neighbours = _count_neighbours(universe, row, col)
            if universe[row][col] == 1 and (neighbours < 2 or neighbours > 3):
                new_universe[row][col] = 0
		    
            elif universe[row][col] == 0 and neighbours == 3:
                new_universe[row][col] = 1

    return new_universe
