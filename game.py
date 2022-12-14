

import numpy as np

class Game:
    def __init__(self, dim=4):
        self.dim = dim
        self.reset()

    def reset(self):
        self.total_reward = 0
        self.state = np.zeros((self.dim, self.dim), dtype=int)
        self.add_tiles([1, 2, 3], probs=[0.75, 0.20, 0.05], num=2)

    
    def add_tiles(self, tile_opts, probs=None, num=1):
        num_added = 0
        xyrange = np.arange(self.dim)
        while num_added < num:
            x = np.random.choice(xyrange)
            y = np.random.choice(xyrange)
            tile = np.random.choice(tile_opts, p=probs)
            if self.state[x][y] == 0:
                self.state[x][y] = tile
                num_added += 1


    def remove_zeros(self, array):
        new_array = array[array != 0].reshape((1, -1))
        return new_array


    def move_and_combine(self, old, padLeftSide):
        no_zero_tiles = self.remove_zeros(old)
        iteration = range(0, no_zero_tiles.size - 1, 1)
        if padLeftSide: no_zero_tiles = np.flip(no_zero_tiles)
        # merge tiles
        for i in iteration:
            if no_zero_tiles[0][i] == no_zero_tiles[0][i+1]:
                no_zero_tiles[0][i] += 1
                no_zero_tiles[0][i+1] = 0
                prevMerged = True
            else:
                prevMerged = False
        # putting back into array of zeros
        new = np.zeros((1, self.dim), dtype=int)
        no_zero_tiles = self.remove_zeros(no_zero_tiles)
        if padLeftSide:
            new[0][(self.dim - len(no_zero_tiles[0])):] = np.flip(no_zero_tiles)
        else:
            new[0][:len(no_zero_tiles[0])] = no_zero_tiles
        
        arr = new.reshape(self.dim,)
        return arr
    

    def update_state(self, action):
        vals = {0: (0, True), 1: (1, True), 2: (0, False), 3:(1, False)}
        (axis, padLeftSide) = vals[action]
        # moving zeros and combining
        updated_state = np.apply_along_axis(self.move_and_combine, axis, self.state, padLeftSide=padLeftSide)
        return updated_state
    

    def step(self, direction):
        # directions: 0 = down, 1 = right, 2 = up, 3 = left
        next_state = self.update_state(direction)
        # if boards equal move not allowed -> dont add new tile
        if not np.equal(next_state, self.state).all():
            # add new tile
            self.state = next_state
            self.add_tiles([1, 2], probs=[0.9, 0.1])
            self.total_reward += 1
            i_reward = 1
        else:
            self.total_reward -= 0.5
            i_reward = -0.5
        x = self.is_done() 
        if x:
            i_reward = -3
        return self.state / 17, i_reward, x
        
        
    def is_done(self):
        done = True
        for i in range(self.dim):
            if not (np.equal(self.state, self.update_state(i))).all():
                done = False
        return done


    def __str__(self):
        printstring = f'\nscore: {self.total_reward}\n'
        maxlen = len(str(2 ** np.max(self.state)))
        printstring += '-' * (self.dim * maxlen + self.dim + 1) + '\n'
        for i in range(self.dim):
            printstring += '|'
            for j in range(self.dim):
                tile = 2 ** self.state[i][j] if self.state[i][j] != 0 else self.state[i][j]
                printstring += f'{tile:{maxlen}}|'
            printstring += '\n' + '-' * (self.dim * maxlen + self.dim + 1) + '\n'
        return printstring


    def play(self):
        while not self.is_done():
            print()
            print(self)
            action = np.random.choice([0, 1, 2, 3]) 
            action = int(input("Action:"))
            self.step(action) 
        print("Game Over!")
        print(self)


if __name__ == "__main__":
    env = Game()
    env.play()
