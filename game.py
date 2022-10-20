

# TODO:
# - max score == 2^17 on 4x4 grid, scale for ML
import time
import numpy as np

class TwosGame:
    def __init__(self):
        self.dim = 4
        self.reward = 0
        self._board = np.zeros((self.dim, self.dim), dtype=int)
        self.add_tiles([1, 2, 3], probs=[0.75, 0.20, 0.05], num=2)    

    def reset(self):
        self.__init__()    

    
    def add_tiles(self, tile_opts, probs=None, num=1):
        num_added = 0
        xyrange = np.arange(self.dim)
        while num_added < num:
            x = np.random.choice(xyrange)
            y = np.random.choice(xyrange)
            tile = np.random.choice(tile_opts, p=probs)
            if self._board[x][y] == 0:
                self._board[x][y] = tile
                num_added += 1


    def env_state(self):
        return self._board


    def remove_zeros(self, array):
        new_array = array[array != 0].reshape((1, -1))
        return new_array


    def moveAndCombine(self, old, padLeftSide):
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
    

    def augment_board(self, direction):
        vals = {0: (0, True), 1: (1, True), 2: (0, False), 3:(1, False)}
        (axis, padLeftSide) = vals[direction]
        # moving zeros and combining
        updated_board = np.apply_along_axis(self.moveAndCombine, axis, self._board, padLeftSide=padLeftSide)
        return updated_board
    

    def move(self, direction):
        # directions: 0 = down, 1 = right, 2 = up, 3 = left
        new_board = self.augment_board(direction)
        # if boards equal move not allowed -> dont add new tile
        if not np.equal(new_board, self._board).all():
            # add new tile
            self._board = new_board
            self.add_tiles([1, 2], probs=[0.9, 0.1])
            self.reward += 1
        
        
    def done(self):
        full_board = not (self._board == 0).any()
        game_over = True
        if game_over:
            for i in range(4):
                if not (np.equal(self._board, self.augment_board(i))).all():
                    game_over = False
        return game_over


    def end(self):
        print("Game Over!")
        print(game)
        exit()


    def __str__(self):
        printstring = f'\nscore: {self.reward}\n'
        maxlen = len(str(2 ** np.max(self._board)))
        printstring += '-' * (self.dim * maxlen + self.dim + 1) + '\n'
        for i in range(self.dim):
            printstring += '|'
            for j in range(self.dim):
                print(self._board[i][j], 2 ** self._board[i][j])
                tile = 2 ** self._board[i][j]
                printstring += f'{tile:{maxlen}}|'
            printstring += '\n' + '-' * (self.dim * maxlen + self.dim + 1) + '\n'
        return printstring


    def play(self):
        while not self.done():
            #print()
            #print(self)
            direction = np.random.choice([0, 1, 2, 3,]) #
            #direction = int(input("Direction:"))
            #print('direction:', direction)
            self.move(direction)        
        #self.end()

    
    def get_ml_board(self):
        return self._board / 17


if __name__ == "__main__":
    game = TwosGame()
    game.play()
    print(game)
    

