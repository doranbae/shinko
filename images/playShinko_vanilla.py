# +--------------------------+
# |         SHINKO           |
# | Play the mobile game     |
# +--------------------------+

import numpy as np

# set random seed
np.random.seed(84)

# define variables
matrix_min         = 1
matrix_max         = 5
matrix_width       = 5
level_num          = 2
shinko_goal        = 5
handicap           = 1
nox_pre_watch_size = 3
flat_matrix_length = matrix_width * level_num

class Play:
    def __init__(self):
        """
        :self.game_over    : Boolean value to determine whether game is over or not
        :self.reward       : Every time you successfully make an addition 5, you gain a reward
        :self.num_moves    : Number of noxes you use to play the game. Given you beat the game,
                             you want to use the smallest possible number of noxes
        :self.matrix       : Initial set of boxes you need to play on
        :self.flat_matrix  : Flattened form of the matrix
        :self.valid_actions: Simplifying feature not to allow the splitting of the matrix
        :self.init_nox_list: At the start of each game, you see next noxes up to 3
        """
        self.game_over = False
        self.reward    = 0
        self.num_moves = 0

        self.matrix = np.random.randint( low  = matrix_min,
                                         high = matrix_max,
                                         size = (level_num, matrix_width) )
        self.flat_matrix = self.matrix.reshape( -1, flat_matrix_length )
        self.valid_actions = np.arange( start = flat_matrix_length/2,
                                        stop  = flat_matrix_length,
                                        step  = 1,
                                        dtype = 'int')

        self.init_nox_list = np.random.randint( low  = matrix_min,
                                                high = matrix_max-handicap,
                                                size = (nox_pre_watch_size) )
        self.win_game = False

    def gen_nox(self):
        """
        You will be abel to see future noxes up to 3.
        :return: new additional nox
        """
        new_nox = np.random.randint( low  = matrix_min,
                                      high = matrix_max-handicap,
                                      size = (1) )
        return new_nox

    def find_best_action(self, nox):
        """
        :return: updates self.valid_action_ranking, a list of actions
                 in the order of most reward given
        """
        valid_pairs = []
        for valid_action in self.valid_actions:
            remainder             = shinko_goal - self.flat_matrix[0][valid_action] - nox
            action_remainder_pair = ( valid_action,  remainder  )
            valid_pairs.append(action_remainder_pair)
        sorted_valid_pairs        = sorted(valid_pairs, key = lambda tup: tup[1], reverse = False)
        self.valid_action_ranking = [pos for (pos, remainder) in sorted_valid_pairs if remainder >= 0]

    def update_valid_actions(self, action):
        """
        :return: Update self.valid_actions which keeps track of
                 list of list of playable matrix indices
        """
        curr_valid_actions = self.valid_actions

        # delete the index of the flat matrix which has a value of 5
        # (--> not possible to play anymore)
        del_idx = np.where( curr_valid_actions == action )[0][0]
        updated_valid_actions = np.delete( curr_valid_actions, del_idx )

        # add new index of the flat matrix which became available to play
        # (--> for example, when one of the box in matrix disappears,
        # the number underneath becomes playable)
        new_valid_action = action - matrix_width
        if new_valid_action >= 0:
            updated_valid_actions = np.append( updated_valid_actions, new_valid_action )
        self.valid_actions = updated_valid_actions

    def update_matrix(self, nox):
        """
        Updates the matrix based on nox and the best action
        Updates reward and num_moves
        Updates self.valid_actions (based on the current action)
        """
        if len( self.valid_action_ranking ) == 0:
            print( '' )
            print( '##################' )
            print( '     GAME OVER' )
            print( '##################' )
            print( ' No more moves_____' )
            print( self.flat_matrix.reshape(-1, matrix_width) )
            print( ' nox: ', nox )
            print( ' Total moves: ', self.num_moves )
            self.game_over = True

        else:
            # best action is the first action from self.valid_action_ranking
            best_action = self.valid_action_ranking[0]
            print( 'Your action: ', best_action )
            self.num_moves += 1
            self.flat_matrix[ 0, best_action ] = self.flat_matrix[0][ best_action ] + nox
            if self.flat_matrix[ 0, best_action ] == 5:
                self.reward    += 1
                self.update_valid_actions( best_action )

    def startGame(self):
        nox_list = []
        nox_list.extend( self.init_nox_list )

        print( 'Initializing the matrix________' )
        print( self.flat_matrix.reshape(-1, matrix_width) )
        print( '-------------------------------' )

        while self.game_over == False:
            print( '( Num moves: {} ) Beat the game when next noxes are: {}'.format( self.num_moves, nox_list ))
            print( self.flat_matrix.reshape(-1, matrix_width) )

            # rank the valid options by reward in self.valid_action_ranking
            nox = nox_list[0]
            self.find_best_action(nox)

            # the machine will always play the best move based on a simple arithmetic rule
            self.update_matrix(nox)

            # update nox_list
            nox_list.extend( list(self.gen_nox()) )
            nox_list = nox_list[1: ]

            if np.all( self.flat_matrix == 5):
                print( '' )
                print( '##################' )
                print( '     YOU WIN' )
                print( '##################' )
                # print( 'Final score: ', self.reward * ( flat_matrix_length /self.num_moves) )
                print( 'Total num of moves: ', self.num_moves )
                self.game_over = True
                self.win_game = True
            print('~*~*~*~*~*~*~*~*~*~*~*~*~*~*')

        return self.win_game


# if __name__ == "__main__":
#     print( 'Lets play Shinko!' )
#     shinko = Play()
#     shinko.startGame()

win_cnt = 0
win_idx = []
for x in range(300):
    np.random.seed(x)
    shinko = Play()
    win = shinko.startGame()

    if win == True:
        win_cnt += 1
        win_idx.append(x)

print( 'Testing done' )
print( 'Win count: {}'.format(win_cnt) )