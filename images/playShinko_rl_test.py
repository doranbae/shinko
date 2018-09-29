# +--------------------------+
# |   Play Shinko with RL    |
# | Test using random sets   |
# +--------------------------+

from keras.models import *
#
# load trained model by loading json and creating model
json_file = open('model_20.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_20.h5")
print("Loaded model from disk")


# define variables
matrix_min         = 1
matrix_max         = 5
matrix_width       = 5
level_num          = 2
shinko_goal        = 5
handicap           = 1
nox_pre_watch_size = 3
flat_matrix_length = matrix_width * level_num

np.random.seed(84)

class TestPlay:
    def __init__(self):
        self.game_over = False
        self.matrix = np.random.randint( low  = matrix_min,
                                         high = matrix_max,
                                         size = (level_num, matrix_width) )
        self.valid_actions = np.arange( start = flat_matrix_length/2,
                                        stop  = flat_matrix_length,
                                        step  = 1,
                                        dtype = 'int')
        initial_nox_list = np.random.randint( low  = matrix_min,
                                      high = matrix_max-handicap,
                                      size = (nox_pre_watch_size) )
        # print('initial nox list: ',initial_nox_list)
        self.nox_list = []
        self.nox_list.extend( initial_nox_list )

        self.score  = 0
        self.reward = 0
        self.num_moves = 0
        self.game_win = False

    def get_state( self, flat_state, nox_1, nox_2, nox_3 ):
        """
        :param flat_state: current matrix flattened
        :param nox_1, nox_2, nox_3: model anticipates future noxes up to 3
        :return: rendered flat matrix based on the future noxes
        """
        first_curr_state  = shinko_goal - flat_state - nox_1
        second_curr_state = shinko_goal - flat_state - nox_2
        third_curr_state  = shinko_goal - flat_state - nox_3
        state = np.hstack( (first_curr_state, second_curr_state, third_curr_state) )
        state = state.reshape(3, -1)
        return state

    def gen_nox(self):
        nox_list = np.random.randint( low  = matrix_min,
                                      high = matrix_max-handicap,
                                      size = (1) )
        return nox_list

    def find_best_action(self, flat_matrix, nox):
        valid_pairs = []
        for valid_action in self.valid_actions:
            remainder             = shinko_goal - flat_matrix[0][valid_action] - nox
            action_remainder_pair = ( valid_action,  remainder  )
            valid_pairs.append(action_remainder_pair)
        sorted_valid_pairs  = sorted(valid_pairs, key = lambda tup: tup[1], reverse = False)
        self.valid_action_ranking = [pos for (pos, remainder) in sorted_valid_pairs if remainder >= 0]
        return self.valid_action_ranking

    def update_valid_actions(self, action):
        """
        delete the pos that made 5 (<-- becomes unplayable)
        add the new pos (<-- became playable)
        :return: updated valid actions
        """
        curr_valid_actions = self.valid_actions

        # delete the index of the flat matrix which has a value of 5
        # (--> not possible to play anymore)
        del_pos = np.where( curr_valid_actions == action)[0][0]
        updated_valid_actions = np.delete( curr_valid_actions, del_pos )

        # add new index of the flat matrix which became available to play
        # (--> for example, when one of the box in matrix disappears,
        # the number underneath becomes playable)
        new_valid_action = action - matrix_width
        if new_valid_action >= 0:
            updated_valid_actions = np.append(updated_valid_actions , new_valid_action )
        self.valid_actions = updated_valid_actions

    def predict_action(self, model, curr_state, flat_matrix, nox):
        predicted_actions = model.predict( curr_state )[0]
        valid_pos = self.find_best_action( flat_matrix, nox )
        valid_moves     = np.array( [ x if idx in valid_pos else np.NaN for idx, x in enumerate( predicted_actions ) ])
        if np.isnan( valid_moves ).all():
            print( '' )
            print( '##################' )
            print( '    GAME OVER' )
            print( '##################' )
            print( ' No more moves_____' )
            print( flat_matrix.reshape(-1, matrix_width) )
            print( ' nox: ', nox )
            print( ' Total moves: ', self.num_moves )
            self.game_over = True
            return -1
        else:
            action = np.nanargmax(valid_moves)
        return action

    def update_matrix(self, flat_matrix, action, nox):
        """
        based on action, nox, and current flat matrix, update for the next state
        """
        new_score = flat_matrix[0][ action ]  + nox
        flat_matrix[ 0, action ] = new_score
        if new_score == 5:
            self.score  += 1
            self.reward += 1
            # if action resulted in 5, update the self.update_valid_actions
            self.update_valid_actions( action )

        self.matrix = flat_matrix.reshape( -1, matrix_width)

    def play(self):
        # initialize the game
        print( 'Initializing the matrix________' )
        print( self.matrix.reshape(-1, matrix_width) )
        print( '-------------------------------' )

        while self.game_over == False:
            flat_matrix = self.matrix.reshape( -1, flat_matrix_length )
            print( '' )
            print( '( Num moves: {} ) Beat the game when next noxes are: {}'.format( self.num_moves, self.nox_list ))
            print( ' ' )
            print( flat_matrix.reshape(-1, matrix_width) )
            print( ' ' )
            # calculate the best move
            # render flat matrix into 3 sets of flat matrix by 3 future noxes
            curr_state = self.get_state(flat_matrix, self.nox_list[0], self.nox_list[1], self.nox_list[2])

            # predict action
            action = self.predict_action( loaded_model, curr_state, flat_matrix, self.nox_list[0] )
            if action == -1:
                # GAME OVER
                break
            # make a move
            print( 'Shinko agent chooses: ', action )
            # update move cnt
            self.num_moves += 1

            # update matrix based on action
            self.update_matrix( flat_matrix, action, self.nox_list[0] )

            # update nox_list
            self.nox_list.extend( list(self.gen_nox()) )
            self.nox_list = self.nox_list[1: ]

            # print the result if won
            if np.all( self.matrix == 5 ):
                print( '' )
                print( '##################' )
                print( '     YOU WIN' )
                print( '##################' )
                print( 'Total num of moves: ', self.num_moves )
                print( flat_matrix.reshape( -1, matrix_width) )

                self.game_over = True
                self.game_win = True
            print('~*~*~*~*~*~*~*~*~*~*~*~*~*~*')
            print( '' )

        return self.game_win

# shinko = TestPlay()
# shinko.play()

win_cnt = 0
win_idx = []
for x in range(300):
    np.random.seed(x)
    shinko = TestPlay()
    win = shinko.play()

    if win == True:
        win_cnt += 1
        win_idx.append(x)

print( 'Testing done' )
print( 'Win count: {}'.format(win_cnt) )