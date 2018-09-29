# +--------------------------+
# |         SHINKO + RL      |
# | Play the mobile game     |
# +--------------------------+
# training module for Shinko game

import numpy as np
import random

# set random seed
np.random.seed(402)

# define variables
matrix_min         = 1
matrix_max         = 5
matrix_width       = 5
level_num          = 2
shinko_goal        = 5
handicap           = 1
nox_pre_watch_size = 3
flat_matrix_length = matrix_width * level_num

# define variables for the model
alpha        = 0.95
epsilon      = 0.4
gamma        = 0.999

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

    def gen_nox(self):
        """
        you will be abel to see future noxes up to 3.
        :return: new additional nox
        """
        new_nox = np.random.randint( low  = matrix_min,
                                      high = matrix_max-handicap,
                                      size = (1) )
        return new_nox


    def find_best_action(self, flat_matrix, nox):
        """
        :return: updates and returns self.valid_action_ranking, a list of actions
                 in the order of most reward given
        """
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

    def get_state(self, flat_state, nox_1, nox_2, nox_3):
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

    def update_next_state(self, nox, flat_curr_state, action):
        """
        based on action, nox, and current flat matrix, update for the next state
        :return: next flat state, reward (if any)
        """
        flat_curr_state[ 0, action ] = flat_curr_state[0][action] + nox
        r = 0
        if flat_curr_state[ 0, action ] == 5:
            r = 1
            # if action resulted in 5, update the self.update_valid_actions
            self.update_valid_actions( action )
        return flat_curr_state, r

    # def execute_training(self, model, random_seed = 10000):
    #     # initialize the nox list
    #     np.random.seed(random_seed)
    #     self.nox_list = []
    #     self.nox_list.extend( self.init_nox_list )
    #     flat_curr_state = self.flat_matrix
    #
    #     #print( 'Initializing the matrix________' )
    #     #print( flat_curr_state.reshape(-1, matrix_width) )
    #     #print( '-------------------------------' )
    #
    #     while self.game_over == False:
    #         # unlike from the vanilla model, self.find_best_actions will be used to
    #         # define a pool of possible candidates for actions
    #         valid_pos        = self.find_best_action( flat_curr_state, self.nox_list[0])
    #
    #         # explore or exploit
    #         if random.uniform(0,1) < epsilon:
    #             if len(valid_pos) == 0:
    #                 # GAME OVER
    #                 self.game_over = True
    #                 break
    #             else:
    #                 action = random.choice(valid_pos)
    #                 self.num_moves += 1
    #                 curr_state = self.get_state( flat_curr_state, self.nox_list[0], self.nox_list[1], self.nox_list[2] )
    #
    #         else:
    #             curr_state = self.get_state( flat_curr_state, self.nox_list[0], self.nox_list[1], self.nox_list[2] )
    #             predict_all_actions = model.predict( curr_state )
    #
    #             # since I am passing 3 sets of flat_matfix, the returned values are
    #             # 3 sets of expected values from each action.
    #             # only choose the first set; hence, predict_all_actions[0]
    #             # from this set of actions, I look up the valid_pos to see whether the move(=action) is playable
    #             valid_moves = np.array( [ x if idx in valid_pos else np.NaN for idx, x in enumerate( predict_all_actions[0] ) ])
    #
    #             # if no move (=action) is playable, game over.
    #             if np.isnan( valid_moves ).all():
    #                 # GAME OVER
    #                 self.game_over = True
    #                 break
    #             else:
    #                 # choose the action based on the highest expected return
    #                 action = np.nanargmax(valid_moves)
    #                 self.num_moves += 1
    #
    #         # based on the action the model just chose, make next_state
    #         flat_next_state, r     = self.update_next_state(self.nox_list[0], flat_curr_state, action )
    #
    #         if np.all( flat_next_state == 5 ):
    #             # print( '' )
    #             # print( '##################' )
    #             # print( '     YOU WIN' )
    #             # print( '##################' )
    #             # print( 'Total num of moves: ', self.num_moves )
    #             self.game_over = True
    #             break
    #
    #         self.nox_list.extend( list(self.gen_nox()) )
    #         self.nox_list = self.nox_list[1: ]
    #
    #         # calculate future reward
    #         next_state             = self.get_state( flat_next_state, self.nox_list[0], self.nox_list[1], self.nox_list[2] )
    #         predict_all_next_actions = model.predict( next_state )
    #         valid_next_pos          = self.find_best_action(flat_next_state, self.nox_list[0])
    #         if len(valid_next_pos)  == 0:
    #             # GAME OVER
    #             self.game_over = True
    #             # future expected reward is 0
    #             transformed_target_vec  = np.zeros( shape = (3, 10) )
    #         else:
    #             # since I am passing 3 sets of next flat matfix, the returned values are
    #             # 3 sets of expected values from each action.
    #             # only choose the first set; hence, predict_all_next_actions[0]
    #             # from this set of actions, I look up the valid_next_pos to see whether the move(=action) is playable
    #             valid_next_actions     = [ x if idx in valid_next_pos else np.NaN for idx, x in enumerate( predict_all_next_actions[0] )]
    #             target                 =  r + alpha * np.nanmax( valid_next_actions )
    #             target_vec             = model.predict( curr_state )[0]
    #             target_vec[action]     = target
    #             dummy                  = np.zeros( shape = (1,10) )
    #             transformed_target_vec = np.hstack( (target_vec.reshape(-1,10), dummy, dummy) )
    #
    #         model.fit( curr_state, transformed_target_vec.reshape(-1,10), epochs = 10, verbose = 0 )
    #         flat_curr_state = flat_next_state
    #
    #     return model

    def execute_training(self, model, random_seed = 10000):
        # initialize the nox list
        np.random.seed(random_seed)
        self.nox_list = []
        self.nox_list.extend( self.init_nox_list )
        flat_curr_state = self.flat_matrix

        #print( 'Initializing the matrix________' )
        #print( flat_curr_state.reshape(-1, matrix_width) )
        #print( '-------------------------------' )

        while self.game_over == False:
            # unlike the vanilla model, self.find_best_actions will be used to
            # define a candidate of possible actions for the current state
            valid_pos        = self.find_best_action( flat_curr_state, self.nox_list[0])

            # get current_state using the next 3 noxes, this is for later to calculate future rewards
            curr_state = self.get_state( flat_curr_state, self.nox_list[0], self.nox_list[1], self.nox_list[2] )

            # if valid_pos is empty (--> no playable action), then the game is over
            if len(valid_pos) == 0:
                self.game_over = True
                # need to output 0 rewards to feed into model.fit
                transformed_target_vec  = np.zeros( shape = (3, 10) )
                flat_next_state = None
            else:
                # go on with the game
                # explore or exploit
                self.num_moves += 1
                if random.uniform(0,1) < epsilon:
                    # explore --> choose any action at random
                    # action = random.choice( valid_pos )
                    action = valid_pos[0]
                else:
                    # exploit --> choose the action with the highest predicted reward
                    predict_all_actions = model.predict( curr_state )

                    # since I am passing 3 sets of flat_matfix, the returned values are
                    # 3 sets of expected values from each action.
                    # only choose the first set; hence, predict_all_actions[0]
                    # from this set of actions, I look up the valid_pos to see whether the move(=action) is playable
                    valid_moves = np.array( [ x if idx in valid_pos else np.NaN for idx, x in enumerate( predict_all_actions[0] ) ])

                    # choose the action with the highest predicted reward
                    action = np.nanargmax(valid_moves)

                # based on the action the model just chose, make next_state
                flat_next_state, r     = self.update_next_state(self.nox_list[0], flat_curr_state, action )

                if np.all( flat_next_state == 5 ):
                    # print( '' )
                    # print( '##################' )
                    # print( '     YOU WIN' )
                    # print( '##################' )
                    # print( 'Total num of moves: ', self.num_moves )
                    self.game_over = True

                # update nox list
                self.nox_list.extend( list(self.gen_nox()) )
                self.nox_list = self.nox_list[1: ]

                # calculate future reward
                next_state               = self.get_state( flat_next_state, self.nox_list[0], self.nox_list[1], self.nox_list[2] )
                predict_all_next_actions = model.predict( next_state )
                valid_next_pos           = self.find_best_action(flat_next_state, self.nox_list[0])
                if len(valid_next_pos)  == 0:
                    # GAME OVER
                    self.game_over = True

                    # future expected reward is 0
                    transformed_target_vec  = np.zeros( shape = (3, 10) )
                else:
                    # since I am passing 3 sets of next flat matfix, the returned values are
                    # 3 sets of expected values from each action.
                    # only choose the first set; hence, predict_all_next_actions[0]
                    # from this set of actions, I look up the valid_next_pos to see whether the move(=action) is playable
                    valid_next_actions     = [ x if idx in valid_next_pos else np.NaN for idx, x in enumerate( predict_all_next_actions[0] )]
                    target                 =  r + alpha * np.nanmax( valid_next_actions )
                    target_vec             = model.predict( curr_state )[0]
                    target_vec[action]     = target
                    dummy                  = np.zeros( shape = (1,10) )
                    transformed_target_vec = np.hstack( (target_vec.reshape(-1,10), dummy, dummy) )

            model.fit( curr_state, transformed_target_vec.reshape(-1,10), epochs = 10, verbose = 0 )
            flat_curr_state = flat_next_state

        return model