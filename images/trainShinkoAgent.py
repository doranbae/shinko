# +--------------------------+
# |   Play Shinko with RL    |
# | Train Shinko Agent       |
# +--------------------------+

from playShinko_rl import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer
from datetime import datetime

# define variables for the game
matrix_width       = 5
level_num          = 2
flat_matrix_length = matrix_width * level_num

# define variables for the model
num_episodes = 500

def qtable_nn_model():
    model = Sequential()
    model.add( InputLayer( batch_input_shape = ( 3, flat_matrix_length ) ) )
    model.add( Dense( 20, activation= 'sigmoid' ) )
    model.add( Dense( 10, activation= 'sigmoid' ) )
    model.add( Dense( flat_matrix_length, activation = 'linear' ) )

    model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

    return model

model = qtable_nn_model()


# # training with 1 set of environment
# print( 'Training Shinko agent...' )
# start = datetime.now()
# seed = random.randint(1000,2000)
# for i in range( num_episodes ):
#     shinko = Play()
#     trained_model = shinko.execute_training(model, random_seed=seed)
# end = datetime.now()
# print( 'Train completed. Time taken: {}'.format( end - start ) )

# training with multiple set of environment
print( 'Training Shinko agent...' )
start = datetime.now()
for x in range(20):
    if x % 2 == 0:
        print( 'Loading {}th game set'.format( x ) )
    seed = random.randint(1000,2000)
    for i in range( num_episodes ):
        shinko = Play()
        trained_model = shinko.execute_training(model, random_seed=seed)

end = datetime.now()
print( 'Train completed. Time taken: {}'.format( end - start ) )

# save the model
model_json = trained_model.to_json()
with open( 'model_20.json', 'w') as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
trained_model.save_weights("model_20.h5")
print( "Saved model to disk" )

