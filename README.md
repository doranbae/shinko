# Playing Shinko with reinforcement learning
Shinko is a mobile puzzle game where you need to make added sum to 5 using blocks given to you. 
![shinko](images/shinko_google_store.png)

Shinko has few specific features.

1. At the start of each game, you are given rows of boxes with random numbers drawn from 1 to 4. I will call them matrix. 
2. You are also given another set of boxes (at length of 3). I will call them noxes (Number BOXES). You can use these noxes to make addition to the outer most layer of the matrix.
4. For example, nox is 3. If you add this nox to 2 in the matrix, then 3+2 is 5.
3. When the addition (nox to matrix) makes 5, 5 (from the matrix) disappears. 
4. Another example is when the result of the addition is bigger than 5. For example, if you add nox 4 to 2 in matrix, the result is 6 (4+2 = 6). In this case, the remainder of 1 (after making 5) breaks off. The origianl 2 disappears (because you have reached 5 or more by adding 4 to 2), however, 1 remains. So the total numbers in the matrix is not changed. This acts as a difficulty or penalty in the game.
4. Your goal is to make all of the element in the matrix disappear with the smallest number of noxes. 
5. You will be able to see up to 3 future noxes. You can strategically make your moves anticipating what noxes you will have in near future.

<img src="images/game_set.png" alt="shinko" width="200"/>

## Reinforcement learning to play the game
After training the RL model, I compared the result with a vanilla model. 
Vaniall model is programmed to find the best action based on the current reward. RL model is trained to find the best action based on current and future rewards.
As a result (hold thy breath!), the vaniall model performs better in terms of number of counts for winning the game. Sadly speaking, my RL model is not good enough. However, one interesting thing the RL model can do is to choose the action anticipating the future reward. This will be discussed later (part, Vanilla vs. RL: strategic action trait)

| Model                    | Vaniall model | RL model- trained using 1 environment setting  | RL model - trained using 20 environment settings  |
| ------------------------ |:-------------:| ----------:| ----------:|
| Played 300 random games  | won 28 times   | won 7 times|won Y times|
 

## Training setup
### Game playing module
`playShinko_vanilla.py` 
<br />
This script plays Shinko based on simple logic of addition. For the sake of simplicity, it has modified the original mobile game by removing the spliting feature. Instead of the game allowing you to add 2 with 5 to make 7 (and then resulting remainder 2 remains), you are prohibited to make any move that will result in the sum of the addition to be larger than 5. `self.valid_actions` keeps track of what moves are valid to make for each turn. Here, `action` refers to the index of the matrix. So if `action` = 7, it means the player chooses to make the addition of nox to flattened matrix index 7.
<br />
<br />
The logic for `playShinko_vanilla.py` is simple. The machine will play the game by 
* Look at the next 1 nox
* Among the valid moves, choose the best move by making this calculation: 5 - matrix - nox
* If the result of the above calcuation is 0, that means the nox will surely make the addition to result in 5
* The model chooses the smallest result value (excluding negatives) as its best move

To see the sample of the vanilla play, execute the file in python
```python
python3 playShinko_vanilla.py
```

### Reinforcement learning module
Reinforcement learning is divided into two components:
* `playShinko_rl.py`  
* `trainShinkoAgent.py`  

#### playShinko_rl.py
This file follows the same game features of the `playShinko_vanilla.py`, but few new/altered configuration to enable the reinforcement learning. Most notably, this file pre-process the input data for the neural network built from keras.
<br />
<br />
In order to to feed the current state and noxes together to the neural network, I have transformed the input data into 3 by n(=matrix_width) array. Shinko agent must be able to anticipate the future value 

![shinko preprocses](images/shinko_input_preprocessing.png)

#### trainShinkoAgent.py
Using keras, I built a neural netork to train to play Shinko. 

```python
def qtable_nn_model():
	# build model
    model = Sequential()
    model.add( InputLayer( batch_input_shape = ( 3, flat_matrix_length ) ) )
    model.add( Dense( 30, activation= 'sigmoid' ) )
    model.add( Dense( flat_matrix_length, activation = 'linear' ) )

    # compile model
    model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
    return model
```

When training, I experimented with number of environments to train with. Here, environment refers to the matrix initialization in the beginning. I tried upto 20 different environment as more than 20 took tool long to train on my MacBook Pro. If I set up more powerful environment in the cloud, I expect I can can be more aggressive with training. 
<br />
<br />
After training is over, you need to save the model. I am using Kera's built in method, which I learned from [here.](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)

```python
# save the model
model_json = trained_model.to_json()
with open( 'model.json', 'w') as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
trained_model.save_weights("model.h5")
print("Saved model to disk")
```

### Comparision/evaluation module
`playShinko_rl_test.py`
It is time to do the testing. Since Shinko has not labeled dataset, it is not likely that I can get an accuracy. Instead, I evaluated how well it can play the game Shinko compared to the vanilla model. Vaniall model is programmed to find the best action based on the current reward. RL model is trained to find the best action based on current and future rewards.  
<br />
<br />
The first thing to do is to load the model. 
```python
# load trained model by loading json and creating model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
```
I want to evaluate the model from the two aspects: (1) whether it wins more games; and (2) whether it can anticipate future reward to play strategically at the current state. 

#### Vanilla vs. RL: win counts 
| Model                    | Vaniall model | RL model- trained using 1 environment setting  | RL model - trained using 20 environment settings  |
| ------------------------ |:-------------:| ----------:| ----------:|
| Played 300 random games  | won 28 times   | won 7 times|won 7 times|
 
I played 300 random games, meaning, 300 Shinko with 300 different environment. The reason why I gave RL model different environment is to avoid overfitting. I want it to be able to play new random games as well. 

#### Vanilla vs. RL: strategic action trait
Only after the first few 1000 training, the RL model was able to strategically choose an action based on future rewards. For example, this is what happened during testing.
<img src="images/Picture1.png" alt="shinko" width="600"/>
I made it play one of the random games, and on its 3rd move, it was given noxes of 3, 1, 2 (in that order) with a matrix of [[3,2,2,1,2],[1,3,2,1,5]]. 
Shinko agent had to choose an action with nox 3 to the matrix, and it chose action 8. Strictly speaking from current reward perspective only, it does not make sense much in the beginning, becuase 3+1 is not 5, but 4. However, when you remember what is coming after 3, is 1. This 1 (next nox) can help making 4 into a 5. And that is precisely what the model choose as an action
<br />
<br />
If it was a vanilla model, it would have probably chosen index 4 or 7 to yield an immediate reward by making a 5. Sadly, I guess this strategic action alone is not enough to beat the game. 

### After thoughts
Few ideas on how to improve the model performance
* Add more layers to the network
* Better reward strategy





