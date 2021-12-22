import numpy as np
import random
from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Flatten, Dense
import math

# hyper-parameters and constants
DECAY_RATE = 0.99
BUFFER_SIZE = 40000
MINIBATCH_SIZE = 64
TOT_FRAME = 3000000
EPSILON_DECAY = 1000000
MIN_OBSERVATION = 5000
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 0.1
NUM_ACTIONS = 6
TAU = 0.01
# Number of frames to throw into network
NUM_FRAMES = 3

class DeepQ(object):
    def __init__(self):
        self.construct_q_network()

    def construct_q_network(self):
        # Uses the network architecture found in DeepMind paper
        self.model = Sequential()
        self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(NUM_ACTIONS))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.00001))

        self.target_model = Sequential()
        self.target_model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 3, 3))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Flatten())
        self.target_model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.target_model.add(Dense(NUM_ACTIONS))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.00001))
        self.target_model.set_weights(self.model.get_weights())

        print("Successfully constructed networks.")

    def predict_movement_Boltzmann(self, data, epsilon):
        # Boltzmann exploration is that the softmax over network outputs provides a measure of the agent’s confidence in each action
        # Boltzmann Exploration Approach
        q_list = self.model.predict(data.reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
        actions = [i for i in range(len(q_list[0]))]
        tau = 500
        probs = []
        for q in q_list[0]:
            #num = exp(Q(s,a)/tau)
            num = math.exp(q/tau)
            exp_values_for_dem = []
            for q_t in q_list[0]:
                #exp(Q(s,a_t)/tau)
                exp_values_for_dem.append(math.exp(q_t/tau))
            #den = sum_t(exp(Q(s,a_t)/tau))
            den = sum(exp_values_for_dem)
            #num/den
            probs.append(round(num/den, 10))
            
        opt_policy = random.choices(actions, weights=probs, k=1)[0]
        return opt_policy, q_list[0,opt_policy]
        
    def predict_movement_UCB(self, data, epsilon):
        # Predict movement with UCB.
        #Bayesian Approach not finished yet.
        q_actions = self.model.predict(data.reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
        probs = []
        for q in q_actions[0]:
            if q != 0:
                if 2*math.log(q/(1+len(q_actions[0]))) > 0:
                    j = q + math.sqrt(2*math.log(q/(1+len(q_actions[0]))))
                else:
                    j = 0
                probs.append(j)
            else: 
                probs.append(0)
        opt_policy = np.argmax(probs)
        return opt_policy, q_actions[0,opt_policy]

    def predict_movement_Epsilon(self, data, epsilon):
        # Predict movement of game controler where is epsilon probability randomly move.
        #epsilon greedy approach
        q_actions = self.model.predict(data.reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, NUM_ACTIONS)
        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        # Trains network to fit given parameters
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, NUM_ACTIONS))

        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)

        loss = self.model.train_on_batch(s_batch, targets)

        # Print the loss every 10 iterations.
        if observation_num % 10 == 0:
            print("Loss value : ", loss)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Saved network.")

    def load_network(self, path):
        self.model = load_model(path)
        print("Loaded network.")

    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)
