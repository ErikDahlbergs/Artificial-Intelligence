
import random
import numpy as np
from models import TransitionModel,ObservationModel,StateModel

#
# Add your Robot Simulator here
#Will probably not need observation model as the robot only moves according to state and transition
class RobotSim:
    def __init__(self, s_model : StateModel, t_model : TransitionModel, o_model : ObservationModel):
        print("Hello World, my name is Paul")
        self.s_model = s_model #creates a state model with a grid, states and methods for handling states, positions and readings
        self.t_model = t_model #Based on the rules provided in assignment description. Get_T returns the transition matrix. Get_T_ij returns probability of state i_j
        self.o_model = o_model

        self.row, self.col, self.h = self.s_model.get_grid_dimensions()

    def move(self, state: int):
        T = self.t_model.get_T()   # Takes all possibilites based on the current state

        nextStates = T[state] + 1e-8 # Added a small number to smooth the array
        nextStates = nextStates/np.sum(nextStates) # Normalize the probabilities

        return np.random.choice(range(len(nextStates)), p=list(nextStates)) # Chooses a random next state based on probabilities in nextStates

    #Returns a sensorreading based on the current state
    def sensor_reading(self, x : int, y : int, h : int):
       
        prob_from_reading = []
        for p in range(self.o_model.get_nr_of_readings()):
            prob_from_reading.append(self.o_model.get_o_reading_state(p, self.s_model.pose_to_state(x,y,h)))

        cho = np.random.choice(range(len(prob_from_reading)), p=list(prob_from_reading))

        sense_x, sense_y = self.s_model.reading_to_position(cho)
        if sense_x >= 0 and sense_x <= (self.row-1) and sense_y >= 0 and sense_y <= (self.col-1): #Since None is the last in nr_of_reading this if statement handles sensor readings outside of grid
            return sense_x, sense_y
        
        return None
        

#
# Add your Filtering approach here (or within the Localiser, that is your choice!)
# The filter does not use the stateModel as it only operates on chance and observation
class HMMFilter:
    def __init__(self, t_model : TransitionModel, o_model : ObservationModel, s_model : StateModel):
        self.t_model = t_model
        self.o_model = o_model
        self.s_model = s_model
        self.guess = 0

        self.row, self.col, self.h = self.s_model.get_grid_dimensions()

    # Implements the formula from lecture slides f(1:t+1) = O(et)*T(Transpose)*f(1:t)
    # probs is used to save previous result. Creates a semi-recursive approach
    def filtering(self, observation: int, f_vec):
        guess = self.s_model.position_to_reading(observation[0], observation[1]) if observation else None

        T_trans = self.t_model.get_T_transp() #T transposed
        O = self.o_model.get_o_reading(guess) #O-matrice
        f_vec = np.matmul(np.matmul(O, T_trans), f_vec) #O*T(transpose)*f_vec

        
        return f_vec/sum(f_vec), self.s_model.state_to_position(np.argmax(f_vec)) # Returns new f_vec, estimate. Alpha = 1/sum(f_vec). Estimate is the most likely state in f_vec
        


        
        
        
