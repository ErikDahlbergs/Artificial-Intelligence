import gym
import random
import requests
import numpy as np
import argparse
import sys
from gym_connect_four import ConnectFourEnv
import pdb
import copy

env: ConnectFourEnv = gym.make("ConnectFour-v0")

SERVER_ADDRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["er8251da-s"] # TODO: fill this list with your stil-id's

def call_server(move):
	res = requests.post(SERVER_ADDRESS + "move",
							  data={
									"stil_id": STIL_ID,
									"move": move, # -1 signals the system to start a new game. any running game is counted as a loss
									"api_key": API_KEY,
							  })
	# For safety some respose checking is done here
	if res.status_code != 200:
		print("Server gave a bad response, error code={}".format(res.status_code))
		exit()
	if not res.json()['status']:
		print("Server returned a bad status. Return message: ")
		print(res.json()['msg'])
		exit()
	return res

def check_stats():
	res = requests.post(SERVER_ADDRESS + "stats",
							  data={
									"stil_id": STIL_ID,
									"api_key": API_KEY,
							  })

	stats = res.json()
	return stats

"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""
def opponents_move(env: ConnectFourEnv):
	env.change_player() # change to oppoent
	avmoves = env.available_moves()
	if not avmoves:
		env.change_player() # change back to student before returning
		return -1

	# TODO: Optional? change this to select actions with your policy too
	# that way you get way more interesting games, and you can see if starting
	# is enough to guarrantee a win
	#action = random.choice(list(avmoves)) # REVERT: som det va innan
	print(env.board)
	action = int(input("Skriv siffra mellan 1 och 7"))-1

	state, reward, done, _ = env.step(action)
	if done:
		if reward == 1: # reward is always in current players view
			reward = -1
	env.change_player() # change back to student before returning
	return state, reward, done

def evaluateSum(cells: list):
	total = sum(cells)
	minimum = min(cells)
	maximum = max(cells)

	if total == 4:
		return 1000
	elif total == -4:
		return -999
	elif total == 3 and minimum > -1:
		return 100
	elif total == -3 and maximum < 1:
		return -99
	elif total == 2 and minimum > -1:
		return 5
	elif total == -2 and maximum < 1:
		return -5
	else:
		return 0
	

def evaluate(Board: ConnectFourEnv):
	value = 0
	board = copy.deepcopy(Board)

	#extract rows
	for row in range(6):
		row_array = list(board.board[row][0:len(board.board[0]-1)])
		for cell in range(4):
			cells = [row_array[cell], row_array[cell+1], row_array[cell+2], row_array[cell+3]]
			value += evaluateSum(list(cells))

	#extract columns
	for col in range(7):
		col_array = [board.board[5][col], board.board[4][col], board.board[3][col], board.board[2][col], board.board[1][col], board.board[0][col] ]
		for cell in range(3):
			cells = [col_array[cell], col_array[cell+1], col_array[cell+2], col_array[cell+3]]
			value += evaluateSum(list(cells))

	#extract diagonal
	for col in range(4):
		for row in range(3):
			diag_array = [board.board[5-row][col], board.board[4-row][col+1], board.board[3-row][col+2], board.board[2-row][col+3]]
			value += evaluateSum(diag_array)

	#extract reversed diagonal
	for col in range(4):
		for row in range(3):
			#for cell in range(4):
			revdiag_array = [board.board[5-row][6-col], board.board[4-row][5-col], board.board[3-row][4-col], board.board[2-row][3-col]]
			value += evaluateSum(revdiag_array)

	return value

def alphabeta(board: ConnectFourEnv, depth, alpha, beta, maximizingPlayer, done):
	if depth == 0 or done:
		return evaluate(board)
	if maximizingPlayer:
		value = -np.inf
		availableMoves = board.available_moves()
		#bestMove = random.choice(list(availableMoves))
		for move in availableMoves:
			new_board = copy.deepcopy(board)
			new_state, new_result, done, _ = new_board.step(move)
			new_board.change_player()
			new_value = alphabeta(new_board, depth-1, alpha, beta, False, done)[0]
			if new_value > value:
				value = new_value
				bestMove = move
			if (value >= beta):
				break
			alpha = max(value, alpha)
		# return bestMove
		return bestMove, value
	else:
		value = np.inf
		availableMoves = board.available_moves()
		#bestMove = random.choice(list(availableMoves))
		for move in availableMoves:
			new_board = copy.deepcopy(board)
			new_state, new_result, done, _ = new_board.step(move)
			new_board.change_player()
			new_value = alphabeta(new_board, depth-1, alpha, beta, True, done)[0]
			if new_value < value:
				value = new_value
				bestMove = move
			if alpha >= value:
				break
			beta = min(beta, value)
		#return bestMove
		return bestMove, value

def alphabeta2(Board: ConnectFourEnv, depth, alpha, beta, maximizingPlayer, game_over):
	if depth == 0 or game_over:
		return evaluate(copy.deepcopy(Board))
	if maximizingPlayer:
		value = -np.inf
		for move in Board.available_moves():
			board = copy.deepcopy(Board)
			board.change_player()
			st, res, done, _ = board.step(move)
			new_value = alphabeta2(board, depth-1, alpha, beta, False, done)
			if new_value > value:
				value = new_value
			if value >= beta:
				break
			alpha = max(alpha, value)
			return value
	else:
		value = np.inf
		for move in Board.available_moves():
			board = copy.deepcopy(Board)
			board.change_player()
			st, res, done, _ = board.step(move)
			new_value = alphabeta2(board, depth-1, alpha, beta, True, done)
			if new_value < value:
				value = new_value
			beta = min(value, beta)
			if (value <= alpha):
				break
			return value

def student_move(env: ConnectFourEnv):
	"""
	TODO: Implement your min-max alpha-beta pruning algorithm here.
	Give it whatever input arguments you think are necessary
	(and change where it is called).
	The function should return a move from 0-6
	"""
	# max = -np.inf # Used to track highest
	# bestMove = None
	# available = env.available_moves()
	# for moves in available:
	# 	eval = alphabeta(env, 1, -np.inf, 4, True)
	# 	if (eval > max):
	# 		max = eval
	# 		bestMove = moves
	value = -np.inf

	for move in env.available_moves():
		environment = copy.deepcopy(env)
		environment.step(move)
		values = alphabeta2(environment, 5, -np.inf, np.inf, True, False)
		if value < values:
			value = values
			choice = move

	#bestMove = alphabeta(env, 5, -np.inf, np.inf, True, False)
	#print(str(bestMove))
	#return bestMove[0]
	
	return choice

def play_game(vs_server = False):
	"""
	The reward for a game is as follows. You get a
	botaction = random.choice(list(avmoves)) reward from the
	server after each move, but it is 0 while the game is running
	loss = -1
	win = +1
	draw = +0.5
	error = -10 (you get this if you try to play in a full column)
	Currently the player always makes the first move
	"""

	# default state
	state = np.zeros((6, 7), dtype=int)

	# setup new game
	if vs_server:
		# Start a new game
		res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

		# This should tell you if you or the bot starts
		print(res.json()['msg'])
		botmove = res.json()['botmove']
		state = np.array(res.json()['state'])
		# reset env to state from the server (if you want to use it to keep track)
		env.reset(board=state)
	else:
		# reset game to starting state
		env.reset(board=None)
		# determine first player
		student_gets_move = random.choice([True, False])
		if student_gets_move:
			print('You start!')
			print()
		else:
			print('Bot starts!')
			print()

	# Print current gamestate
	print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
	print(state)
	print()

	done = False
	while not done:
		# Select your move
		stmove = student_move(env) # TODO: change input here
	  
		# make both student and bot/server moves
		if vs_server:
			# Send your move to server and get response
			res = call_server(stmove)
			print(res.json()['msg'])

			# Extract response values
			result = res.json()['result']
			botmove = res.json()['botmove']
			state = np.array(res.json()['state'])
			# reset env to state from the server (if you want to use it to keep track)
			env.reset(board=state)
		else:
			if student_gets_move:
				# Execute your move
				avmoves = env.available_moves()
				if stmove not in avmoves:
					print("You tied to make an illegal move! You have lost the game.")
					break
				state, result, done, _ = env.step(stmove)

			student_gets_move = True # student only skips move first turn if bot starts

			# print or render state here if you like

			# select and make a move for the opponent, returned reward from students view
			if not done:
				state, result, done = opponents_move(env)

		# Check if the game is over
		if result != 0:
			done = True
			if not vs_server:
				print("Game over. ", end="")
			if result == 1:
				print("You won!")
			elif result == 0.5:
				print("It's a draw!")
			elif result == -1:
				print("You lost!")
			elif result == -10:
				print("You made an illegal move and have lost!")
			else:
				print("Unexpected result result={}".format(result))
			if not vs_server:
				print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
		else:
			print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

		# Print current gamestate
		print(state)
		#print(input("tanke"))
		print()

def main():
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group()
	group.add_argument("-l", "--local", help = "Play locally", action="store_true")
	group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
	parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
	args = parser.parse_args()

	# Print usage info if no arguments are given
	if len(sys.argv)==1:
		parser.print_help(sys.stderr)
		sys.exit(1)

	if args.local:
		play_game(vs_server = False)
	elif args.online:
		play_game(vs_server = True)

	if args.stats:
		stats = check_stats()
		print(stats)

	# TODO: Run program with "--online" when you are ready to play against the server
	# the results of your games there will be logged
	# you can check your stats bu running the program with "--stats"

if __name__ == "__main__":
	 main()
