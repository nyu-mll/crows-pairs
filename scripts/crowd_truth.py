import pandas as pd
import numpy as np 
from collections import Counter
import random

# file1 = "../mturk_data/validation/validation_FULL9505.csv"
# file2 = "../mturk_data/validation/final1999_val_pilot2.csv"

file1 = "../mturk_data/validation/partial_second_stage_results.csv"
MAP = {0:'stereo', 1:'antistereo', 2:'neither'}
revMAP = {'stereo':0, 'antistereo':1}

df_our = pd.read_csv(open(file1, "r"))
# df_our_pilot = pd.read_csv(open(file2, "r"))

bias_types = ['Answer.age.age', 'Answer.disability.disability', 'Answer.gender.gender', 'Answer.nationality.nationality', 'Answer.physical-appearance.physical-appearance', 'Answer.race-color.race-color', 'Answer.religion.religion', 'Answer.sexual-orientation.sexual-orientation', 'Answer.socioeconomic.socioeconomic', 'Answer.none.none']

s1_options = ['Answer.yesBias1.yesBias1', 'Answer.noBias1.noBias1', 'Answer.neitherBias1.neitherBias1']
s2_options = ['Answer.yesBias2.yesBias2', 'Answer.noBias2.noBias2', 'Answer.neitherBias2.neitherBias2']


df = df_our

accepted = []
rejected = []
min_dist = 0
accept = 0
real_accept = 0

examples = df['HITId'].unique()
for ex in examples:
	# ex_accepted = False
	dist = False
	df_ex = df.loc[df['HITId'] == ex]
	s1_votes = []
	s2_votes = []

	mindist_yes = df_ex['Answer.yesMin.yesMin'].sum()
	mindist_no = df_ex['Answer.notMin.notMin'].sum()
	mindist_maj_vote = np.asarray([mindist_yes,mindist_no]).argmax()
	if mindist_maj_vote == 0:
		min_dist += 1
		dist = True
	else:
		pass

	single_pass = [False]*3
	single_passer = [False]*3
	directions = []
	biases = []
	# if df_ex['Input.bias_types'].iloc[0] == '[]':
		# import pdb; pdb.set_trace()
	for j in range(len(df_ex)-2):
		s1_vote = [df_ex.iloc[j][opt] for opt in s1_options]
		s2_vote = [df_ex.iloc[j][opt] for opt in s2_options]
		if np.asarray(s1_vote).argmax() != np.asarray(s2_vote).argmax():
			single_pass[j] = True
			if dist:
				single_passer[j] = True
			 
			if s1_vote[0] == True:
				direction = 'stereo'
			if s1_vote[1] == True:
				direction = 'antistereo'
			if s1_vote[2] == True and s2_vote[0] == True:
				direction = 'antistereo'
			if s1_vote[2] == True and s2_vote[1] == True:
				direction = 'stereo'
				# print('bop')
			directions.append(direction)

			for bias in bias_types:
				if df_ex.iloc[j][bias] == True:
					biases.append(bias.replace('Answer.', '').split(".")[0])

		s1_votes.append(np.asarray(s1_vote).argmax())
		s2_votes.append(np.asarray(s2_vote).argmax())
	
	dirs = Counter(directions)
	bias = Counter(biases)
	# import pdb; pdb.set_trace()
	# acc_ex = {'disadvantaged':df_ex['Input.disadvantaged'].iloc[0], 'advantaged':df_ex['Input.advantaged'].iloc[0], 'directions':directions, 'gold-direction':max(dirs), 'bias_types':biases, 'gold-bias':max(bias), 'hitID':df_ex['Input.hitID'].iloc[0]}

	if sum(single_pass)+1 >= 3:
		accept += 1

	if sum(single_passer)+1 >= 3:
		real_accept += 1
		# accepted.append(acc_ex)
		if len(dirs.values()) == 2:
			tmp = list(dirs.values())
			if tmp[0] == tmp[1]:
				real_accept -= 1
				# accepted.pop()
				# rejected.append(acc_ex)
	else:
		# print(df_ex['Input.disadvantaged'].iloc[0])
		# print(df_ex['Input.advantaged'].iloc[0])
		pass
		# rejected.append(acc_ex)

print(accept, real_accept)
print(len(examples))
df = pd.DataFrame(accepted)
df_r = pd.DataFrame(rejected)

########################################################################
########################################################################
######################### STEREO SET ###################################
########################################################################
########################################################################

file1 = "../mturk_data/validation/stereoset_val_pilot2.csv"
df_stereo = pd.read_csv(open(file1, "r"))

s1_options = ['Answer.yesBias1.yesBias1', 'Answer.noBias1.noBias1', 'Answer.neitherBias1.neitherBias1']
s2_options = ['Answer.yesBias2.yesBias2', 'Answer.noBias2.noBias2', 'Answer.neitherBias2.neitherBias2']

df = df_stereo
min_dist = 0
examples = df['HITId'].unique()
accepted = pd.DataFrame(columns=['stereo', 'anti'])
rejected = pd.DataFrame(columns=['stereo', 'anti'])

accept = 0
real_accept = 0
for ex in examples:
	dist = False
	df_ex = df.loc[df['HITId'] == ex]
	s1_votes = []
	s2_votes = []

	mindist_yes = df_ex['Answer.yesMin.yesMin'].sum()
	mindist_no = df_ex['Answer.notMin.notMin'].sum()
	mindist_maj_vote = np.asarray([mindist_yes,mindist_no]).argmax()
	if mindist_maj_vote == 0:
		min_dist += 1
		dist = True
	else:
		pass

	single_pass = [False]*4
	single_passer = [False]*4
	for j in range(len(df_ex)-1):
		s1_vote = [df_ex.iloc[j][opt] for opt in s1_options]
		s2_vote = [df_ex.iloc[j][opt] for opt in s2_options]
		if np.asarray(s1_vote).argmax() == 0:
			if np.asarray(s2_vote).argmax() == 1 or np.asarray(s2_vote).argmax() == 2:
				single_pass[j] = True
				if dist:
					single_passer[j] = True

		s1_votes.append(np.asarray(s1_vote).argmax())
		s2_votes.append(np.asarray(s2_vote).argmax())

	
	if sum(single_pass) >= 3:
		accept += 1
	else:
		pass
		# print(s1_votes, s2_votes)
		# print(df_ex['Input.stereo'].iloc[0])
		# print(df_ex['Input.anti'].iloc[0], '\n')
	if sum(single_passer) >= 3:
		real_accept += 1
	# else:
	# 	if not dist:
	# 		print(s1_votes, s2_votes)
	# 		print(df_ex['Input.stereo'].iloc[0])
	# 		print(df_ex['Input.anti'].iloc[0], '\n')

print(accept, real_accept)
exit(1)

########################################################################
########################################################################
######################### STAGE-1 VALIDATION ###########################
########################################################################
########################################################################

accepted = []
rejected = []
min_dist = 0
accept = 0
real_accept = 0
for df in [df_our, df_our_pilot]:
	# min_dist = 0
	examples = df['HITId'].unique()
	# accepted = []
	# rejected = []

	# accept = 0
	# real_accept = 0
	for ex in examples:
		# ex_accepted = False
		dist = False
		df_ex = df.loc[df['HITId'] == ex]
		s1_votes = []
		s2_votes = []

		mindist_yes = df_ex['Answer.yesMin.yesMin'].sum()
		mindist_no = df_ex['Answer.notMin.notMin'].sum()
		mindist_maj_vote = np.asarray([mindist_yes,mindist_no]).argmax()
		if mindist_maj_vote == 0:
			min_dist += 1
			dist = True
		else:
			pass

		single_pass = [False]*5
		single_passer = [False]*5
		directions = [df_ex['Input.direction'].iloc[0]]
		biases = [df_ex['Input.bias_types'].iloc[0]]
		# if df_ex['Input.bias_types'].iloc[0] == '[]':
			# import pdb; pdb.set_trace()
		for j in range(len(df_ex)):
			s1_vote = [df_ex.iloc[j][opt] for opt in s1_options]
			s2_vote = [df_ex.iloc[j][opt] for opt in s2_options]
			if np.asarray(s1_vote).argmax() != np.asarray(s2_vote).argmax():
				single_pass[j] = True
				if dist:
					single_passer[j] = True
				 
				if s1_vote[0] == True:
					direction = 'stereo'
				if s1_vote[1] == True:
					direction = 'antistereo'
				if s1_vote[2] == True and s2_vote[0] == True:
					direction = 'antistereo'
				if s1_vote[2] == True and s2_vote[1] == True:
					direction = 'stereo'
					# print('bop')
				directions.append(direction)

				for bias in bias_types:
					if df_ex.iloc[j][bias] == True:
						biases.append(bias.replace('Answer.', '').split(".")[0])

			s1_votes.append(np.asarray(s1_vote).argmax())
			s2_votes.append(np.asarray(s2_vote).argmax())
		
		dirs = Counter(directions)
		bias = Counter(biases)
		acc_ex = {'disadvantaged':df_ex['Input.disadvantaged'].iloc[0], 'advantaged':df_ex['Input.advantaged'].iloc[0], 'directions':directions, 'gold-direction':max(dirs), 'bias_types':biases, 'gold-bias':max(bias), 'hitID':df_ex['Input.hitID'].iloc[0]}

		if sum(single_pass)+1 >= 3:
			accept += 1

		if sum(single_passer)+1 >= 3:
			real_accept += 1
			accepted.append(acc_ex)
			if len(dirs.values()) == 2:
				tmp = list(dirs.values())
				if tmp[0] == tmp[1]:
					real_accept -= 1
					accepted.pop()
					rejected.append(acc_ex)
		else:
			rejected.append(acc_ex)

print(accept, real_accept)
second_stage_val_sample = random.sample(accepted, 100)
df = pd.DataFrame(accepted)
df_r = pd.DataFrame(rejected)
df_sample = pd.DataFrame(second_stage_val_sample)
df_sample.to_csv("second_stage_val_sample.csv", index=False)
df.to_csv("filtered_lmBias_data.csv")
exit(1)










