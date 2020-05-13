import pandas as pd
from collections import Counter
import math

path = "../mturk_data/batch_pilot3.csv"
f = open(path, "r")

df = pd.read_csv(f)
workers = df['WorkerId'].unique()

# import pdb; pdb.set_trace()
bias_types = ['Answer.bias_type.age','Answer.bias_type.disability', 'Answer.bias_type.gender',
			  'Answer.bias_type.nationality', 'Answer.bias_type.other', 'Answer.bias_type.physical-appearance',
       		  'Answer.bias_type.race-color', 'Answer.bias_type.religion',
       		  'Answer.bias_type.sexual-orientation', 'Answer.bias_type.socioeconomic',
       		  'Answer.other-option']

bonuses = []
print("(WorkerId, Bonus amount, One AssignmentId they completed)")
for i, worker in enumerate(workers):
	worker_df = df.loc[df['WorkerId'] == workers[i]]
	# print(len(worker_df))
	worker_bonus = 0
	num_types = [] 
	if len(worker_df) > 3:
		for bias in bias_types:
			trues = len(worker_df[worker_df[bias] == True])
			if trues > 0:
				num_types += [bias] * trues
		
		cur = Counter(num_types)
		curt = list(cur.values())
		while len(curt) >= 4:
			# print(curt)
			worker_bonus += math.floor(len(curt)/4)
			curt = [x-1 for x in curt[:4]] + curt[4:]
			curt = list(filter(lambda a: a != 0, curt))

	if worker_bonus > 0:
		bonuses.append((worker, worker_bonus, worker_df['AssignmentId'].iloc[0]))

print(bonuses)