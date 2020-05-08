import json
import random
import pandas as pd
f = open("../data/stereoset/dev.json", "r")
data = json.load(f)

stereo = data['data']
intra = stereo['intrasentence']
sample = random.sample(intra, 100)
# print(len(sample))

stereo_dump = pd.DataFrame(['id', 'target', 'bias_type', 'stereo', 'anti'])
stereo_dump = []
for i, ex in enumerate(sample):
	# print(i)
	# import pdb; pdb.set_trace()
	example = {}
	example['id'] = ex['id']
	example['target'] = ex['target']
	example['bias_type'] = ex['bias_type']
	for j in range(3):
		label = ex['sentences'][j]['gold_label']
		if label == 'stereotype':
			example['stereo'] = ex['sentences'][j]['sentence']
		elif label == 'anti-stereotype':
			example['anti'] = ex['sentences'][j]['sentence']
		else: # label is 'unrelated'
			pass
	stereo_dump.append(example)

stereo_dump = pd.DataFrame(stereo_dump)
print(stereo_dump)
stereo_dump.to_csv("../data/validation_pilot/stereo_sample_val_pilot1.csv")
