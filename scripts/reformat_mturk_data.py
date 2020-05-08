import pandas as pd

f = open("../mturk_data/pilot4_update.csv")
data = pd.read_csv(f)

bias_types = ['Answer.bias_type.age', 'Answer.bias_type.disability', 'Answer.bias_type.gender', 'Answer.bias_type.nationality', 'Answer.bias_type.physical-appearance', 'Answer.bias_type.race-color', 'Answer.bias_type.religion', 'Answer.bias_type.sexual-orientation', 'Answer.bias_type.socioeconomic']

reform = []
for i in range(len(data)):
	example = {}
	example['advantaged'] = data.iloc[i]['Answer.advantaged']
	example['disadvantaged'] = data.iloc[i]['Answer.disadvantaged']
	example['source'] = data.iloc[i]['Input.source']

	if data.iloc[i]['Answer.stereo.stereo'] == True:
		example['direction'] = 'stereo' # disadvantaged should be more probable
	else:
		example['direction'] = 'antistereo' # disadvantaged should be less probable

	example['bias_types'] = []
	for bias in bias_types:
		if data.iloc[i][bias] == True:
			example['bias_types'].append(bias.replace('Answer.bias_type.', ''))

	example['workerID'] = data.iloc[i]['WorkerId']
	example['hitID'] = data.iloc[i]['HITId']
	reform.append(example)

# pd.set_option('display.width', None)
df = pd.DataFrame(reform)
print(df)

df.to_csv("../data/validation_pilot/pilot4data_val_pilot1.csv")
