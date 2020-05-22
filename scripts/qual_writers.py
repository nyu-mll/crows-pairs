import pandas as pd

workers = "../data/workers_list.csv"
writers = "../mturk_data/dataFULL_1999_examples.csv"

df1 = pd.read_csv(open(workers, "r"))
df2 = pd.read_csv(open(writers, "r"))

# import pdb; pdb.set_trace()
cur = 0
writer_ids = df2['WorkerId'].unique()
for id_ in writer_ids:
	df1.loc[df1['Worker ID'] == id_, 'UPDATE-LMBiasWriter'] = str(1)
	cur += 1
print(cur, len(writer_ids))
# exit(1)
df1.to_csv("../data/worker_list_qualAdded.csv", index=False)