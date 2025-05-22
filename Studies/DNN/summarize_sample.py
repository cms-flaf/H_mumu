import ROOT as root
from glob import glob
from collections import Counter
from pprint import pprint

def test_file(root_file):
	f = root.TFile(root_file)
	tree = f.Get("Events")
	for event in tree:
		break
	data = {'filename' : root_file}
	data['n_events'] = tree.GetEntries()
	data['count'] = count_types(tree)
	#data['branch_names'] = [x.GetName() for x in event.GetListOfBranches()]
	data['nBranches'] = len(event.GetListOfBranches())
	return data

def count_types(tree):
	count = Counter()
	for event in tree:
		count.update([event.sample_type])
	return count

if __name__ == "__main__":
	for filename in sorted(glob("output_samples/*.root")):
        	print("**************")
       		pprint(test_file(filename))
