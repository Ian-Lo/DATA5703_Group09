import numpy as np
import jsonlines
import Utils

# open the annotations
annotations_path = Utils.create_abs_path('PubTabNet_2.0.0.jsonl')
annotations = jsonlines.open(annotations_path, 'r')

# randomly select 20k indices
rng = np.random.seed(291)

indices = np.random.choice(np.array(range(500000)), size=20000, replace=False)

dev_indices = indices[0:10000]
test_indices = indices[10000:20000]

dev_imgids = []
test_imgids = []

idx = 0

for annotation in annotations:

    imgid = annotation['imgid']
    split = annotation['split']

    if split == 'train':

        if idx in dev_indices:

            dev_imgids.append(imgid)

        elif idx in test_indices:

            test_imgids.append(imgid)

    idx += 1

dev_imgids = np.array(dev_imgids)
np.save('SubsetCriteria/dev-imgids.npy', dev_imgids)

test_imgids = np.array(test_imgids)
np.save('SubsetCriteria/test-imgids.npy', test_imgids)
