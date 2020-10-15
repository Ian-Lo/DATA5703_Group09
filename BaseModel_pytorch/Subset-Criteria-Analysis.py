import jsonlines
from tqdm import tqdm
import numpy as np

# open the descriptions
descriptions = jsonlines.open('Statistics/descriptions.jsonl', 'r')

# change these criteria to compute the size of the subset
# only tables fitting these criteria will be counted
num_structural_tokens_criterion = 200
num_cells_criterion = 30
max_num_cell_content_tokens_criterion = 50

count = 0
max_num_structural_tokens = 0
max_num_cells = 0
max_max_num_cell_content_tokens = 0

imgids = []

# check every description and only counts the ones fitting the criteria
for description in tqdm(descriptions):

    num_structural_tokens = description['n_str_tk']
    num_cells = description['n_cells']
    max_num_cell_content_tokens = description['max_n_cell_tk']

    # count only if fitting all criteria
    if (num_structural_tokens <= num_structural_tokens_criterion) & \
       (num_cells <= num_cells_criterion) & \
       (max_num_cell_content_tokens <= max_num_cell_content_tokens_criterion):

        count += 1

        # keep track of the actual maxima
        # note: we do this because some criteria might cause other criteria never to be reached
        if num_structural_tokens > max_num_structural_tokens:
            max_num_structural_tokens = num_structural_tokens
        if num_cells > max_num_cells:
            max_num_cells = num_cells
        if max_num_cell_content_tokens > max_max_num_cell_content_tokens:
            max_max_num_cell_content_tokens = max_num_cell_content_tokens

        # collect the imageid
        imgid = description['imgid']
        imgids.append(imgid)

imgids = np.array(imgids)
np.save('SubsetCriteria/subset_imgids.npy', imgids)

with open('SubsetCriteria/subset_analysis.txt', 'w') as f:
    f.write('INITIAL CRITERIA\n')
    f.write('Criterion on the Number of Structural Tokens: {}\n'.format(num_structural_tokens_criterion))
    f.write('Criterion on the Number of Cells: {}\n'.format(num_cells_criterion))
    f.write('Criterion on the Maximum Number of Cell Content Tokens: {}\n'.format(max_num_cell_content_tokens_criterion))
    f.write('\n')
    f.write('ANALYSIS\n')
    f.write('Number of Examples in the Subset: {}\n'.format(count))
    f.write('Actual Maximum Number of Structural Tokens: {}\n'.format(max_num_structural_tokens))
    f.write('Actual Maximum Number of Cells: {}\n'.format(max_num_cells))
    f.write('Actual Maximum of Maximum of Cell Content Tokens: {}\n'.format(max_max_num_cell_content_tokens))

print('Number of Examples in the Subset: {}'.format(count))

if len(imgids) != len(set(imgids)):
    print('Duplicated ImageIds!!!')
    assert 0



