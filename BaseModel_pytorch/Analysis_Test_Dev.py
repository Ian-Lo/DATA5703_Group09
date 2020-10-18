import jsonlines
import numpy as np


# imgids of examples to use for the creation of the subset
dev_imgids = np.load('SubsetCriteria/dev_imgids.npy')
test_imgids = np.load('SubsetCriteria/test_imgids.npy')

for split in ['dev', 'test']:

    # open the descriptions
    descriptions = jsonlines.open('Statistics/descriptions.jsonl', 'r')

    count = 0
    max_num_structural_tokens = 0
    max_num_cells = 0
    max_max_num_cell_content_tokens = 0

    if split == 'dev':

        imgids = dev_imgids

    elif split == 'test':

        imgids = test_imgids

    for description in descriptions:

        # unpack the description
        imgid = description['imgid']
        num_structural_tokens = description['n_str_tk']
        num_cells = description['n_cells']
        max_num_cell_content_tokens = description['max_n_cell_tk']

        if imgid in imgids:

            # keep track of the actual maxima
            if num_structural_tokens > max_num_structural_tokens:
                max_num_structural_tokens = num_structural_tokens
            if num_cells > max_num_cells:
                max_num_cells = num_cells
            if max_num_cell_content_tokens > max_max_num_cell_content_tokens:
                max_max_num_cell_content_tokens = max_num_cell_content_tokens

            count += 1

    with open(f'SubsetCriteria/{split}_analysis.txt', 'w') as f:
        f.write('ANALYSIS\n')
        f.write('Number of Examples in the Subset: {}\n'.format(count))
        f.write('Actual Maximum Number of Structural Tokens: {}\n'.format(max_num_structural_tokens))
        f.write('Actual Maximum Number of Cells: {}\n'.format(max_num_cells))
        f.write('Actual Maximum of Maximum of Cell Content Tokens: {}\n'.format(max_max_num_cell_content_tokens))
