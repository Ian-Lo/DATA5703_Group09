import Utils
import jsonlines
from collections import Counter
from tqdm import tqdm


# open the annotations file
annotations_filename = Utils.create_abs_path('pubtabnet/PubTabNet_2.0.0.jsonl')
annotations = jsonlines.open(annotations_filename, 'r')

# create a descriptions file
descriptions = jsonlines.open('Statistics/descriptions.jsonl', 'w')

# set up counters
structural_tokens_counter = Counter()
cell_content_tokens_counter = Counter()
num_structural_tokens_counter = Counter()
num_cells_counter = Counter()
max_num_cell_content_tokens_counter = Counter()

for annotation in tqdm(annotations):

    # tags for the image
    imgid = annotation['imgid']
    filename = annotation['filename']
    split = annotation['split']

    if split == 'train':

        # list of structural tokens
        structural_tokens = annotation['html']['structure']['tokens']
        # list of lists of cell content tokens, one per cell
        # note: some cells are empty and they are represented by an empty list
        cells_content_tokens = [cell['tokens'] for cell in annotation['html']['cells']]

        # lists of cell content tokens, one per cell, flattened across cells
        # note: empty lists do not contribute
        cell_content_tokens = [token for cell_content_tokens in cells_content_tokens for token in cell_content_tokens]

        # collect counts
        structural_tokens_counter.update(structural_tokens)
        cell_content_tokens_counter.update(cell_content_tokens)

        # statistics on the annotation
        num_structural_tokens = len(structural_tokens)
        num_cells = len(cells_content_tokens)
        max_num_cell_content_tokens = max([len(cell) for cell in cells_content_tokens])

        # collect counts
        num_structural_tokens_counter.update([num_structural_tokens])
        num_cells_counter.update([num_cells])
        max_num_cell_content_tokens_counter.update([max_num_cell_content_tokens])

        # assemble the description
        description = {'imgid': imgid,
                       'n_str_tk': num_structural_tokens,
                       'n_cells': num_cells,
                       'max_n_cell_tk': max_num_cell_content_tokens}
        # write the descriptions
        descriptions.write(description)

# close JSON files
annotations.close()
descriptions.close()

# store to file
with open('Statistics/structural_tokens_count.csv', 'w') as f:
    f.write('token,count\n')
    for key, value in structural_tokens_counter.most_common():
        f.write('{},{}\n'.format(key, value))

# store to file
# we use a tab separator as one of the tokens is ','
# and that causes confusion while parsing the file later on
with open('Statistics/cell_content_tokens_count.tsv', 'w') as f:
    f.write('token\tcount\n')
    for key, value in cell_content_tokens_counter.most_common():
        # we filter out 'unprintable' characters
        # as they are impossible to parse later on
        if key.isprintable():
            f.write('{}\t{}\n'.format(key, value))
        else:
            print('unprintable character codepoint: ', ord(key), ' count: ', int(value))

# store to file
with open('Statistics/num_structural_tokens_count.csv', 'w') as f:
    f.write('num structural tokens,count\n')
    for key, value in num_structural_tokens_counter.most_common():
        f.write('{},{}\n'.format(key, value))

# store to file
with open('Statistics/num_cells_count.csv', 'w') as f:
    f.write('num cells,count\n')
    for key, value in num_cells_counter.most_common():
        f.write('{},{}\n'.format(key, value))

# store to file
with open('Statistics/max_num_cell_content_tokens_count.csv', 'w') as f:
    f.write('max num tokens,count\n')
    for key, value in max_num_cell_content_tokens_counter.most_common():
        f.write('{},{}\n'.format(key, value))

# create the vocabulary of structural tokens
# this is based on the entire 500k+ dataset
# tokens are ordered in decreasing frequency
structural_token2integer = {'<pad>': 0, '<start>': 1, '<end>': 2, '<oov>': 3}
structural_integer2token = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<oov>'}
j = 4
for key, value in structural_tokens_counter.most_common():
    structural_token2integer[key] = j
    structural_integer2token[j] = key
    j += 1

# store to file
with open('Vocabularies/structural_token2integer.csv', 'w') as f:
    for key, value in structural_token2integer.items():
        f.write('{},{}\n'.format(key, value))

# store to file
with open('Vocabularies/structural_integer2token.csv', 'w') as f:
    for key, value in structural_integer2token.items():
        f.write('{},{}\n'.format(key, value))

# create the vocabulary of structural tokens
# this is based on the entire 500k+ dataset
# tokens are ordered in decreasing frequency
cell_content_token2integer = {'<pad>': 0, '<start>': 1, '<end>': 2, '<oov>': 3}
cell_content_integer2token = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<oov>'}
j = 4
for key, value in cell_content_tokens_counter.most_common():
    # we filter out 'unprintable' characters
    # as they are impossible to parse later on
    if key.isprintable():
        cell_content_token2integer[key] = j
        cell_content_integer2token[j] = key
        j += 1

# store to file
# we use a tab separator as one of the tokens is ','
# and that causes confusion while parsing the file later on
with open('Vocabularies/cell_content_token2integer.tsv', 'w') as f:
    for key, value in cell_content_token2integer.items():
        f.write('{}\t{}\n'.format(key, value))

# store to file
# we use a tab separator as one of the tokens is ','
# and that causes confusion while parsing the file later on
with open('Vocabularies/cell_content_integer2token.tsv', 'w') as f:
    for key, value in cell_content_integer2token.items():
        f.write('{}\t{}\n'.format(key, value))
