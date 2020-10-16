import os


# create a path to a location outside iCloud
def create_abs_path(path):

    path = os.path.abspath(os.path.join('/Users/andersborges/Downloads/pubtabnet/', path)) # pubtabnet path

    return path


# load the vocabulary of structural tokens
# this is based on the entire 500k+ dataset
def load_structural_vocabularies():

    structural_token2integer = {}
    with open('Vocabularies/structural-token2integer.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip('\n').split(sep=',')
            token = item[0]
            integer = int(item[1])
            structural_token2integer[token] = integer

    structural_integer2token = {}
    with open('Vocabularies/structural-integer2token.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip('\n').split(sep=',')
            integer = int(item[0])
            token = item[1]
            structural_integer2token[integer] = token

    return structural_token2integer, structural_integer2token


# load the vocabulary of cell content tokens
# this is based on the entire 500k+ dataset
def load_cell_content_vocabularies():

    cell_content_token2integer = {}
    with open('Vocabularies/cell-content-token2integer.tsv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip('\n').split(sep='\t')
            token = item[0]
            integer = int(item[1])
            cell_content_token2integer[token] = integer

    cell_content_integer2token = {}
    with open('Vocabularies/cell-content-integer2token.tsv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip('\n').split(sep='\t')
            integer = int(item[0])
            token = item[1]
            cell_content_integer2token[integer] = token

    return cell_content_token2integer, cell_content_integer2token
