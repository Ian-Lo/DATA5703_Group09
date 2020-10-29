import os


# create absolute path from local path of the "main" file
def absolute_path(folder_name, file_name):

    return os.path.abspath(os.path.join(os.path.dirname(__file__), folder_name, file_name))


# class to manage the path of the dataset
# note: everything is declared at class level
# so that we do not need to instantiate any object
# note: this class acts as a functional global variable
class DatasetPath:

    # the relative path of the location with the HDF5 files
    relative_path = ''

    @classmethod
    def set_relative_path(cls, relative_path):

        cls.relative_path = relative_path

    @classmethod
    def create_abs_path(cls, path):

        path = absolute_path(cls.relative_path, path)

        return path


# load the vocabulary of structural tokens
# this is based on the entire 500k+ dataset
def load_structural_vocabularies():

    structural_token2integer = {}
    file_path = absolute_path('Vocabularies', 'structural_token2integer.csv')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip('\n').split(sep=',')
            token = item[0]
            integer = int(item[1])
            structural_token2integer[token] = integer

    structural_integer2token = {}
    file_path = absolute_path('Vocabularies', 'structural_integer2token.csv')
    with open(file_path, 'r') as f:
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
    file_path = absolute_path('Vocabularies', 'cell_content_token2integer.tsv')
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip('\n').split(sep='\t')
            token = item[0]
            integer = int(item[1])
            cell_content_token2integer[token] = integer

    cell_content_integer2token = {}
    file_path = absolute_path('Vocabularies', 'cell_content_integer2token.tsv')
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip('\n').split(sep='\t')
            integer = int(item[0])
            token = item[1]
            cell_content_integer2token[integer] = token

    return cell_content_token2integer, cell_content_integer2token
