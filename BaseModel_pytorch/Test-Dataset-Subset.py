import Utils
import h5py
import jsonlines


# load vocabularies
structural_token2integer, structural_integer2token = Utils.load_structural_vocabularies()
cell_content_token2integer, cell_content_integer2token = Utils.load_cell_content_vocabularies()

suffix = '00'

storage_features_maps_path = Utils.create_abs_path('Dataset/features_maps_' + suffix + '.hdf5')
storage_features_maps = h5py.File(storage_features_maps_path, "r")

storage_structural_tokens_path = Utils.create_abs_path('Dataset/structural_tokens_' + suffix + '.hdf5')
storage_structural_tokens = h5py.File(storage_structural_tokens_path, "r")

storage_triggers_path = Utils.create_abs_path('Dataset/triggers_' + suffix + '.hdf5')
storage_triggers = h5py.File(storage_triggers_path, "r")

storage_cells_content_tokens_path = Utils.create_abs_path('Dataset/cells_content_tokens_' + suffix + '.hdf5')
storage_cells_content_tokens = h5py.File(storage_cells_content_tokens_path, "r")

features_maps = storage_features_maps['data']
structural_tokens = storage_structural_tokens['data']
triggers = storage_triggers['data']
cells_content_tokens = storage_cells_content_tokens['data']

example1_fm = features_maps[0]
example1_s_tk = structural_tokens[0]
example1_trig = triggers[0]
example1_cc_tk = cells_content_tokens[0]

print([structural_integer2token[x] for x in example1_s_tk])
print(example1_trig)
print([cell_content_integer2token[x] for x in example1_cc_tk[0]])

storage_features_maps.close()
storage_structural_tokens.close()
storage_triggers.close()
storage_cells_content_tokens.close()

# open the annotations file
annotations_filename = Utils.create_abs_path('pubtabnet/PubTabNet_2.0.0.jsonl')
annotations = jsonlines.open(annotations_filename, 'r')

# open the metadata file
metadata_filename = Utils.create_abs_path('Dataset/metadata.jsonl')
metadata = jsonlines.open(metadata_filename, 'r')

for meta in metadata:

    example_imgid = meta['imgid']
    example_filename = meta['filename']
    example_storage = meta['storage']
    example_index = meta['index']

    break

print(example_imgid)
print(example_filename)
print(example_storage)
print(example_index)

for annotation in annotations:

    imgid = annotation['imgid']

    if imgid == example_imgid:

        print(annotation['filename'])
        print(annotation['html']['structure']['tokens'])
        print([cell['tokens'] for cell in annotation['html']['cells']][0])

        break

annotations.close()
metadata.close()



