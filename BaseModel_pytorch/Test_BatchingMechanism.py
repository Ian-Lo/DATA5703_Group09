from BatchingMechanism import BatchingMechanism

# instantiate the object
batching = BatchingMechanism(dataset_split='train', number_examples=125, batch_size=20, storage_size=100)

# initialise the object
# here the object works out how many storages and how many examples from every storage are needed
batching.initialise()

epochs = [0, 1]

# example without randomisation
# useful for debugging

for epoch in epochs:

    # create random batches of examples
    # these "batches" are the just information needed to retrieve the actual tensors
    # batch = (storage number, [list of indices within the storage])
    batches = batching.build_batches(randomise=False)

    for batch in batches:

        # call 'get_batch' to actually load the tensors from file
        features_maps, structural_tokens, triggers, cells_content_tokens = batching.get_batch(batch)

        print('epoch ', epoch, ' batch ', batch)
        print(features_maps.size(), structural_tokens.size(), triggers.size(), cells_content_tokens.size())

print('\n')

# example with randomisation
# examples in a batch are all from the same storage file
# indices are random within the storage
# consecutive batches can be from different storages
# both storages and indices are shuffled
# despite randomisation, all examples are used only once per epoch
for epoch in epochs:

    # create random batches of examples
    # these "batches" are the just information needed to retrieve the actual tensors
    # batch = (storage number, [list of indices within the storage])
    batches = batching.build_batches(randomise=True)

    for batch in batches:

        # call 'get_batch' to actually load the tensors from file
        features_maps, structural_tokens, triggers, cells_content_tokens = batching.get_batch(batch)

        print('epoch ', epoch, ' batch ', batch)
        print(features_maps.size(), structural_tokens.size(), triggers.size(), cells_content_tokens.size())
