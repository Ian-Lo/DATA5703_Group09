from CheckPoint import CheckPoint


# at instantiation, provide a simple tag that describe the model run
# this tag is used to create a subfolder under the 'Checkpoints/' folder
checkpoint = CheckPoint('anders')

# save the checkpoint, the previous copy of the checkpoint is over-written
# note: ALWAYS run this function BEFORE running the other functions
# note: it is encouraged to save the checkpoint at the end of every epoch
# note: here we use strings for testing but in reality you should pass
# the real objects describing model blocks and corresponding optimizers
checkpoint.save_checkpoint(3, 'encoder', 'decoder_structural', 'decoder_cell_content',
                              'encoder_optimizer', 'decoder_structural_optimizer', 'decoder_cell_content_optimizer')

# archive the checkpoint to a sub-folder of the 'Checkpoints/' folder
# note: if there are many epochs, maybe just archive every n epochs
checkpoint.archive_checkpoint()

# archive the checkpoint only if the evaluation metric
# is better than the previous best evaluation metric
# note: the previous best checkpoint will be over-written
# note: it would be good to call this function after every call of 'save_checkpoint'
# note: we suppose that there is one single value that
# describe the goodness of the model (larger value, better model)
# this could be, for example, the average TEDS score on the dev set
checkpoint.archive_checkpoint_if_best(5)

# load the local checkpoint
state = checkpoint.load_checkpoint()

# un-bundle the state
print(state['model_tag'])
print(state['encoder'])
print(state['decoder_structural'])
print(state['decoder_cell_content'])
print(state['encoder_optimizer'])
print(state['decoder_structural_optimizer'])
print(state['decoder_cell_content_optimizer'])