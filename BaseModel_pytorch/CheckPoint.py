import torch
import os


class CheckPoint:

    def __init__(self, model_tag):

        self.model_tag = model_tag

        self.state = None

        self.best_evaluation_metric = 0

    # save the current checkpoint
    # note: this checkpoint is local and unique
    # note: the previous checkpoint is over-written
    def save_checkpoint(self, epoch,
                              encoder, decoder_structural, decoder_cell_content,
                              encoder_optimizer, decoder_structural_optimizer, decoder_cell_content_optimizer):

        # assemble the state of the model
        state = {'model_tag': self.model_tag,
                 'epoch': epoch,
                 'encoder': encoder,
                 'decoder_structural': decoder_structural,
                 'decoder_cell_content': decoder_cell_content,
                 'encoder_optimizer': encoder_optimizer,
                 'decoder_structural_optimizer': decoder_structural_optimizer,
                 'decoder_cell_content_optimizer': decoder_cell_content_optimizer}

        self.state = state

        # store the checkpoint
        # this local checkpoint is over-witten
        # every time a new checkpoint is saved
        filename = 'checkpoint.pth.tar'
        torch.save(state, filename)

    # archive the checkpoint into a sub-folder of 'Checkpoints/'
    def archive_checkpoint(self):

        # create archive folder if it does not exist
        path = f'Checkpoints/{self.model_tag}'
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            os.makedirs(path)

        # create filename
        epoch = self.state['epoch']
        suffix = '{:0>3}'.format(epoch)
        filename = f'checkpoint_{suffix}.pth.tar'

        # archive the checkpoint
        torch.save(self.state, os.path.join(path, filename))

    # archive the checkpoint, and label it as best_checkpoint,
    # if the evaluation metric is better than the last best evaluation metric
    def archive_checkpoint_if_best(self, evaluation_metric):

        # keep track of the best checkpoint
        if evaluation_metric > self.best_evaluation_metric:

            # create archive folder if it does not exist
            path = f'Checkpoints/{self.model_tag}'
            path = os.path.abspath(path)
            if not os.path.isdir(path):
                os.makedirs(path)

            # create filename
            filename = 'best_checkpoint.pth.tar'

            # store the best checkpoint
            torch.save(self.state, os.path.join(path, filename))

    # load the current checkpoint
    # note: this is always the local (most recent) checkpoint
    # not any of the archived (including best) ones
    @staticmethod
    def load_checkpoint():

        # load the checkpoint
        # this local checkpoint is over-witten
        # every time a new checkpoint is saved
        filename = 'checkpoint.pth.tar'
        state = torch.load(filename)

        return state
