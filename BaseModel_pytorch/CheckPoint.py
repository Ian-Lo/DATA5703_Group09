import Utils
import torch
import os
from datetime import datetime, timezone
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


class CheckPoint:

    # load the current checkpoint
    # note: there is no need to instantiate an object
    # call as: CheckPoint.load_checkpoint()
    @classmethod
    def load_checkpoint(cls, file_path='checkpoint.pth.tar'):

        # load the latest checkpoint
        # this local checkpoint is over-witten
        # every time a new checkpoint is saved
        file_path = Utils.absolute_path('', file_path)
        state = torch.load(file_path)

        return state

    def refresh_gauth(self):
        """
        docstring
        """
        self.gauth.Refresh()
        drive = GoogleDrive(self.gauth)
        self.drive = drive

    def __init__(self, model_tag, gauth=None, checkpoint_temp_id=None):

        self.model_tag = model_tag

        self.state = None

        self.best_evaluation_metric = 0

        # If gauth exists then exceute below
        if gauth:
            self.gauth = gauth
            self.refresh_gauth()
            self.checkpoint_temp_id = checkpoint_temp_id

            # code below is for uploading to Google Drive
            self.folders = self.drive.ListFile(
                {'q': f"'{checkpoint_temp_id}' in parents  \
                        and trashed = false \
                        and mimeType contains 'vnd.google-apps.folder' \
                        "}).GetList()
            # create subfolder for model
            model_folder = self.drive.CreateFile()
            model_folder['title'] = self.model_tag
            model_folder['parents'] = [
                {'id': self.checkpoint_temp_id}]  # assign parent folder
            # specify object as folder
            model_folder['mimeType'] = 'application/vnd.google-apps.folder'
            model_folder.Upload()  # upload to google drive
            model_folder_id = model_folder["id"]
            self.model_folder_id = model_folder_id

    # save the current checkpoint
    # note: this checkpoint is local and unique
    # note: the previous checkpoint is over-written

    def save_checkpoint(self, epoch, encoder_structural, encoder_cell_content, decoder_structural,
                        decoder_cell_content, encoder_structural_optimizer, encoder_cell_content_optimizer,
                        decoder_structural_optimizer, decoder_cell_content_optimizer, loss, loss_s, loss_cc):

        # assemble the state of the model
        state = {'model_tag': self.model_tag,
                 'epoch': epoch,
                 'encoder_structural': encoder_structural.state_dict(),
                 'encoder_cell_content': encoder_cell_content.state_dict(),
                 'decoder_structural': decoder_structural.state_dict(),
                 'decoder_cell_content': decoder_cell_content.state_dict(),
                 'encoder_structural_optimizer': encoder_structural_optimizer.state_dict(),
                 'encoder_cell_content_optimizer': encoder_cell_content_optimizer.state_dict(),
                 'decoder_structural_optimizer': decoder_structural_optimizer.state_dict(),
                 'decoder_cell_content_optimizer': decoder_cell_content_optimizer.state_dict(),
                 'loss': loss,
                 'loss_s': loss_s,
                 'loss_cc': loss_cc,
                 'time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H-%M-%S UTC')}

        self.state = state

        # store the checkpoint
        # this local checkpoint is over-witten
        # every time a new checkpoint is saved
        file_path = Utils.absolute_path('', 'checkpoint.pth.tar')
        torch.save(self.state, file_path)

    # archive the checkpoint into a sub-folder of 'Checkpoints/'
    def archive_checkpoint(self):

        # create archive folder if it does not exist
        folder_name = Utils.absolute_path(f'Checkpoints/{self.model_tag}', '')
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # create filename
        epoch = self.state['epoch']
        suffix = '{:0>3}'.format(epoch)
        file_path = Utils.absolute_path(
            folder_name, f'checkpoint_{suffix}.pth.tar')

        # archive the checkpoint
        torch.save(self.state, file_path)

    # archive the checkpoint, and label it as best_checkpoint,
    # if the evaluation metric is better than the last best evaluation metric
    def archive_checkpoint_if_best(self, evaluation_metric):

        # keep track of the best checkpoint
        if evaluation_metric > self.best_evaluation_metric:

            # create archive folder if it does not exist
            folder_name = Utils.absolute_path(
                f'Checkpoints/{self.model_tag}', '')
            if not os.path.isdir(folder_name):
                os.makedirs(folder_name)

            # create filename
            file_path = Utils.absolute_path(
                folder_name, 'best_checkpoint.pth.tar')

            # store the best checkpoint
            torch.save(self.state, file_path)

    def copy_checkpoint(self):

        # create filename
        epoch = self.state['epoch']
        suffix = '{:0>3}'.format(epoch)
        fn = f'checkpoint_{suffix}.pth.tar'

        self.refresh_gauth()
        checkpoint_gdrive = self.drive.CreateFile()
        checkpoint_gdrive['title'] = os.path.basename(fn)
        checkpoint_gdrive['parents'] = [{'id': self.model_folder_id}]
        checkpoint_gdrive.SetContentFile('checkpoint.pth.tar')
        checkpoint_gdrive.Upload()
