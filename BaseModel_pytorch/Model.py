import torch

class Model:
    """Combined class for encoder, structural decoder and cell decoder."""

    def __init__(self,encoder, encoder_optimizer, decoder_structural,decoder_structural_optimizer,decoder_cell_content,decoder_cell_content_optimizer,structural_token2integer, structural_integer2token , cell_content_token2integer, cell_content_integer2token  ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
        self.encoder = encoder.to(self.device)
        self.encoder_optimizer = encoder_optimizer
        self.decoder_structural = decoder_structural.to(self.device)
        self.decoder_structural_optimizer = decoder_structural_optimizer
        self.decoder_cell_content = decoder_cell_content.to(self.device)
        self.decoder_cell_content_optimizer = decoder_cell_content_optimizer
        self.structural_token2integer = structural_token2integer
        self.structural_integer2token = structural_integer2token
        self.cell_content_token2integer = cell_content_token2integer
        self.cell_content_integer2token = cell_content_integer2token

    def set_eval(self):
        self.decoder_structural = self.decoder_structural.eval()
        self.decoder_cell_content = self.decoder_cell_content.eval()
        self.encoder = self.encoder.eval()

    def set_train(self):
        self.decoder_structural = self.decoder_structural.train()
        self.decoder_cell_content = self.decoder_cell_content.train()
        self.encoder = self.encoder.train()
