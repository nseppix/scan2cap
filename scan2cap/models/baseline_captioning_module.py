import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, embed_dim=300, decoder_dim, vocab_size, encoder_dim=556, dropout=0.5):
        """
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(Decoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocabulary.size(0)
        self.dropout = dropout
        

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)


    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, data_dict, mode):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_scene_size)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        encoder_out = data_dict["ref_obj_features"]
        embedded_captions = data_dict["lang_feat"]
        caption_lengths = data_dict["lang_len"]
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        if mode == "train":
            
            # Sort input data by decreasing lengths; why? apparent below
            caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
            encoder_out = encoder_out[sort_ind]
            encoded_captions = embedded_captions[sort_ind]


            # Initialize LSTM state
            h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
            #preds = self.fc(self.dropout(h))
            #emb_h = self.vocabulary[preds.max(1)]
            # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
            # So, decoding lengths are actual lengths - 1
            decode_lengths = (caption_lengths - 1).tolist()

            # Create tensors to hold word predicion scores and alphas
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

            # At each time-step, decode by
            # then generate a new word in the decoder with the previous word
            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                
                h, c = self.decode_step(
                    torch.cat([embedded_captions[:batch_size_t, t, :], encoder_out[:batch_size_t], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
                preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
                predictions[:batch_size_t, t, :] = preds

            data_dict["caption_predictions"] = predictions
            data_dict["sort_ind_captioning"] = sort_ind
        
        else:
            # here I would do the evaluate forward

        return data_dict