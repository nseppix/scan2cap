import torch
from torch import nn
import numpy as np
import torchvision

from lib.config import CONF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, vocab_list, embedding_dict, use_votenet, embed_dim=300, vote_dimension=128, encoder_dim=256, decoder_dim=512, dropout=0.5, objectness_thresh=0.5):
        """
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(Decoder, self).__init__()

        self.use_votenet = use_votenet
        self.vote_dimension = vote_dimension
        self.encoder_dim = encoder_dim 
        if self.use_votenet:
            self.encoder_dim += vote_dimension
        self.objectness_thresh = objectness_thresh
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.dropout = dropout
        

        self.vocab_size = len(vocab_list)
        embedding_dict["<end>"] = np.zeros(self.embed_dim)
        idx2embedding = []
        for word in vocab_list:
            if word in embedding_dict:
                embedding = embedding_dict[word]
            else:
                embedding = embedding_dict["unk"]
            idx2embedding.append(embedding)
        self.idx2embedding = nn.Parameter(torch.tensor(idx2embedding, dtype=torch.float32, requires_grad=False))
        self.initial_embedding = nn.Parameter(torch.tensor(embedding_dict["unk"], dtype=torch.float32, requires_grad=False).unsqueeze(0))

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)  # linear layer to find scores over vocabulary
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

        :param encoder_out: encoded images, a tensor of dimension (batch_size, encoder_dim)
        :return: hidden state, cell state
        """
        h = self.init_h(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(encoder_out)
        return h, c

    def forward(self, data_dict):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_scene_size)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        obj_features = data_dict["ref_obj_features"]

        batch_size = obj_features.shape[0]

        if self.use_votenet:
            objectness = torch.softmax(data_dict["objectness_scores"], dim=-1)[:, :, -1]
            object_mask = objectness > self.objectness_thresh
            has_objects = torch.any(object_mask, dim=1)
            vote_features = objectness.new_zeros((batch_size, self.vote_dimension))
            if torch.any(has_objects):
                vote_features[has_objects] = torch.stack([torch.mean(features_[object_mask_], dim=0) for features_, object_mask_ in zip(data_dict["aggregated_vote_features"][has_objects], object_mask[has_objects])])
            obj_features = torch.cat([vote_features, obj_features], dim=1)
        target_caption = data_dict["lang_indices"]
        target_caption_embeddings = self.idx2embedding[target_caption]
        target_caption_lengths = data_dict["lang_len"]
        batch_size = obj_features.size(0)

        if self.training:
            
            # Sort input data by decreasing lengths; why? apparent below
            # caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
            target_caption_lengths, sort_ind = torch.sort(target_caption_lengths, dim=0, descending=True)
            obj_features = obj_features[sort_ind]
            target_caption_embeddings = target_caption_embeddings[sort_ind]

            # Initialize LSTM state
            h, c = self.init_hidden_state(obj_features)  # (batch_size, decoder_dim)
            #preds = self.fc(self.dropout(h))
            #emb_h = self.vocabulary[preds.max(1)]
            # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
            # So, decoding lengths are actual lengths - 1
            decode_lengths = (target_caption_lengths).tolist()

            target_caption_embeddings = torch.cat([self.initial_embedding.unsqueeze(1).expand(batch_size, 1, -1), target_caption_embeddings], dim=1)

            # Create tensors to hold word predicion scores
            predictions = obj_features.new_zeros((batch_size, self.vocab_size, target_caption.size(1)))

            # At each time-step, decode by
            # then generate a new word in the decoder with the previous word
            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                h, c = self.decode_step(
                    torch.cat([target_caption_embeddings[:batch_size_t, t, :], obj_features[:batch_size_t,:]], dim=1),
                    (h[:batch_size_t,:], c[:batch_size_t,:]))  # (batch_size_t, decoder_dim)
                preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
                predictions[:batch_size_t, :, t] = preds

            #now we need to resort the output into its initial order
            _, orig_idx = sort_ind.sort(0)
            predictions_resorted = predictions[orig_idx]
            data_dict["caption_predictions"] = predictions_resorted

        else:
            # init hidden, cell state & prediction
            h, c = self.init_hidden_state(obj_features)

            # scores = self.fc(h)
            # wrd_idx = torch.argmax(scores, dim=1)
            # embedded_word = self.idx2embedding[wrd_idx]
            #
            # predictions = scores.new_zeros((batch_size, CONF.TRAIN.MAX_DES_LEN), dtype=torch.int64) - 1
            # predictions[:, 0] = wrd_idx
            # step = 1

            predictions = obj_features.new_zeros((batch_size, CONF.TRAIN.MAX_DES_LEN), dtype=torch.int64) - 1
            embedded_word = self.initial_embedding.expand(batch_size, -1)
            step = 0


            incomplete_indices = [i for i in range(batch_size)]

            while True:
                h, c = self.decode_step(
                    torch.cat([embedded_word, obj_features[incomplete_indices]], dim=1), (h,c))

                prediction = self.fc(h)
                wrd_idx = torch.argmax(prediction, dim=1)
                embedded_word = self.idx2embedding[wrd_idx]

                predictions[incomplete_indices, step] = wrd_idx

                keep_indices = wrd_idx != 0
                h = h[keep_indices]
                c = c[keep_indices]
                embedded_word = embedded_word[keep_indices]
                completed_indices = torch.nonzero(wrd_idx == 0).squeeze(-1).tolist()
                completed_indices = [incomplete_indices[i] for i in completed_indices]
                for c_i in completed_indices:
                    incomplete_indices.remove(c_i)

                step += 1
                if step >= CONF.TRAIN.MAX_DES_LEN or len(incomplete_indices) == 0: # we can think about this value
                    break

            data_dict["caption_indices"] = predictions
            scores = obj_features.new_zeros((batch_size, self.vocab_size, CONF.TRAIN.MAX_DES_LEN))

            # tmp = torch.tensor(predictions)
            tmp = predictions.clone().detach()
            tmp[tmp < 0] = 0
            scores.scatter_(1, tmp.unsqueeze(1), 1)

            data_dict["caption_predictions"] = scores

        return data_dict