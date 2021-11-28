import os
import numpy as np
import tensorflow as tf
from models.ops import create_optimizer, count_params, regularizer
from models.layers import word_embedding_lookup, char_embedding_lookup, conv1d, video_query_attention, highlight_layer
from models.layers import context_query_concat, feature_encoder, conditioned_predictor, localization_loss, iou_regression
from models.layers import generate_proposal_boxes, dynamic_head
from models.layers import bilstm, multi_modal_sa, st_video_encoder, boundary_predictor, boundary_loss

class LPNet:
    def __init__(self, configs, graph):
        self.configs = configs
        graph = graph if graph is not None else tf.Graph()
        with graph.as_default():
            self.global_step = tf.train.create_global_step()
            self._add_placeholders()
            self._build_model()
            if configs.mode == 'train':
                print('\x1b[1;33m' + 'Total trainable parameters: {}'.format(count_params()) + '\x1b[0m', flush=True)
            else:
                print('\x1b[1;33m' + 'Total parameters: {}'.format(count_params()) + '\x1b[0m', flush=True)

    def _add_placeholders(self):
        self.video_inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, self.configs.video_feature_dim],
                                           name='video_inputs')
        self.video_seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name='video_sequence_length')
        self.word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_ids')
        self.char_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='char_ids')
        self.highlight_labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name='highlight_labels')
        
        self.is_training = tf.placeholder(tf.bool, shape=[])
        # self.dx1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='dx')
        # self.dy1 = tf.placeholder(dtype=tf.int32, shape=[None, None], name='dy')
        # self.mask1 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='batch_mask')

        self.y1 = tf.placeholder(dtype=tf.float32, shape=[None, None], name='start_indexes')
        self.y2 = tf.placeholder(dtype=tf.float32, shape=[None, None], name='end_indexes')
        # hyper-parameters
        self.drop_rate = tf.placeholder_with_default(input=0.0, shape=[], name='dropout_rate')
        # create mask
        self.v_mask = tf.sequence_mask(lengths=self.video_seq_length, maxlen=tf.reduce_max(self.video_seq_length),
                                       dtype=tf.int32)
        self.q_mask = tf.cast(tf.cast(self.word_ids, dtype=tf.bool), dtype=tf.int32)

        self.q_length = tf.reduce_sum(self.q_mask, -1)
        self.v_length = tf.reduce_sum(self.v_mask, -1)

    def _build_model(self):
        # word embedding & visual features
        init_word_vectors = np.load(os.path.join(self.configs.save_dir, 'word_vectors.npz'))['vectors']
        word_emb = word_embedding_lookup(self.word_ids, dim=self.configs.word_dim, drop_rate=self.drop_rate,
                                         vectors=init_word_vectors, finetune=False, reuse=False, name='word_embeddings')
        char_emb = char_embedding_lookup(self.char_ids, char_size=self.configs.char_size, dim=self.configs.char_dim,
                                         kernels=[1, 2, 3, 4], filters=[10, 20, 30, 40], drop_rate=self.drop_rate,
                                         activation=tf.nn.relu, reuse=False, name='char_embeddings')
        word_emb = tf.concat([word_emb, char_emb], axis=-1)
        video_features = tf.nn.dropout(self.video_inputs, rate=self.drop_rate)

        # feature projection (map both word and video feature to the same dimension)
        vfeats = conv1d(video_features, dim=self.configs.hidden_size, use_bias=True, reuse=False, name='video_conv1d')
        qfeats = conv1d(word_emb, dim=self.configs.hidden_size, use_bias=True, reuse=False, name='query_conv1d')

        vfeats0 = feature_encoder(vfeats, hidden_size=self.configs.hidden_size, num_heads=self.configs.num_heads,
                                  max_position_length=self.configs.max_position_length, drop_rate=self.drop_rate,
                                  mask=self.v_mask, reuse=False, name='feature_encoder')
        qfeats0 = feature_encoder(qfeats, hidden_size=self.configs.hidden_size, num_heads=self.configs.num_heads,
                                 max_position_length=self.configs.max_position_length, drop_rate=self.drop_rate,
                                 mask=self.q_mask, reuse=True, name='feature_encoder')

        # # video query attention
        outputs, self.vq_score = video_query_attention(vfeats0, qfeats0, self.v_mask, self.q_mask, reuse=False,
                                                       drop_rate=self.drop_rate, name='video_query_attention')

        # # weighted pooling and concatenation
        outputs0 = context_query_concat(outputs, qfeats0, q_mask=self.q_mask, reuse=False, name='context_query_concat')

        self.highlight_loss, self.highlight_scores = highlight_layer(outputs0, self.highlight_labels, mask=self.v_mask,
                                                                     reuse=False, name='highlighting_layer')
        outputs0 = tf.multiply(outputs0, tf.expand_dims(self.highlight_scores, axis=-1))

        start_logits, end_logits = conditioned_predictor(outputs0, hidden_size=self.configs.hidden_size,
                                                         seq_len=self.video_seq_length, mask=self.v_mask,
                                                       reuse=False, name='conditioned_predictor')
        # compute localization loss
        self.start_prob, self.end_prob, self.start_index, self.end_index, self.loss = boundary_loss(
                                                    start_logits, end_logits, self.y1, self.y2, self.v_length,self.configs)


        self.proposal_box, self.dx, self.dy, self.boxes = generate_proposal_boxes(vfeats0, self.v_length, self.configs)
        self.reg_loss, self.l1reg_loss, self.l1_loss, self.iou_loss, self.regular, self.regular2, self.train, abc, self.px, self.py = dynamic_head(
            configs=self.configs,
            proposal_box=self.proposal_box,
            v_mask=self.v_mask,
            q_mask=self.q_mask,
            vfeats0=outputs,
            qfeats0=qfeats0,
            drop_rate=self.drop_rate,
            dx=self.dx,
            dy=self.dy,
            boxes=self.boxes,
            y1=self.y1,
            y2=self.y2,
            train=self.is_training)

        self.my_loss = 5 * self.iou_loss + self.regular   #+ self.l1_loss  #+ 100*self.reg_loss + self.highlight_loss + 0.2*self.loss#+ self.regular + self.regular2 + self.l1_loss
        self.reg_loss = 100 * self.reg_loss + self.highlight_loss + self.loss