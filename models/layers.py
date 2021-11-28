import math
from random import seed
import numpy as np
import tensorflow as tf
import scipy.signal as signal
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.gen_math_ops import arg_min
from models.ops import get_shape_list, mask_logits, regularizer, trilinear_attention
from utils.runner_utils import calculate_iou
from utils.data_utils import pad_video_sequence


def layer_norm(inputs, epsilon=1e-6, reuse=None, name='layer_norm'):
    """Layer normalize the tensor x, averaging over the last dimension."""
    with tf.variable_scope(name,
                           default_name="layer_norm",
                           values=[inputs],
                           reuse=reuse):
        dim = get_shape_list(inputs)[-1]

        scale = tf.get_variable("layer_norm_scale", [dim],
                                regularizer=regularizer,
                                initializer=tf.ones_initializer())
        bias = tf.get_variable("layer_norm_bias", [dim],
                               regularizer=regularizer,
                               initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean),
                                  axis=[-1],
                                  keep_dims=True)
        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)
        result = norm_inputs * scale + bias

        return result


def word_embedding_lookup(word_ids,
                          dim,
                          vectors,
                          drop_rate=0.0,
                          finetune=False,
                          reuse=None,
                          name='word_embeddings'):
    with tf.variable_scope(name, reuse=reuse):
        table = tf.Variable(vectors,
                            name='word_table',
                            dtype=tf.float32,
                            trainable=finetune)
        unk = tf.get_variable(name='unk',
                              shape=[1, dim],
                              dtype=tf.float32,
                              trainable=True)
        zero = tf.zeros(shape=[1, dim], dtype=tf.float32)

        word_table = tf.concat([zero, unk, table], axis=0)
        word_emb = tf.nn.embedding_lookup(word_table, word_ids)
        word_emb = tf.nn.dropout(word_emb, rate=drop_rate)

        return word_emb


def char_embedding_lookup(char_ids,
                          char_size,
                          dim,
                          kernels,
                          filters,
                          drop_rate=0.0,
                          activation=tf.nn.relu,
                          padding='VALID',
                          reuse=None,
                          name='char_embeddings'):
    with tf.variable_scope(name, reuse=reuse):
        # char embeddings lookup
        table = tf.get_variable(name='char_table',
                                shape=[char_size - 1, dim],
                                dtype=tf.float32,
                                trainable=True)
        zero = tf.zeros(shape=[1, dim], dtype=tf.float32)
        char_table = tf.concat([zero, table], axis=0)
        char_emb = tf.nn.embedding_lookup(char_table, char_ids)
        char_emb = tf.nn.dropout(char_emb, rate=drop_rate)

        # char-level cnn
        outputs = []
        for i, (kernel, channel) in enumerate(zip(kernels, filters)):
            weight = tf.get_variable('filter_%d' % i,
                                     shape=[1, kernel, dim, channel],
                                     regularizer=regularizer,
                                     dtype=tf.float32)

            bias = tf.get_variable('bias_%d' % i,
                                   shape=[channel],
                                   regularizer=regularizer,
                                   dtype=tf.float32,
                                   initializer=tf.zeros_initializer())

            output = tf.nn.conv2d(char_emb,
                                  weight,
                                  strides=[1, 1, 1, 1],
                                  padding=padding,
                                  name='conv_%d' % i)

            output = tf.nn.bias_add(output, bias=bias)
            output = tf.reduce_max(activation(output), axis=2)
            outputs.append(output)

        outputs = tf.concat(values=outputs, axis=-1)

        return outputs


def conv1d(inputs,
           dim,
           kernel_size=1,
           use_bias=False,
           activation=None,
           padding='VALID',
           reuse=None,
           name='conv1d'):
    with tf.variable_scope(name, reuse=reuse):
        shapes = get_shape_list(inputs)
        kernel = tf.get_variable(name='kernel',
                                 shape=[kernel_size, shapes[-1], dim],
                                 dtype=tf.float32,
                                 regularizer=regularizer)

        outputs = tf.nn.conv1d(inputs,
                               filters=kernel,
                               stride=1,
                               padding=padding)

        if use_bias:
            bias = tf.get_variable(name='bias',
                                   shape=[1, 1, dim],
                                   dtype=tf.float32,
                                   regularizer=regularizer,
                                   initializer=tf.zeros_initializer())

            outputs += bias

        if activation is not None:
            return activation(outputs)

        else:
            return outputs


def conv1d_(inputs,
           dim,
           kernel_size=1,
           stride=1,
           use_bias=False,
           activation=None,
           padding='VALID',
           reuse=None,
           name='conv1d_'):
    with tf.variable_scope(name, reuse=reuse):
        shapes = get_shape_list(inputs)
        kernel = tf.get_variable(name='kernel_',
                                 shape=[kernel_size, shapes[-1], dim],
                                 dtype=tf.float32,
                                 regularizer=regularizer)

        outputs = tf.nn.conv1d(inputs,
                               filters=kernel,
                               stride=stride,
                               padding=padding)

        if use_bias:
            bias = tf.get_variable(name='bias_',
                                   shape=[1, 1, dim],
                                   dtype=tf.float32,
                                   regularizer=regularizer,
                                   initializer=tf.zeros_initializer())

            outputs += bias

        if activation is not None:
            return activation(outputs)

        else:
            return outputs

def depthwise_separable_conv(inputs,
                             kernel_size,
                             dim,
                             use_bias=True,
                             reuse=None,
                             activation=tf.nn.relu,
                             name='depthwise_separable_conv'):
    with tf.variable_scope(name, reuse=reuse):
        shapes = get_shape_list(inputs)

        depthwise_filter = tf.get_variable(
            name='depthwise_filter',
            dtype=tf.float32,
            regularizer=regularizer,
            shape=[kernel_size[0], kernel_size[1], shapes[-1], 1])  #[7, 1, d, 1]

        pointwise_filter = tf.get_variable(name='pointwise_filter',
                                           shape=[1, 1, shapes[-1], dim],
                                           dtype=tf.float32,
                                           regularizer=regularizer)  #[1, 1, d, d]

        outputs = tf.nn.separable_conv2d(inputs,
                                         depthwise_filter,
                                         pointwise_filter,
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')

        if use_bias:
            b = tf.get_variable('bias',
                                outputs.shape[-1],
                                regularizer=regularizer,
                                initializer=tf.zeros_initializer())
            outputs += b

        outputs = activation(outputs)

        return outputs


def add_positional_embedding(inputs,
                             max_position_length,
                             reuse=None,
                             name='positional_embedding'):
    with tf.variable_scope(name, reuse=reuse):
        batch_size, seq_length, dim = get_shape_list(inputs)
        assert_op = tf.assert_less_equal(seq_length, max_position_length)

        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(
                name='position_embeddings',
                shape=[max_position_length, dim],
                dtype=tf.float32)

            position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                           [seq_length, -1])
            num_dims = len(inputs.shape.as_list())
            position_broadcast_shape = []

            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)

            position_broadcast_shape.extend([seq_length, dim])
            position_embeddings = tf.reshape(position_embeddings,
                                             shape=position_broadcast_shape)
            outputs = inputs + position_embeddings

        return outputs


def conv_block(inputs,
               kernel_size,
               dim,
               num_layers,
               drop_rate=0.0,
               reuse=None,
               name='conv_block'):
    with tf.variable_scope(name, reuse=reuse):
        outputs = tf.expand_dims(inputs, axis=2)
        for layer_idx in range(num_layers):
            residual = outputs
            outputs = layer_norm(outputs,
                                 reuse=reuse,
                                 name='layer_norm_conv_%d' % layer_idx)

            outputs = depthwise_separable_conv(
                outputs,
                kernel_size=(kernel_size, 1),
                dim=dim,
                use_bias=True,
                activation=tf.nn.relu,
                name='depthwise_conv_layers_%d' % layer_idx,
                reuse=reuse)

            outputs = tf.nn.dropout(outputs, rate=drop_rate) + residual
        return tf.squeeze(outputs, 2)


def multihead_attention(inputs,
                        dim,
                        num_heads,
                        mask=None,
                        drop_rate=0.0,
                        reuse=None,
                        name='multihead_attention'):
    with tf.variable_scope(name, reuse=reuse):
        if dim % num_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the attention heads (%d)'
                % (dim, num_heads))
        batch_size, seq_length, _ = get_shape_list(inputs)
        head_size = dim // num_heads

        def transpose_for_scores(input_tensor, batch_size_, seq_length_,
                                 num_heads_, head_size_):
            output_tensor = tf.reshape(
                input_tensor,
                shape=[batch_size_, seq_length_, num_heads_, head_size_])
            output_tensor = tf.transpose(output_tensor, perm=[0, 2, 1, 3])
            return output_tensor

        # projection
        query = conv1d(inputs,
                       dim=dim,
                       use_bias=True,
                       reuse=reuse,
                       name='query')
        key = conv1d(inputs, dim=dim, use_bias=True, reuse=reuse, name='key')
        value = conv1d(inputs,
                       dim=dim,
                       use_bias=True,
                       reuse=reuse,
                       name='value')

        # reshape & transpose: (batch_size, seq_length, dim) --> (batch_size, num_heads, seq_length, head_size)
        query = transpose_for_scores(query, batch_size, seq_length, num_heads,
                                     head_size)
        key = transpose_for_scores(key, batch_size, seq_length, num_heads,
                                   head_size)
        value = transpose_for_scores(value, batch_size, seq_length, num_heads,
                                     head_size)

        # compute attention score
        query = tf.multiply(query, 1.0 / math.sqrt(float(head_size)))
        attention_score = tf.matmul(query, key, transpose_b=True)

        if mask is not None:
            shapes = get_shape_list(attention_score)
            mask = tf.cast(tf.reshape(mask,
                                      shape=[shapes[0], 1, 1, shapes[-1]]),
                           dtype=tf.float32)
            attention_score += (1.0 - mask) * -1e30

        attention_score = tf.nn.softmax(
            attention_score
        )  # shape = (batch_size, num_heads, seq_length, seq_length)
        attention_score = tf.nn.dropout(attention_score, rate=drop_rate)

        # compute value
        value = tf.matmul(
            attention_score,
            value)  # shape = (batch_size, num_heads, seq_length, head_size)
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        value = tf.reshape(
            value, shape=[batch_size, seq_length, num_heads * head_size])

        return value


def multihead_attention_block(inputs,
                              dim,
                              num_heads,
                              mask=None,
                              use_bias=True,
                              drop_rate=0.0,
                              reuse=None,
                              name='multihead_attention_block'):
    with tf.variable_scope(name, reuse=reuse):
        # multihead attention layer
        outputs = layer_norm(inputs, reuse=reuse, name='layer_norm_1')
        outputs = tf.nn.dropout(outputs, rate=drop_rate)

        outputs = multihead_attention(outputs,
                                      dim=dim,
                                      num_heads=num_heads,
                                      mask=mask,
                                      drop_rate=drop_rate,
                                      name='multihead_attention')

        outputs = tf.nn.dropout(outputs, rate=drop_rate)
        residual = outputs + inputs

        # feed forward layer
        outputs = layer_norm(residual, reuse=reuse, name='layer_norm_2')
        outputs = tf.nn.dropout(outputs, rate=drop_rate)

        outputs = conv1d(outputs,
                         dim=dim,
                         use_bias=use_bias,
                         activation=None,
                         reuse=reuse,
                         name='dense')

        outputs = tf.nn.dropout(outputs, rate=drop_rate)
        outputs = outputs + residual

        return outputs

def multi_modal_sa(vfeats, qfeats, vmask, qmask, qlength, reuse=None, name='multi_modal_self_attention'):
    with tf.variable_scope(name, reuse=reuse):
        b, lv, dv = get_shape_list(vfeats)
        b, lq, dq = get_shape_list(qfeats)

        vfeats = tf.expand_dims(vfeats, 1)
        qfeats = tf.expand_dims(qfeats, 2)

        vfeats = tf.tile(vfeats, [1, lq, 1, 1])
        qfeats = tf.tile(qfeats, [1, 1, lv, 1])

        outputs = tf.reshape(tf.concat([vfeats, qfeats], -1), [b*lq, lv, dv+dq])

        vmask = tf.tile(tf.expand_dims(vmask, 1), [1, lq, 1])

        outputs = conv1d(outputs, dim=512, use_bias=True, reuse=False, name='mm_dense')
        outputs = multihead_attention_block(outputs, 512, num_heads=1, mask=vmask, use_bias=True, drop_rate=0.0, reuse=None, name='multihead_attention_block')

        outputs = tf.reshape(outputs, [b, lq, lv, -1])

        qmask = tf.cast(tf.expand_dims(tf.tile(tf.expand_dims(qmask, -1), [1, 1, lv]), -1), tf.float32)

        outputs = qmask * outputs #[b, lq, lv, d]

        # outputs = tf.reduce_sum(outputs, 1) #[b, lv, d]
        outputs = tf.reduce_max(outputs, 1)
        # qlength = tf.tile(tf.reshape(qlength, [b, 1, 1]), [1, lv, 1])
        # outputs = tf.divide(outputs, tf.cast(qlength, tf.float32))
        outputs = tf.reshape(outputs, [b, lv, 512])

        outputs = conv1d(outputs, dim=dv, use_bias=True, reuse=False, name='mmsa_dense')
        return outputs


def feature_encoder(inputs,
                    hidden_size,
                    num_heads,
                    max_position_length,
                    drop_rate,
                    mask,
                    reuse=None,
                    name='feature_encoder'):
    with tf.variable_scope(name, reuse=reuse):
        features = add_positional_embedding(
            inputs,
            max_position_length=max_position_length,
            reuse=reuse,
            name='positional_embedding')

        features = conv_block(features,
                              kernel_size=7,
                              dim=hidden_size,
                              num_layers=4,
                              reuse=reuse,
                              drop_rate=drop_rate,
                              name='conv_block')
        features = multihead_attention_block(features,
                                             dim=hidden_size,
                                             num_heads=num_heads,
                                             mask=mask,
                                             use_bias=True,
                                             drop_rate=drop_rate,
                                             reuse=False,
                                             name='multihead_attention_block')

        return features

def st_video_encoder(inputs,
                    hidden_size,
                    num_heads,
                    max_position_length,
                    drop_rate,
                    mask,
                    reuse=None,
                    name='st_video_encoder'):
    with tf.variable_scope(name, reuse=reuse):
        
        features = tf.reshape(inputs, [-1, 16, 16])
        features = multihead_attention_block(features,
                                            dim=16,
                                            num_heads=num_heads,
                                            mask=None,
                                            use_bias=True,
                                            drop_rate=drop_rate,
                                            reuse=False,
                                            name='multihead_attention_block_spatial')
        features = tf.reshape(features, tf.shape(inputs))

        features = add_positional_embedding(
            features,
            max_position_length=max_position_length,
            reuse=reuse,
            name='positional_embedding')

        features = conv_block(features,
                              kernel_size=7,
                              dim=hidden_size,
                              num_layers=4,
                              reuse=reuse,
                              drop_rate=drop_rate,
                              name='conv_block')
        features = multihead_attention_block(features,
                                             dim=hidden_size,
                                             num_heads=num_heads,
                                             mask=mask,
                                             use_bias=True,
                                             drop_rate=drop_rate,
                                             reuse=False,
                                             name='multihead_attention_block')

        return features
def video_query_attention(video_features,
                          query_features,
                          v_mask,
                          q_mask,
                          drop_rate=0.0,
                          reuse=None,
                          name='video_query_attention'):
    with tf.variable_scope(name, reuse=reuse):
        dim = get_shape_list(video_features)[-1]
        v_maxlen = tf.reduce_max(tf.reduce_sum(v_mask, axis=1))
        q_maxlen = tf.reduce_max(tf.reduce_sum(q_mask, axis=1))

        score = trilinear_attention([video_features, query_features],
                                    v_maxlen=v_maxlen,
                                    q_maxlen=q_maxlen,
                                    drop_rate=drop_rate,
                                    reuse=reuse,
                                    name='efficient_trilinear')

        mask_q = tf.expand_dims(q_mask, 1)
        score_ = tf.nn.softmax(mask_logits(score, mask=mask_q))

        mask_v = tf.expand_dims(v_mask, 2)
        score_t = tf.transpose(tf.nn.softmax(mask_logits(score, mask=mask_v),
                                             dim=1),
                               perm=[0, 2, 1])

        v2q = tf.matmul(score_, query_features)
        q2v = tf.matmul(tf.matmul(score_, score_t), video_features)

        attention_outputs = tf.concat(
            [video_features, v2q, video_features * v2q, video_features * q2v],
            axis=-1)
        outputs = conv1d(attention_outputs,
                         dim=dim,
                         use_bias=False,
                         activation=None,
                         reuse=reuse,
                         name='dense')
        return outputs, score


def video_query_attention_(video_features,
                           query_features,
                           v_len,
                           q_mask,
                           drop_rate=0.0,
                           reuse=None,
                           name='video_query_attention_'):
    with tf.variable_scope(name, reuse=reuse):
        dim = get_shape_list(video_features)[-1]
        d = get_shape_list(video_features)[0]
        v_mask = tf.ones([d, v_len])
        v_maxlen = tf.reduce_max(tf.reduce_sum(v_mask, axis=1))
        q_maxlen = tf.reduce_max(tf.reduce_sum(q_mask, axis=1))

        score = trilinear_attention([video_features, query_features],
                                    v_maxlen=v_maxlen,
                                    q_maxlen=q_maxlen,
                                    drop_rate=drop_rate,
                                    reuse=reuse,
                                    name='efficient_trilinear')

        mask_q = tf.expand_dims(q_mask, 1)
        score_ = tf.nn.softmax(mask_logits(score, mask=mask_q))

        mask_v = tf.expand_dims(v_mask, 2)
        score_t = tf.transpose(tf.nn.softmax(mask_logits(score, mask=mask_v),
                                             dim=1),
                               perm=[0, 2, 1])

        v2q = tf.matmul(score_, query_features)
        q2v = tf.matmul(tf.matmul(score_, score_t), video_features)

        attention_outputs = tf.concat(
            [video_features, v2q, video_features * v2q, video_features * q2v],
            axis=-1)
        outputs = conv1d(attention_outputs,
                         dim=dim,
                         use_bias=False,
                         activation=None,
                         reuse=reuse,
                         name='dense')
        return outputs, score


def context_query_concat(inputs,
                         qfeats,
                         q_mask,
                         reuse=None,
                         name='context_query_concat'):
    with tf.variable_scope(name, reuse=reuse):
        dim = get_shape_list(qfeats)[-1]

        # compute pooled query feature
        weight = tf.get_variable(name='weight',
                                 shape=[dim, 1],
                                 dtype=tf.float32,
                                 regularizer=regularizer)
        x = tf.tensordot(qfeats, weight,
                         axes=1)  # shape = (batch_size, seq_length, 1)
        q_mask = tf.expand_dims(q_mask,
                                axis=-1)  # shape = (batch_size, seq_length, 1)
        x = mask_logits(x, mask=q_mask)
        alphas = tf.nn.softmax(x, axis=1)
        q_pooled = tf.matmul(tf.transpose(qfeats, perm=[0, 2, 1]), alphas)
        q_pooled = tf.squeeze(q_pooled, axis=-1)  # shape = (batch_size, dim)

        # concatenation
        q_pooled = tf.tile(tf.expand_dims(q_pooled, axis=1),
                           multiples=[1, tf.shape(inputs)[1], 1])
        outputs = tf.concat([inputs, q_pooled], axis=-1)
        outputs = conv1d(outputs,
                         dim=dim,
                         use_bias=True,
                         reuse=False,
                         name='dense')
        return outputs

def context_query_position_concat(inputs,
                         qfeats,
                         position,
                         q_mask,
                         reuse=None,
                         name='context_query_position_concat'):
    with tf.variable_scope(name, reuse=reuse):
        dim = get_shape_list(qfeats)[-1]
        batch_size = get_shape_list(qfeats)[0]

        # compute pooled query feature
        weight = tf.get_variable(name='weight',
                                 shape=[dim, 1],
                                 dtype=tf.float32,
                                 regularizer=regularizer)
        x = tf.tensordot(qfeats, weight,
                         axes=1)  # shape = (batch_size, seq_length, 1)
        q_mask = tf.expand_dims(q_mask,
                                axis=-1)  # shape = (batch_size, seq_length, 1)
        x = mask_logits(x, mask=q_mask)
        alphas = tf.nn.softmax(x, axis=1)
        q_pooled = tf.matmul(tf.transpose(qfeats, perm=[0, 2, 1]), alphas)
        q_pooled = tf.squeeze(q_pooled, axis=-1)  # shape = (batch_size, dim)

        # concatenation
        q_pooled = tf.tile(tf.expand_dims(q_pooled, axis=1),
                           multiples=[1, tf.shape(inputs)[1], 1])

        # inputs = inputs + position

        # position = tf.tile(tf.expand_dims(position, 0), [batch_size, 1, 1])
        # outputs = tf.concat([inputs, position, q_pooled], axis=-1)
        # outputs = tf.concat([inputs, q_pooled], axis=-1)

        # outputs = conv1d(outputs,
        #                  dim=dim,
        #                  use_bias=True,
        #                  reuse=False,
        #                  name='dense')
        outputs = tf.multiply(inputs, q_pooled)
        outputs = tf.nn.l2_normalize(outputs, dim=-1, name='l2_norm')
        outputs = conv1d(outputs,
                         dim=dim,
                         use_bias=True,
                         reuse=False,
                         name='dense')
        return outputs

def pooling_query(qfeats, q_mask, reuse=None, name='pooling_query'):
    with tf.variable_scope(name, reuse=reuse):
        dim = get_shape_list(qfeats)[-1]

        # compute pooled query feature
        weight = tf.get_variable(name='weight',
                                 shape=[dim, 1],
                                 dtype=tf.float32,
                                 regularizer=regularizer)
        x = tf.tensordot(qfeats, weight,
                         axes=1)  # shape = (batch_size, seq_length, 1)
        q_mask = tf.expand_dims(q_mask,
                                axis=-1)  # shape = (batch_size, seq_length, 1)
        x = mask_logits(x, mask=q_mask)
        alphas = tf.nn.softmax(x, axis=1)
        q_pooled = tf.matmul(tf.transpose(qfeats, perm=[0, 2, 1]), alphas)
        q_pooled = tf.squeeze(q_pooled, axis=-1)  # shape = (batch_size, dim)

        return q_pooled


def highlight_layer(inputs,
                    labels,
                    mask,
                    epsilon=1e-12,
                    reuse=None,
                    name='highlight_layer'):
    with tf.variable_scope(name, reuse=reuse):
        logits = conv1d(inputs,
                        dim=1,
                        use_bias=True,
                        padding='VALID',
                        reuse=reuse,
                        name='dense')
        logits = tf.squeeze(logits, axis=-1)  # (batch_size, seq_length)
        logits = mask_logits(logits, mask=mask)

        # prepare labels and weights
        labels = tf.cast(labels, dtype=logits.dtype)
        weights = tf.where(tf.equal(labels, 0.0),
                           x=labels + 1.0,
                           y=labels * 2.0)
        
        labels = tf.clip_by_value(labels, 1e-7, 1.0)

        # binary cross entropy with sigmoid activation
        loss_per_location = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        loss_per_location = loss_per_location * weights
        mask = tf.cast(mask, dtype=logits.dtype)
        loss = tf.reduce_sum(
            loss_per_location * mask) / (tf.reduce_sum(mask) + epsilon)

        # compute scores
        scores = tf.sigmoid(logits)
        return loss, scores


def dynamic_rnn(inputs, seq_len, dim, reuse=None, name='dynamic_rnn'):
    with tf.variable_scope(name, reuse=reuse):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=dim,
                                       use_peepholes=False,
                                       name='lstm_cell')
        outputs, _ = tf.nn.dynamic_rnn(cell,
                                       inputs,
                                       sequence_length=seq_len,
                                       dtype=tf.float32)
        return outputs

def bilstm(inputs, seq_len, dim, reuse=None, name='bilstm'):

    with tf.variable_scope(name, reuse=reuse):
        fwd_cell = tf.nn.rnn_cell.LSTMCell(num_units=dim,
                                       use_peepholes=False,
                                       name='fwd_cell')
        bwd_cell = tf.nn.rnn_cell.LSTMCell(num_units=dim,
                                       use_peepholes=False,
                                       name='bwd_cell')
        h, _ = tf.nn.bidirectional_dynamic_rnn(fwd_cell, bwd_cell, inputs, sequence_length=seq_len, dtype=tf.float32)

        outputs = tf.concat(h, -1)
        outputs = conv1d(outputs, dim, 1, use_bias=True, activation=tf.nn.relu, name='bilstm_dense')
        return outputs

def conditioned_predictor(inputs,
                          hidden_size,
                          seq_len,
                          mask,
                          reuse=None,
                          name='conditioned_predictor'):
    with tf.variable_scope(name, reuse=reuse):
        start_features = dynamic_rnn(inputs,
                                     seq_len,
                                     dim=hidden_size,
                                     reuse=False,
                                     name='start_rnn')
        end_features = dynamic_rnn(start_features,
                                   seq_len,
                                   dim=hidden_size,
                                   reuse=False,
                                   name='end_rnn')

        start_features = conv1d(tf.concat([start_features, inputs], axis=-1),
                                dim=hidden_size,
                                use_bias=True,
                                reuse=False,
                                activation=tf.nn.relu,
                                name='start_hidden')

        end_features = conv1d(tf.concat([end_features, inputs], axis=-1),
                              dim=hidden_size,
                              use_bias=True,
                              reuse=False,
                              activation=tf.nn.relu,
                              name='end_hidden')

        start_logits = conv1d(start_features,
                              dim=1,
                              use_bias=True,
                              reuse=reuse,
                              name='start_dense')
        end_logits = conv1d(end_features,
                            dim=1,
                            use_bias=True,
                            reuse=reuse,
                            name='end_dense')

        start_logits = mask_logits(
            tf.squeeze(start_logits,
                       axis=-1), mask=mask)  # shape = (batch_size, seq_length)
        end_logits = mask_logits(tf.squeeze(end_logits, axis=-1),
                                 mask=mask)  # shape = (batch_size, seq_length)

        return start_logits, end_logits

def boundary_predictor(inputs,
                          hidden_size,
                          seq_len,
                          mask,
                          reuse=None,
                          name='boundary_predictor'):
    with tf.variable_scope(name, reuse=reuse):

        start_features = conv1d(inputs,
                                dim=hidden_size,
                                use_bias=True,
                                reuse=False,
                                activation=tf.nn.relu,
                                name='start_hidden')

        end_features = conv1d(inputs,
                              dim=hidden_size,
                              use_bias=True,
                              reuse=False,
                              activation=tf.nn.relu,
                              name='end_hidden')

        start_logits = conv1d(start_features,
                              dim=1,
                              use_bias=True,
                              reuse=reuse,
                              name='start_dense')
        end_logits = conv1d(end_features,
                            dim=1,
                            use_bias=True,
                            reuse=reuse,
                            name='end_dense')

        start_logits = mask_logits(
            tf.squeeze(start_logits,
                       axis=-1), mask=mask)  # shape = (batch_size, seq_length)
        end_logits = mask_logits(tf.squeeze(end_logits, axis=-1),
                                 mask=mask)  # shape = (batch_size, seq_length)

        return start_logits, end_logits

def pad_video_sequence(sequences, max_length=256):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])

    feature_length = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []

    for i in range(32):
        seq = sequences[i]
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])

        if add_length > 0:
            add_feature = np.zeros(shape=[add_length, feature_length],
                                   dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)

        else:
            seq_ = seq

        sequence_padded.append(seq_)

    return sequence_padded, sequence_length


def iou_regression(configs,
                   v_mask,
                   q_mask,
                   vfeats0,
                   qfeats0,
                   drop_rate,
                   mask,
                   dx,
                   dy,
                   y1,
                   y2,
                   reuse=None,
                   name='iou_regression'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        outputs = tf.expand_dims(vfeats0, 1)
        # mask = tf.expand_dims(mask, -1)

        selected_feature = tf.multiply(mask, outputs)
        # print(get_shape_list(selected_feature))
        shape = get_shape_list(selected_feature)
        selected_feature = conv1d(tf.reshape(selected_feature,
                                             [-1, shape[-2], shape[-1]]),
                                  dim=shape[-1],
                                  use_bias=True,
                                  reuse=False,
                                  name='dense')
        selected_feature = tf.reshape(selected_feature, shape)
        selected_feature_reduce = tf.reduce_sum(selected_feature, 2)
        mask_reduce = tf.reduce_sum(mask, 2)
        inputs = tf.div(selected_feature_reduce, mask_reduce)

        x_index = tf.cast(tf.reshape(tf.arg_max(y1, 1), [-1, 1]), tf.int32)
        y_index = tf.cast(tf.reshape(tf.arg_max(y2, 1), [-1, 1]), tf.int32)

        union = (tf.minimum(dx, x_index), tf.maximum(dy, y_index))

        inter = (tf.maximum(dx, x_index), tf.minimum(dy, y_index))

        iou = tf.maximum(
            tf.cast((inter[1] - inter[0]), tf.float32) / tf.cast(
                (union[1] - union[0]), tf.float32), 0.0)

        output = tf.layers.dense(inputs, 128, activation=tf.nn.relu)
        logits = tf.squeeze(
            tf.layers.dense(output, 1, activation=tf.nn.sigmoid), -1)

        reg_loss = tf.losses.mean_squared_error(labels=iou, predictions=logits)

        p_x = tf.batch_gather(dx, tf.expand_dims(tf.arg_max(logits, -1), -1))
        p_y = tf.batch_gather(dy, tf.expand_dims(tf.arg_max(logits, -1), -1))

        offset_loss = 0

        return reg_loss, offset_loss, [dx, dy], p_x, p_y


def smooth_l1_loss(labels, predictions, scope=tf.GraphKeys.LOSSES):
    """smooth_l1_loss

    Args:
        labels ([type]): [description]
        predictions ([type]): [description]
        scope ([type], optional): [description]. Defaults to tf.GraphKeys.LOSSES.

    Returns:
        [type]: [description]
    """
    with tf.variable_scope(scope):
        diff = tf.abs(labels - predictions)
        less_than_one = tf.cast(tf.less(diff, 0.5),
                                tf.float32)  # Bool to float32
        smooth_l1_loss = (less_than_one * diff ** 2) + \
            (1.0 - less_than_one) * (diff - 0.25)  # 同上图公式
        return tf.reduce_mean(smooth_l1_loss)  # 取平均值


def localization_loss(start_logits, end_logits, y1, y2, configs):

    start_prob = tf.nn.softmax(start_logits, axis=1)
    end_prob = tf.nn.softmax(end_logits, axis=1)

    outer = tf.matmul(tf.expand_dims(start_prob, axis=2),
                      tf.expand_dims(end_prob, axis=1))
    outer = tf.matrix_band_part(outer, num_lower=0, num_upper=-1)

    batch_size = tf.shape(outer)[0]
    length = tf.shape(outer)[1]
    outer_reshape = tf.reshape(outer, [batch_size, -1])
    if configs.mode == "train":
        k = 16
    elif configs.mode == "test":
        k = 16
    topk, topk_index = tf.nn.top_k(outer_reshape, k, sorted=False)
    dx = tf.div(topk_index, length)
    dy = tf.mod(topk_index, length)

    mask_dx = tf.sequence_mask(lengths=dx, maxlen=length, dtype=tf.float32)
    mask_dy = tf.sequence_mask(lengths=dy + 1, maxlen=length, dtype=tf.float32)
    mask = mask_dy - mask_dx

    starts = tf.reduce_max(outer, axis=2)
    ends = tf.reduce_max(outer, axis=1)
    start_index = tf.argmax(starts, axis=1)
    end_index = tf.argmax(ends, axis=1)

    start_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=start_logits, labels=y1)
    end_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=end_logits,
                                                            labels=y2)
    loss = tf.reduce_mean(start_losses + end_losses)

    return start_prob, end_prob, start_index, end_index, mask, dx, dy, loss

def kl_div(gt, pred, lengths):

    individual_loss = []

    for i in range(1):
        length = lengths[i]
        p = pred[i][:length]
        p = tf.nn.softmax(pred[i][:length], dim=-1)
        q = gt[i][:length]

        p = tf.clip_by_value(p, 1e-7, 1.0)
        q = tf.clip_by_value(q, 1e-7, 1.0)

        individual_loss.append(tf.reduce_sum(q * tf.log(q/p), -1))
    total_loss = tf.reduce_mean(tf.stack(individual_loss))
    return total_loss

def boundary_loss(start_logits, end_logits, y1, y2, lengths, configs):

    start_prob = tf.nn.softmax(start_logits, axis=1)
    end_prob = tf.nn.softmax(end_logits, axis=1)

    outer = tf.matmul(tf.expand_dims(start_prob, axis=2),
                      tf.expand_dims(end_prob, axis=1))
    outer = tf.matrix_band_part(outer, num_lower=0, num_upper=-1)

    starts = tf.reduce_max(outer, axis=2)
    ends = tf.reduce_max(outer, axis=1)
    start_index = tf.argmax(starts, axis=1)
    end_index = tf.argmax(ends, axis=1)

    # KL diversity
    start_losses = kl_div(y1, start_logits, lengths)
    end_losses = kl_div(y2, end_logits, lengths)

    loss = tf.reduce_mean(start_losses + end_losses)

    return start_prob, end_prob, start_index, end_index, loss

def generate_proposal_boxes(vfeats, length, configs):

    batch_size = tf.shape(vfeats)[0]

    if configs.mode == "train":
        k = 100
    elif configs.mode == "test":
        k = 100

    proposal_box0 = tf.random_uniform([k, 1], maxval=1.0)
    # proposal_box1 = tf.random_uniform([k, 1], minval=0.0, maxval=0.5)
    proposal_box1 = tf.random_uniform([k, 1], minval=0.0, maxval=0.25)
    proposal_box = tf.Variable(tf.concat([proposal_box0, proposal_box1], 1), trainable=True, name='proposal_box')

    l = tf.expand_dims(tf.cast(length, tf.float32), -1)
    a = proposal_box[:, 0]
    b = proposal_box[:, 1]

    dx = tf.cast(tf.floor(tf.matmul(l, tf.expand_dims((a - b), -1), transpose_b=True)), tf.int32)
    dy = tf.cast(tf.ceil(tf.matmul(l, tf.expand_dims((a + b), -1), transpose_b=True)), tf.int32)

    ones = tf.ones([k, 1], dtype=tf.float32)
    zeros = tf.zeros([k, 1], dtype=tf.float32)

    starts = tf.expand_dims(a - b, -1)
    ends = tf.expand_dims(a + b, -1)
    boxes = tf.nn.relu(tf.concat([zeros, starts, ones, ends], 1))  # [k*4]

    return proposal_box, dx, dy, boxes


def fpn(vfeats):

    batch_size, length, dim = get_shape_list(vfeats)
    
    vfeats = tf.image.resize_images(tf.expand_dims(vfeats, 1), size=(1, length*2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    vfeats = tf.squeeze(vfeats, 1)
    vfeats = conv1d_(vfeats, kernel_size=3, stride=1, dim=dim, use_bias=True, activation=tf.nn.relu, name="conv")

    vfeats0_orig = conv1d_(vfeats, kernel_size=1, stride=1, dim=dim, use_bias=True, activation=tf.nn.relu, name="orig")

    vfeats1 = conv1d_(vfeats, kernel_size=3, stride=2, dim=dim, use_bias=True, activation=tf.nn.relu, padding='SAME',name="conv2")
    vfeats1_orig = conv1d_(vfeats1, kernel_size=1, stride=1, dim=dim, use_bias=True, activation=tf.nn.relu, name="orig")
    
    vfeats2 = conv1d_(vfeats1, kernel_size=3, stride=2, dim=dim, use_bias=True, activation=tf.nn.relu, padding='SAME', name="conv2")
    vfeats2_orig = conv1d_(vfeats2, kernel_size=1, stride=1, dim=dim, use_bias=True, activation=tf.nn.relu, name="orig")
    vfeats2_up = tf.image.resize_images(tf.expand_dims(vfeats2_orig, 1), size=(1, get_shape_list(vfeats1)[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    vfeats2_up = tf.squeeze(vfeats2_up, 1)

    vfeats_1 = vfeats2_up + vfeats1_orig
    vfeats1_up = tf.image.resize_images(tf.expand_dims(vfeats_1, 1), size=(1, get_shape_list(vfeats)[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    vfeats1_up = tf.squeeze(vfeats1_up, 1)
    vfeats_0 = vfeats1_up + vfeats0_orig

    v0 = conv1d_(vfeats_0, kernel_size=3, stride=1, dim=dim, use_bias=True, activation=tf.nn.relu, name="conv3")
    v1 = conv1d_(vfeats_1, kernel_size=3, stride=1, dim=dim, use_bias=True, activation=tf.nn.relu, name="conv3")
    v2 = conv1d_(vfeats2_orig, kernel_size=3, stride=1, dim=dim, use_bias=True, activation=tf.nn.relu, name="conv3")

    return v0, v1, v2


def binary_focal_loss_fixed(n_classes, logits, true_label):
    alpha = tf.constant(0.25, dtype=tf.float32)
    gamma = tf.constant(2, dtype=tf.float32)
    epsilon = 1.e-8
    # y_true and y_pred
    y_true = tf.one_hot(true_label, n_classes)
    y_pred = tf.clip_by_value(logits, epsilon, 1. - epsilon)
    p_t = y_true * y_pred \
            + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)

    weight = tf.pow((tf.ones_like(y_true) - p_t), gamma)

    alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
    #- alpha * (1-p_t)^gamma * log(p_t)
    focal_loss = - alpha_t * weight * tf.log(p_t)
    return tf.reduce_mean(focal_loss)


def dynamic_head(configs,
                 proposal_box,
                 v_mask,
                 q_mask,
                 vfeats0,
                 qfeats0,
                 drop_rate,
                 dx,
                 dy,
                 boxes,
                 y1,
                 y2,
                 train,
                 reuse=None,
                 name='dynamic_head'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        batch_size = tf.shape(vfeats0)[0]
        dim = tf.shape(vfeats0)[-1]

        box_index = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size), -1), [1, 100]), [batch_size * 100])
        inputs = tf.image.crop_and_resize(tf.expand_dims(vfeats0, 1),
                                          tf.tile(boxes, [batch_size, 1]),
                                          box_ind=box_index,
                                          crop_size=(1, 16))
        inputs = tf.reshape(inputs, [batch_size, 100, 1, 16, dim])

        # outputs = tf.expand_dims(vfeats0, 1)
        length = tf.reduce_sum(v_mask, -1)
        ll = tf.cast(length, tf.float32)
        N = batch_size
        b = 100
        l = 16
        d = 256

        features = tf.reshape(inputs, [N * b, l, d])
        pro_features = tf.get_variable(
            "pro_features", [b, d],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(),
            trainable=True)
        pro_features = tf.expand_dims(pro_features, 0)
        # MHSA
        pro_features2 = multihead_attention_block(
            pro_features,
            dim=d,
            num_heads=4,
            mask=None,
            use_bias=True,
            drop_rate=drop_rate,
            reuse=False,
            name='multihead_attention_pro')
        pro_features = pro_features + tf.nn.dropout(pro_features2,
                                                    rate=drop_rate)
        pro_features = layer_norm(pro_features,
                                  name="layer_norm3")  #[1, b, 128]
        pro_features_batch = tf.reshape(tf.tile(pro_features, [N, 1, 1]),
                                        [1, N * b, d])

        parameters = tf.layers.dense(pro_features_batch,
                                     2 * d * 64,
                                     name="dynamic_layer")
        parameters = tf.reshape(parameters, [N * b, 1, 2 * d * 64])
        param1 = tf.reshape(parameters[:, :, :d * 64], [-1, d, 64])
        param2 = tf.reshape(parameters[:, :, d * 64:], [-1, 64, d])

        features = tf.matmul(features, param1)  #[Nb, 1, 32]
        features = layer_norm(features, name="layer_norm4")
        features = tf.nn.relu(features)

        features = tf.matmul(features, param2)  #[Nb, 1, 128]
        features = layer_norm(features, name="layer_norm5")
        features = tf.nn.relu(features)

        # not pro feature for 4 lines
        features = tf.reshape(features, [N * b, d * l])
        features = tf.layers.dense(features, d, name="out_layer")
        features = layer_norm(features, name="layer_norm6")
        features = tf.nn.relu(features)

        features = tf.reshape(features, [N, b, d])

        features = pro_features + tf.nn.dropout(features,
                                                rate=drop_rate)  #[1, b, 128]
        features = layer_norm(features, name="layer_norm7")

        inputs = tf.reshape(features, [N, b, d])

        position_embedding = tf.layers.dense(proposal_box, 32, activation=tf.nn.relu) #[k,256]

        # weighted pooling and concatenation
        inputs = context_query_position_concat(inputs,
                                               qfeats0,
                                               position_embedding,
                                               q_mask=q_mask,
                                               reuse=False,
                                               name='context_query_position_concat')

        x_index = tf.cast(tf.reshape(tf.arg_max(y1, 1), [-1, 1]), tf.float32)
        y_index = tf.cast(tf.reshape(tf.arg_max(y2, 1), [-1, 1]), tf.float32)

        length = tf.reduce_sum(v_mask, -1)
        ll = tf.cast(length, tf.float32)
        x = tf.cast(x_index, tf.float32) # [8, 1]
        y = tf.cast(y_index, tf.float32)

        dx = tf.expand_dims((proposal_box[:, 0] - proposal_box[:, 1]), 0) * tf.expand_dims(ll, -1) # [1,100]*[8,1]
        dy = tf.expand_dims((proposal_box[:, 0] + proposal_box[:, 1]), 0) * tf.expand_dims(ll, -1)

        l1 = (tf.abs(dx-x) + tf.abs(dy-y))/tf.expand_dims(ll, 1)

        union = (tf.minimum(dx, x), tf.maximum(dy, y))

        inter = (tf.maximum(dx, x), tf.minimum(dy, y))

        iou = tf.maximum(
            tf.cast((inter[1] - inter[0]), tf.float32) / tf.cast(
                (union[1] - union[0]), tf.float32), 0.0)

        zero = tf.zeros_like(iou)

        iou = tf.where(dy - dx > 0, x=iou, y=zero)

        match_cost = 1-iou + l1 #[8, 100]
        assigned_label = tf.arg_min(match_cost, -1) # [8]

        one = tf.ones_like(iou)

        output = tf.layers.dense(inputs, 256, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        output = layer_norm(output, name="layer_norm8")
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, rate=drop_rate)
        logits = tf.squeeze(
            tf.layers.dense(output, 1, activation=tf.nn.sigmoid), -1) #[bs, 100]
        # logits = tf.squeeze(
        #     tf.layers.dense(output, 1), -1)
        # reg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=iou, logits=logits))
        weights = tf.where(iou < 0.001, x=0.92 + zero, y=one)
        reg_loss = tf.losses.mean_squared_error(labels=iou, predictions=logits, weights=weights)

        p_s = tf.cond(train, lambda: tf.batch_gather(proposal_box[:, 0], tf.arg_max(iou, -1)), lambda: tf.batch_gather(proposal_box[:, 0], tf.arg_max(logits, -1)))
        p_e = tf.cond(train, lambda: tf.batch_gather(proposal_box[:, 1], tf.arg_max(iou, -1)), lambda: tf.batch_gather(proposal_box[:, 1], tf.arg_max(logits, -1)))

        p_x = tf.reshape((p_s - p_e) * ll, [-1, 1]) #(32,)
        p_y = tf.reshape((p_s + p_e) * ll, [-1, 1])
        u = (tf.minimum(p_x, x), tf.maximum(p_y, y))
        i = (tf.maximum(p_x, x), tf.minimum(p_y, y))
        iou_pred = tf.maximum(
            tf.cast((i[1] - i[0]), tf.float32) / tf.cast(
                (u[1] - u[0]), tf.float32), 1e-10)
        c = tf.cast((u[1] - u[0]), tf.float32)
        # I = tf.maximum(tf.cast((i[1] - i[0]), tf.float32), 0.0)
        # U = tf.cast(p_y - p_x + y - x, tf.float32)
        # dcenter = 0.25 * tf.cast(p_y + p_x - y - x, tf.float32) **2
        # # giou = iou_pred - (c - U + I) / c
        # diou = iou_pred - dcenter/c **2 
        one = tf.ones_like(iou_pred)
        sig = tf.where(iou_pred > 0.95, x=one, y=iou_pred)
       
        iou_loss = tf.reduce_mean(tf.ones_like(iou_pred) - sig)
        # iou_loss = tf.reduce_mean(tf.ones_like(iou_pred) - giou)
        # iou_loss = tf.reduce_mean(tf.ones_like(iou_pred) - diou)
        
        
        p_s_label = tf.cast((x + y) / (2 * ll), tf.float32)
        p_e_label = tf.cast((y - x) / (2 * ll), tf.float32)
        norm = tf.cast((y - x + 1) / ll, tf.float32)

        # l1_loss = tf.reduce_mean(
        #     tf.abs(p_s - p_s_label) / norm + tf.abs(p_e - p_e_label) / norm)
        # l1_loss = tf.reduce_mean(
        #     tf.abs(p_s - p_s_label) + tf.abs(p_e - p_e_label))
        l1_loss = tf.reduce_mean((tf.abs(p_x - x)+tf.abs(p_y - y))/tf.expand_dims(ll, 1))
        # l1_loss = 0
        regular = tf.reduce_mean(
            tf.nn.relu(proposal_box[:, 1] - proposal_box[:, 0]) +
            tf.nn.relu(proposal_box[:, 1] + proposal_box[:, 0] - 1.0))

        regular2 = 0.0

        proposal_box1 = 0
        l1reg_loss = 0

        p_x = tf.cast(tf.rint(p_x), tf.int32)
        p_y = tf.cast(tf.rint(p_y), tf.int32)

        return reg_loss, l1reg_loss, l1_loss, iou_loss, regular, regular2, iou, [dx, dy], p_x, p_y