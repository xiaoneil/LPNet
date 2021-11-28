import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.data_utils import batch_iter
import pickle
import os

def write_tf_summary(writer, value_pairs, global_step):
    for tag, value in value_pairs:
        summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summ, global_step=global_step)
    writer.flush()


def calculate_iou_accuracy(ious, threshold):
    total_size = float(len(ious))
    count = 0

    for iou in ious:
        if iou >= threshold:
            count += 1

    return float(count) / total_size * 100.0


def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))

    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])

    return max(0.0, iou)


def convert_to_time(start_index, end_index, num_features, duration):
    s_times = np.arange(0, num_features).astype(np.float32) * duration / float(num_features)
    e_times = np.arange(1, num_features + 1).astype(np.float32) * duration / float(num_features)
    if start_index >= num_features:
        start_index = num_features - 1
    if end_index >= num_features:
        end_index = num_features - 1
    if start_index < 0:
        start_index = 0
    if end_index <0:
        end_index = 0
    start_time = s_times[start_index]
    end_time = e_times[end_index]

    return start_time, end_time


def get_feed_dict(batch_data, model, drop_rate=None, mode='train'):
    if mode == 'train':  # training
        #(_, video_features, word_ids, char_ids, video_seq_length, start_label, end_label, highlight_labels, dx, dy, batch_mask) = batch_data

        # feed_dict = {model.video_inputs: video_features, model.video_seq_length: video_seq_length,
        #              model.word_ids: word_ids, model.char_ids: char_ids, model.y1: start_label, model.y2: end_label,
        #              model.drop_rate: drop_rate, model.highlight_labels: highlight_labels,
        #              model.dx1 : dx, model.dy1 : dy, model.mask1 : batch_mask}
        (_, video_features, word_ids, char_ids, video_seq_length, start_label, end_label, highlight_labels, is_training) = batch_data

        feed_dict = {model.video_inputs: video_features, model.video_seq_length: video_seq_length,
                     model.word_ids: word_ids, model.char_ids: char_ids, model.y1: start_label, model.y2: end_label,
                     model.drop_rate: drop_rate, model.highlight_labels: highlight_labels, model.is_training:is_training}

        return feed_dict

    else:  # eval
        # raw_data, video_features, word_ids, char_ids, video_seq_length, _, _, _, dx, dy, batch_mask = batch_data

        # feed_dict = {model.video_inputs: video_features, model.video_seq_length: video_seq_length,
        #              model.word_ids: word_ids, model.char_ids: char_ids,
        #              model.dx1 : dx, model.dy1 : dy, model.mask1 : batch_mask}

        # raw_data, video_features, word_ids, char_ids, video_seq_length, *_ = batch_data
        # feed_dict = {model.video_inputs: video_features, model.video_seq_length: video_seq_length,
        #              model.word_ids: word_ids, model.char_ids: char_ids}

        raw_data, video_features, word_ids, char_ids, video_seq_length, start_label, end_label, highlight_labels, is_training = batch_data
        feed_dict = {model.video_inputs: video_features, model.video_seq_length: video_seq_length,
                     model.word_ids: word_ids, model.char_ids: char_ids, model.y1: start_label, model.y2: end_label, model.is_training:is_training}
        return raw_data, feed_dict


# def eval_test(sess, model, dataset, video_features, configs, epoch=None, global_step=None, name="test"):
#     num_test_batches = math.ceil(len(dataset) / configs.batch_size)
#     ious = list()
#     extent = list()
#     prob = list()

#     for data in tqdm(batch_iter(dataset, video_features, configs.batch_size, configs.extend, False),
#                      total=num_test_batches, desc="evaluate {}".format(name)):

#         raw_data, feed_dict = get_feed_dict(data, model, mode=name)
#         # start_indexes, end_indexes = sess.run([model.start_index, model.end_index], feed_dict=feed_dict)
#         # iou_loss = sess.run([model.iou_loss], feed_dict=feed_dict)
#         start_indexes, end_indexes, start_prob, end_prob, iou_loss = sess.run([model.dx1, model.dy1, model.start_prob, model.end_prob, model.iou_loss], feed_dict=feed_dict)

#         # print(y1)
#         prob.append(iou_loss)
#         for record, start_index_, end_index_ in zip(raw_data, start_indexes, end_indexes):
#             for start_index, end_index in zip(start_index_, end_index_):
#             # print(record["feature_shape"]) 62
#                 start_time, end_time = convert_to_time(start_index, end_index, record["feature_shape"], record["duration"])
#                 iou = calculate_iou(i0=[start_time, end_time], i1=[record["start_time"], record["end_time"]])
#                 ious.append(iou)
#                 s = start_time - record["start_time"]
#                 e = end_time - record["end_time"]
#                 seg = record["end_time"] - record["start_time"]
#                 d = record["duration"]
#                 item = [s,e,seg,d]
#                 extent.append(item)


#     # r1i3 = calculate_iou_accuracy(ious, threshold=0.1)
#     r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
#     r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
#     r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
#     mi = np.mean(ious) * 100.0

#     value_pairs = 0

#     # write the scores
#     score_str = "Epoch {}, Step {}:\n".format(epoch, global_step)
#     score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
#     score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
#     score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
#     print("在这里", mi, type(mi), np.shape(ious))
#     score_str += "mean IoU: {:.2f}\n".format(mi)
#     # return extent, r1i3, r1i5, r1i7, mi, value_pairs, score_str
#     return r1i3, r1i5, r1i7, mi, value_pairs, score_str


def eval_test(sess, model, dataset, video_features, configs, epoch=None, global_step=None, name="test"):
    num_test_batches = math.ceil(len(dataset) / configs.batch_size)
    ious = list()
    extent = list()
    prob = list()
    pse = list()

    # query_txts = ["person reading a book.", "person opens the door."]
    # fps_list = [30.00, 19.75]
    for data in tqdm(batch_iter(dataset, video_features, configs.batch_size, configs.extend, train=False, shuffle=False),
                     total=num_test_batches, desc="evaluate {}".format(name)):

        raw_data, feed_dict = get_feed_dict(data, model, mode=name)
        # start_indexes, end_indexes = sess.run([model.start_index, model.end_index], feed_dict=feed_dict)
        # start_indexes, end_indexes, dx, dy, length= sess.run([model.px, model.py, model.dx, model.dy, model.video_seq_length], feed_dict=feed_dict)
        start_indexes, end_indexes, proposal_box = sess.run([model.px, model.py, model.proposal_box], feed_dict=feed_dict)

        # iou_loss = sess.run([model.iou_loss], feed_dict=feed_dict)
        # start_indexes, end_indexes, start_prob, end_prob, iou_loss = sess.run([model.dx, model.dy, model.start_prob, model.end_prob, model.iou_loss], feed_dict=feed_dict)

        # print(proposal_box)
        # np.savetxt('tocos_pre6.out', proposal_box)

        # print(np.shape(start_indexes))
        # prob.append(iou_loss)
        i = 0
        for record, start_index, end_index in zip(raw_data, start_indexes, end_indexes):
            # print(record["feature_shape"]) 62
            start_time, end_time = convert_to_time(start_index, end_index, record["feature_shape"], record["duration"])
            # print(start_time, end_time)
            # prediction_result = {'video_path':"/home/xsn/VSLNet/nlvl/charaders/videos/" + record["video_id"] + ".mp4",
            #                          'fps':fps_list[i],
            #                          'query_txt':query_txts[i],
            #                          'prediction':[start_time[0], end_time[0]],
            #                          'ground_truth':[record["start_time"], record["end_time"]]}

            # with open("prediction_result_"+str(i)+".pkl",'wb') as f:
            #     pickle.dump(prediction_result, f)
            # i = i + 1

            iou = calculate_iou(i0=[start_time, end_time], i1=[record["start_time"], record["end_time"]])
            ious.append(iou)

            # print(record.keys()) #dict_keys(['video_id', 'start_time', 'end_time', 'duration', 'start_index', 'end_index', 'feature_shape', 'word_ids', 'char_ids'])
            s = record["start_time"]/record["duration"]
            e = record["end_time"]/record["duration"]
            p = (e+s)/2
            l = (e-s)/2
            item = [p, l]
            extent.append(item)

            # print(record.keys()) #dict_keys(['video_id', 'start_time', 'end_time', 'duration', 'start_index', 'end_index', 'feature_shape', 'word_ids', 'char_ids'])
            # s = start_time - record["start_time"]
            # e = end_time - record["end_time"]
            # seg = record["end_time"] - record["start_time"]
            # d = record["duration"]
            # item = [s,e,seg,d]
            # extent.append(item)
            if iou > 0.8:
                s = record["start_time"]
                e = record["end_time"]
                ps = float(start_time[0])
                # print(type(end_time))
                if isinstance(end_time, np.ndarray):
                    pe = float(end_time[0])
                else:
                    pe = float(end_time)
                vid = record["video_id"]
                d = record["duration"]
                item = [s,e,ps,pe,vid, d]
                # print(type(s), type(e), type(ps), type(pe), type(vid))
                if s > 3.0 and e < record['duration']-3.0:
                    pse.append(item)

    # np.savetxt('gth0.8.out', pse) 
    # np.savetxt('t_real_proposal.out', extent) 
    # r1i3 = calculate_iou_accuracy(ious, threshold=0.1)
    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    mi = np.mean(ious) * 100.0

    # value_pairs = [("{}/Rank@1, IoU=0.3".format(name), r1i3), ("{}/Rank@1, IoU=0.5".format(name), r1i5),
    #                ("{}/Rank@1, IoU=0.7".format(name), r1i7), ("{}/mean IoU".format(name), mi[0])]
    value_pairs = [("{}/Rank@1, IoU=0.3".format(name), r1i3),
                   ("{}/Rank@1, IoU=0.5".format(name), r1i5),
                   ("{}/Rank@1, IoU=0.7".format(name), r1i7),
                   ("{}/mean IoU".format(name), mi)]
    # write the scores
    score_str = "Epoch {}, Step {}:\n".format(epoch, global_step)
    score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
    score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
    score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
    # print("在这里", mi, type(mi), np.shape(ious))
    # score_str += "mean IoU: {:.2f}\n".format(mi[0])
    score_str += "mean IoU: {}\n".format(mi)
    # return extent, r1i3, r1i5, r1i7, mi, value_pairs, score_str
    # return pse, r1i3, r1i5, r1i7, mi, value_pairs, score_str
    return r1i3, r1i5, r1i7, mi, value_pairs, score_str
