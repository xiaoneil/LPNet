import os
import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from argparse import ArgumentParser
from models.LPNet import LPNet
from utils.prepro_tacos import prepro_tacos
from utils.data_utils import load_video_features, load_json, write_json, batch_iter
from utils.runner_utils import write_tf_summary, eval_test, get_feed_dict

parser = ArgumentParser()
parser.add_argument("--gpu_idx", type=str, default="0", help="GPU index")
parser.add_argument("--seed", type=int, default=12345, help="random seed")
parser.add_argument("--mode", type=str, default="train", help="prepro | train | test")
parser.add_argument("--feature", type=str, default="org", help="[new | org], org: the visual feature from Gao et al.")
parser.add_argument("--root", type=str, default='data/TACoS', help="root directory for store raw data")
parser.add_argument("--wordvec_path", type=str, default="data/glove.840B.300d.txt", help="glove word embedding path")
parser.add_argument("--home_dir", type=str, default=None, help="home directory for saving models")
parser.add_argument("--save_dir", type=str, default=None, help="directory for saving processed dataset")
parser.add_argument("--num_train_steps", type=int, default=None, help="number of training steps")
parser.add_argument("--char_size", type=int, default=None, help="number of characters")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--video_feature_dim", type=int, default=1024, help="video feature input dimension")
parser.add_argument("--char_dim", type=int, default=50, help="character dimension")
parser.add_argument("--hidden_size", type=int, default=256, help="hidden size")
parser.add_argument("--max_position_length", type=int, default=512, help="max position length")
parser.add_argument("--highlight_lambda", type=float, default=5.0, help="lambda for highlight region")
parser.add_argument("--extend", type=float, default=0.1, help="highlight region extension")
parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
parser.add_argument("--drop_rate", type=float, default=0.1, help="dropout rate")
parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
parser.add_argument("--period", type=int, default=100, help="training loss print period")
parser.add_argument("--eval_period", type=int, default=None, help="evaluation period")
configs = parser.parse_args()

# os environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu_idx

np.random.seed(configs.seed)
tf.set_random_seed(configs.seed)
tf.random.set_random_seed(configs.seed)

# specify the dataset directory
if configs.home_dir is None:
    configs.home_dir = "ckpt/tacos_{}_{}".format(configs.feature, configs.max_position_length)
configs.save_dir = "datasets/tacos_{}/{}".format(configs.feature, configs.max_position_length)
configs.video_feature_dim = 1024 if configs.feature == "new" else 4096

if configs.mode.lower() == "prepro":
    prepro_tacos(configs)

elif configs.mode.lower() == "train":
    video_feature_path = os.path.join(configs.root, "tacos_features_{}".format(configs.feature))
    video_features = load_video_features(video_feature_path, max_position_length=configs.max_position_length)

    train_set = load_json(os.path.join(configs.save_dir, "train_set.json"))
    test_set = load_json(os.path.join(configs.save_dir, "test_set.json"))
    num_train_batches = math.ceil(len(train_set) / configs.batch_size)

    if configs.eval_period is None:
        configs.eval_period = num_train_batches
    if configs.num_train_steps is None:
        configs.num_train_steps = num_train_batches * configs.epochs
    if configs.char_size is None:
        configs.char_size = len(load_json(os.path.join(configs.save_dir, "char_dict.json")))

    log_dir = os.path.join(configs.home_dir, "event")
    model_dir = os.path.join(configs.home_dir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # write configs to json file
    write_json(vars(configs), save_path=os.path.join(model_dir, "configs.json"), pretty=True)

    with tf.Graph().as_default() as graph:
        model = LPNet(configs, graph=graph)
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            optimizer = tf.train.AdamOptimizer(configs.init_lr,
                                               beta1=0.9,
                                               beta2=0.999,
                                               name='AdamOptimizer')
            # train_op = optimizer.minimize(model.my_loss, global_step=model.global_step)
            trainable_vars = tf.trainable_variables()
            freeze_bbox_var_list = [t for t in trainable_vars if not t.name.startswith(u'proposal_box')]
            bbox_var_list = [t for t in trainable_vars if t.name.startswith(u'proposal_box')]
            train_op1 = optimizer.minimize(model.reg_loss, global_step=model.global_step, var_list=freeze_bbox_var_list)
            train_op2 = optimizer.minimize(model.my_loss, var_list=bbox_var_list)
            saver = tf.train.Saver(max_to_keep=5)
            writer = tf.summary.FileWriter(log_dir)
            sess.run(tf.global_variables_initializer())

            best_r1i7 = -1.0
            score_writer = open(os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8")

            for epoch in range(configs.epochs):
                for data in tqdm(batch_iter(train_set, video_features, configs.batch_size, configs.extend, True, True),
                                 total=num_train_batches, desc="Epoch %d / %d" % (epoch + 1, configs.epochs)):

                    # run the model
                    feed_dict = get_feed_dict(data, model, configs.drop_rate)
                    # _, loss, h_loss,lloss, rloss, global_step = sess.run([train_op, model.my_loss, model.highlight_loss, model.loss, model.reg_loss,
                    #                                          model.global_step], feed_dict=feed_dict)
                    # if global_step % configs.period == 0:
                    #     write_tf_summary(writer, [("train/my_loss", loss), ("train/highlight_loss", h_loss),("train/reg_loss", rloss), ("train/cls_loss", lloss)], global_step)

                    _, _, loss, rloss, iloss, lloss, kloss, hloss, global_step = sess.run(
                        [ 
                            train_op1, train_op2,
                            model.my_loss,
                            model.reg_loss, model.iou_loss, model.l1_loss, model.loss, model.highlight_loss,
                            model.global_step
                        ],
                        feed_dict=feed_dict)
                    if global_step % configs.period == 0:
                        # write_tf_summary(writer, [("train/my_loss", loss)], global_step)
                        write_tf_summary(writer, [("train/my_loss", loss),
                                                  ("train/reg_loss", rloss),
                                                  ("train/iou_loss", iloss),
                                                  ("train/l1_loss", lloss),
                                                  ("train/kl_loss", kloss),
                                                  ("train/hl_loss", hloss)],
                                         global_step)

                    # evaluate
                    if global_step % num_train_batches == 0:

                        r1i3, r1i5, r1i7, mi, value_pairs, score_str = eval_test(
                            sess=sess, model=model, dataset=test_set, video_features=video_features,
                            configs=configs, epoch=epoch + 1, global_step=global_step, name="test")

                        write_tf_summary(writer, value_pairs, global_step)
                        score_writer.write(score_str)
                        score_writer.flush()

                        # save the model according to the result of Rank@1, IoU=0.7
                        if r1i7 > best_r1i7:
                            best_r1i7 = r1i7
                            filename = os.path.join(model_dir, "model_{}.ckpt".format(global_step))
                            saver.save(sess, filename)

            score_writer.close()

elif configs.mode.lower() == "test":

    # load previous configs
    model_dir = os.path.join(configs.home_dir, "model")
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()

    # load video features
    video_feature_path = os.path.join(configs.root, "tacos_features_{}".format(configs.feature))
    video_features = load_video_features(video_feature_path, max_position_length=configs.max_position_length)

    # load test dataset
    test_set = load_json(os.path.join(configs.save_dir, "test_set.json"))

    # restore model and evaluate
    with tf.Graph().as_default() as graph:
        model = LPNet(configs, graph=graph)
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))

            r1i3, r1i5, r1i7, mi, *_ = eval_test(sess, model, dataset=test_set, video_features=video_features,
                                                 configs=configs, name="test")

            print("\n" + "\x1b[1;31m" + "Rank@1, IoU=0.3:\t{:.2f}".format(r1i3) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "Rank@1, IoU=0.5:\t{:.2f}".format(r1i5) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "Rank@1, IoU=0.7:\t{:.2f}".format(r1i7) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "{}:\t{:.2f}".format("mean IoU".ljust(15), mi[0]) + "\x1b[0m", flush=True)

else:
    raise ValueError("Unknown mode {}!!!".format(configs.mode))
