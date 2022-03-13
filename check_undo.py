from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List
import os
import models
import data
import tflib as tl


import argparse
from functools import partial
import json
import traceback

import csv
import imlib as im
import numpy as np
import pylib
import tensorflow as tf
tf1 = tf.compat.v1
tf1.disable_eager_execution()


# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', help='experiment_name')
parser.add_argument('--gpu', type=str, default='all', help='gpu')
parser.add_argument('--dataroot', type=str, default='/data/Datasets/CelebA')
parser.add_argument('--project_dataroot', type=str, default='.')
parser.add_argument('--check', action="store_true")
parser.add_argument('--save', action="store_true")
# if assigned, only given images will be tested.
parser.add_argument('--img', type=int, nargs='+',
                    default=None, help='e.g., --img 182638 202599')
parser.add_argument('--img_range', type=int, nargs=2,
                    default=None, help='e.g., --img_range 182638 202599')
# for multiple attributes
parser.add_argument('--test_atts', nargs='*', default=None)
parser.add_argument('--test_ints', nargs='*', default=None,
                    help='leave to None for all 1')
args_ = parser.parse_args()
with open('%s/output/%s/setting.txt' % (args_.project_dataroot, args_.experiment_name)) as f:
    args = json.load(f)

# model
atts = args['atts']
n_att = len(atts)
img_size = args['img_size']
shortcut_layers = args['shortcut_layers']
inject_layers = args['inject_layers']
enc_dim = args['enc_dim']
dec_dim = args['dec_dim']
dis_dim = args['dis_dim']
dis_fc_dim = args['dis_fc_dim']
enc_layers = args['enc_layers']
dec_layers = args['dec_layers']
dis_layers = args['dis_layers']

label = args['label']
use_stu = args['use_stu']
stu_dim = args['stu_dim']
stu_layers = args['stu_layers']
stu_inject_layers = args['stu_inject_layers']
stu_kernel_size = args['stu_kernel_size']
stu_norm = args['stu_norm']
stu_state = args['stu_state']
multi_inputs = args['multi_inputs']
rec_loss_weight = args['rec_loss_weight']
one_more_conv = args['one_more_conv']

dataroot = args_.dataroot
img = None
if args_.img is not None:
    if img is None:
        img = []
    img += args_.img
    print('Using selected images:', img)
if args_.img_range is not None:
    if img is None:
        img = []
    img += [i for i in range(args_.img_range[0], args_.img_range[1])]
    print(f"Using Images range({args_.img_range[0]}, {args_.img_range[1]})")


gpu = args_.gpu
if gpu != 'all':
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

# testing
# multiple attributes
test_atts = args_.test_atts
test_ints = args_.test_ints
if test_atts is not None and test_ints is None:
    test_ints = [1 for i in range(len(test_atts))]

thres_int = args['thres_int']
# others
use_cropped_img = args['use_cropped_img']
experiment_name = args_.experiment_name


# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================

# data
sess = tl.session()
te_data = data.Celeba(dataroot, atts, img_size, 1, part='test',
                      sess=sess, crop=not use_cropped_img, im_no=img)
# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers,
               multi_inputs=multi_inputs)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers,
               inject_layers=inject_layers, one_more_conv=one_more_conv)
Gstu = partial(models.Gstu, dim=stu_dim, n_layers=stu_layers, inject_layers=stu_inject_layers,
               kernel_size=stu_kernel_size, norm=stu_norm, pass_state=stu_state)

D = partial(models.D, n_att=n_att, dim=dis_dim,
            fc_dim=dis_fc_dim, n_layers=dis_layers)
# inputs
sample_image = tf1.placeholder(tf1.float32, shape=[
    None, img_size, img_size, 3])
target_att_raw = tf1.placeholder(tf1.float32, shape=[None, n_att])
raw_b_sample = tf1.placeholder(
    tf1.float32, shape=[None, n_att])

# sample
test_label = target_att_raw - raw_b_sample if label == 'diff' else target_att_raw
if use_stu:
    x_sample = Gdec(Gstu(Genc(sample_image, is_training=False),
                         test_label, is_training=False), test_label, is_training=False)
    checker = D(sample_image)
else:
    x_sample = Gdec(Genc(sample_image, is_training=False),
                    test_label, is_training=False)
    checker = D(sample_image)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# initialization
ckpt_dir = '%s/output/%s/checkpoints' % (
    args_.project_dataroot, experiment_name)
tl.load_checkpoint(ckpt_dir, sess)


def transform_atts(original: np.ndarray, test_atts: List[str], test_ints: List[int], thres_int: int):
    altered_att_list = [original.copy() for _ in range(1)]
    for a in test_atts:
        i = atts.index(a)
        altered_att_list[-1][:, i] = 1 - altered_att_list[-1][:, i]
        altered_att_list[-1] = data.Celeba.check_attribute_conflict(
            altered_att_list[-1], atts[i], atts)
    raw_original_att = original.copy()
    raw_original_att = (raw_original_att * 2 - 1) * thres_int
    raw_altered_att_list = []
    for i, altered_att in enumerate(altered_att_list):
        raw_altered_att = (altered_att * 2 - 1) * thres_int
        for t_att, t_int in zip(test_atts, test_ints):
            raw_altered_att[..., atts.index(
                t_att)] = raw_altered_att[..., atts.index(t_att)] * int(t_int)
        raw_altered_att_list.append(raw_altered_att)
    return raw_original_att, raw_altered_att_list


def undo_apply(sess: tf1.Session, input_sample: np.ndarray, raw_original_att: np.ndarray, raw_altered_att: np.ndarray):
    altered = sess.run(
        x_sample, feed_dict={
            sample_image: input_sample,
            target_att_raw: raw_altered_att,
            raw_b_sample: raw_original_att
        }
    )
    # print(raw_altered_att-raw_original_att)
    altered_undo = sess.run(
        x_sample, feed_dict={
            sample_image: altered,
            target_att_raw: raw_original_att,
            raw_b_sample: raw_altered_att
        }
    )
    # print(-raw_altered_att+raw_original_att)
    reapplication = sess.run(
        x_sample, feed_dict={
            sample_image: altered_undo,
            target_att_raw: raw_altered_att,
            raw_b_sample: raw_original_att
        }
    )
    output_list = [altered, altered_undo, reapplication]

    return output_list

# test


def undo_check(test_atts: List[str], test_ints: List[int], data: data.Dataset):
    data.reset()
    correct = 0
    incorrect = 0
    reapplication_correct = 0
    reapplication_incorrect = 0

    for idx, batch in enumerate(te_data):
        if idx % 100 == 0:
            print(f"done with {idx}", flush=True)
        xa_sample_ipt = batch[0]
        a_sample_ipt = batch[1]

        raw_original_att, raw_altered_att_list = transform_atts(
            original=a_sample_ipt, test_atts=test_atts, test_ints=test_ints, thres_int=thres_int)

        for i, raw_altered_att in enumerate(raw_altered_att_list):
            output_list = undo_apply(
                sess, xa_sample_ipt, raw_original_att, raw_altered_att)
            original_check, _ = sess.run(
                checker, feed_dict={
                    sample_image: xa_sample_ipt
                }
            )
            altered = output_list[0]
            altered_check, _ = sess.run(
                checker, feed_dict={
                    sample_image: altered
                }
            )
            altered_undo = output_list[1]
            altered_undo_check, _ = sess.run(
                checker, feed_dict={
                    sample_image: altered_undo
                }
            )
            reapplication = output_list[2]
            reapplication_check, _ = sess.run(
                checker, feed_dict={
                    sample_image: reapplication
                }
            )
            print(original_check,altered_check,altered_undo_check,reapplication_check)
            if altered_undo_check[0, 0].item() > altered_check[0, 0].item():
                correct += 1
            else:
                incorrect += 1

            if altered_undo_check[0, 0].item() > reapplication_check[0, 0].item():
                reapplication_correct += 1
            else:
                reapplication_incorrect += 1
    return dict(
        att=str(test_atts),
        correct=correct,
        incorrect=incorrect,
        reapplication_correct=reapplication_correct,
        reapplication_incorrect=reapplication_incorrect,
    )


def save_undo_application(test_atts: List[str], test_ints: List[int], data: data.Dataset):
    data.reset()
    for idx, batch in enumerate(te_data):
        xa_sample_ipt = batch[0]
        a_sample_ipt = batch[1]
        x_sample_opt_list = [xa_sample_ipt, np.full(
            (1, img_size, img_size // 10, 3), -1.0)]
        raw_original_att, raw_altered_att_list = transform_atts(
            original=a_sample_ipt, test_atts=test_atts, test_ints=test_ints, thres_int=thres_int)

        for i, raw_altered_att in enumerate(raw_altered_att_list):
            output_list = undo_apply(
                sess, xa_sample_ipt, raw_original_att, raw_altered_att)
            x_sample_opt_list += output_list
        sample = np.concatenate(x_sample_opt_list, 2)

        save_folder = 'undo'
        save_dir = '%s/output/%s/%s' % (args_.project_dataroot,
                                        experiment_name, save_folder)
        pylib.mkdir(save_dir)
        im.imwrite(sample.squeeze(0), '%s/%06d%s.png' % (save_dir,
                                                         idx +
                                                         182638 if img is None else img[idx],
                                                         '_%s' % (str(test_atts))))

        print('%06d.png done!' % (idx + 182638 if img is None else img[idx]))


def undo_check_all(data: data.Dataset):
    save_folder = 'undo'
    save_dir = '%s/output/%s/%s' % (args_.project_dataroot,
                                    experiment_name, save_folder)
    pylib.mkdir(save_dir)
    with open(os.path.join(save_dir, "results.csv"), "w") as results_file:
        field_names = ["att", "correct", "incorrect",
                       "reapplication_correct", "reapplication_incorrect"]
        writer = csv.DictWriter(results_file, fieldnames=field_names)
        writer.writeheader()
        for att in atts:
            writer.writerow(undo_check([att], [1], data))
            print(f"Done with Attribute {att}")


try:
    print(atts)
    print("\n\n\n------------------")
    if args_.save:
        save_undo_application(test_atts=test_atts,
                              test_ints=test_ints, data=te_data)

    if args_.check:
        if test_atts is None:
            undo_check_all(data=te_data)
        else:
            print(undo_check(test_atts=test_atts,
                  test_ints=test_ints, data=te_data))
except:
    traceback.print_exc()
finally:
    sess.close()
