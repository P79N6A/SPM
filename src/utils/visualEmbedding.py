import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm
from nets import SMP3
from utils2 import MixData, Trainer, Visual
from utils2.Cloud import zh_to_en
import argparse
import os
import shutil
import pickle
from mainPre import config
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='0,1')
    parser.add_argument('--dataout', default='./data/multi_data7_less/multi_data7_less')
    parser.add_argument('--emb_dim', type=int, default=config["emb_dim"])
    parser.add_argument('--batch_size', type=int, default=config["batch_size"])
    # word2vec arguments
    parser.add_argument('--min_count', type=int, default=config["min_count"])
    parser.add_argument('--load_emb', type=int, default=config["load_emb"])
    # model arguments
    parser.add_argument('--doc_len', type=int, default=config["doc_len"])
    parser.add_argument('--n_skill', type=int, default=config["n_sent"])
    parser.add_argument('--skill_len', type=int, default=config["sent_len"])
    parser.add_argument('--conv_size', type=int, default=config["conv_size"])
    parser.add_argument('--mode', type=str, default="SMP")
    parser.add_argument("--aux", type=int, default=0)
    parser.add_argument('--lr', type=float, default=config["learning_rate"])
    parser.add_argument('--reg', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--n_epoch', type=int, default=3)
    # tf arguments
    parser.add_argument('--board_dir', default="")
    parser.add_argument('--load_ckpt', type=int, default=0)
    parser.add_argument("--load_pkl", type=int, default=1)
    return parser.parse_args()


def visual_emb(
        summary_writer,
        model: SMP3.Model,
        mix_data: MixData.MixData,
        board_dir,
        visual_ids,
):

    word_dict = mix_data.word_dict
    ids = mix_data.feature_name_sparse
    id_dict = {k: v for v, k in enumerate(ids)}

    if os.path.exists('{}/zh_to_en.json'.format(board_dir)):
        with open('{}/zh_to_en.json'.format(board_dir)) as f:
            zh_en_dic = json.load(f)
    else:
        zh_en_dic = dict()
        for label, index in tqdm(word_dict.items()):
            try:
                en_label = zh_to_en(label)
            except:
                en_label = "error"
            zh_en_dic[label] = en_label
        with open('{}/zh_to_en.json'.format(board_dir), 'w') as f:
            json.dump(zh_en_dic, f, ensure_ascii=False, indent=2)

    word_W = model.word_emb_W
    words_idxs = list(word_dict.items())
    words_idxs = [(word, idx) for word, idx in words_idxs if word in zh_en_dic]
    word_W = tf.gather(word_W, [idx for word, idx in words_idxs])

    id_W = model.id_emb_W
    id_idxs = [id_dict[visual_id] for visual_id in visual_ids]
    id_W = tf.gather(id_W, id_idxs)

    W = tf.concat([word_W, id_W], axis=0)
    W = tf.Variable(initial_value=W, name="vis")

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = W.name
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(summary_writer, config)

    with open("{}/metadata.tsv".format(board_dir), 'w') as f:
        f.write("Index\tLabel\n")
        for label, index in tqdm(words_idxs):
            f.write("{}\t{}\n".format(index, zh_en_dic[label]))
        for index, label in tqdm(enumerate(visual_ids, len(word_dict))):
            f.write("{}\t{}\n".format(index, label))


if __name__ == '__main__':
    # 参数接收器
    args = parse_args()
    if args.board_dir == "":
        args.board_dir = args.mode

    # 显卡占用
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    if args.load_pkl and os.path.exists('{}.pkl2'.format(args.dataout)):
        with open('{}.pkl2'.format(args.dataout), 'rb') as f:
            mix_data = pickle.load(f)
    else:
        mix_data = MixData.MixData(
            fpin=args.dataout,
            wfreq=args.min_count,
            doc_len=args.doc_len,
            n_skill=args.n_skill,
            skill_len=args.skill_len,
            emb_dim=args.emb_dim,
            load_emb=args.load_emb,
        )
        with open('{}.pkl2'.format(args.dataout), 'wb') as f:
            pickle.dump(mix_data, f)

    train_data = lambda: mix_data.data_generator(
        fp='{}.train2'.format(args.dataout),
        batch_size=args.batch_size,
    )

    test_data = lambda: mix_data.data_generator(
        fp='{}.test2'.format(args.dataout),
        # fp='data/multi_data7/multi_data7.test2'.format(args.dataout),
        batch_size=args.batch_size,
    )

    test_data_raw = lambda: mix_data.data_generator(
        fp='{}.test2'.format(args.dataout),
        batch_size=args.batch_size,
        raw=True,
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        board_dir = './board/{}'.format(args.board_dir)
        # if os.path.exists(board_dir):
        #     shutil.rmtree(board_dir)
        writer = tf.summary.FileWriter(board_dir)

        model = SMP3.Model(
            doc_len=mix_data.doc_len,
            feature_len=len(mix_data.feature_name),
            emb_dim=args.emb_dim,
            n_feature=len(mix_data.feature_name_sparse),
            n_word=len(mix_data.word_dict),
            conv_size=args.conv_size,
            emb_pretrain=mix_data.embs,
            l2=args.reg,
            mode=args.mode,
            dropout=args.dropout,
            auxiliary_loss=args.aux,
        )

        with open('{}.test2'.format(args.dataout)) as f:
            data = f.readlines()
            visual_ids = [
                "expect_id={}".format(line.split('\001')[0])
                for line in data
            ]
            visual_ids = set(visual_ids)

        visual_emb(
            summary_writer=writer,
            model=model,
            mix_data=mix_data,
            board_dir=board_dir,
            visual_ids=visual_ids,
        )

        # writer.add_graph(sess.graph)
        saver = tf.train.Saver()

        if args.load_ckpt:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(board_dir)  # 注意此处是checkpoint存在的目录，千万不要写成‘./log’
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)  # 自动恢复model_checkpoint_path保存模型一般是最新
                print("Model restored...")
            else:
                print('No Model')
        else:
            Trainer.train(
                sess=sess,
                model=model,
                writer=writer,
                train_data_fn=train_data,
                test_data_fn=test_data,
                lr=args.lr,
                n_epoch=args.n_epoch,
                save_path="{}.ckpt.{}".format(args.dataout, args.mode),
            )

        visual_str, cv_words, jd_words = Visual.visual(
            sess=sess,
            model=model,
            test_data_fn=test_data,
            raw_data_fn=test_data_raw,
        )
        with open('{}.html'.format(args.dataout), 'w') as f:
            f.write(visual_str)
        with open('{}.job_implicit_prefer'.format(args.dataout), 'w') as f:
            f.write(cv_words)
        with open('{}.exp_implicit_prefer'.format(args.dataout), 'w') as f:
            f.write(jd_words)

        saver.save(
            sess=sess,
            save_path="{}/model.ckpt".format(board_dir),
            global_step=1,
        )

    writer.close()

