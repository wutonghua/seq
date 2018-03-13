#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import jieba
kouchifile=pd.read_csv('c.csv')
#print(kouchifile.head())
question = kouchifile['title'].astype('str')
answer=kouchifile['zhenduan'].astype('str')
cw = lambda x: ''.join(jieba.cut(x))
kouchifile['question'] = question.apply(cw)
kouchifile['words'] = answer.apply(cw)
ignoring_words = ['【', '】','.',' ','。' ,'！','？','，','：',',','\xa0','?','0','1','2','3','4','5','6','7','8','9','�','；','（','）','”']

def preprocessing(sen):
    res = []
    for word in sen:
        if word not in ignoring_words:
            res.append(word)
    return res
X_question=kouchifile['question']
X_answer=kouchifile['words']
question1 = [preprocessing(x) for x in X_question]
answer1 = [preprocessing(x) for x in X_answer]
with open('question.csv','w') as f1:
    for line in question1:
        # print(line)
        f1.write(' '.join(line) + '\n')
with open('answer.csv','w') as f2:
    for line in answer1:
        f2.write(' '.join(line) + '\n')
import random


def convert_seq2seq_files(questions, answers, TESTSET_SIZE=500):
	# 创建文件
	train_enc = open('train.enc', 'w')  # 问
	train_dec = open('train.dec', 'w')  # 答
	test_enc = open('test.enc', 'w')  # 问
	test_dec = open('test.dec', 'w')  # 答

	# 选择500数据作为测试数据
	test_index = random.sample([i for i in range(len(questions))], TESTSET_SIZE)

	for i in range(len(questions)):
		if i in test_index:
			test_enc.write(' '.join(questions[i]) + '\n')
			test_dec.write(' '.join(answers[i]) + '\n')
		else:
			train_enc.write(' '.join(questions[i]) + '\n')
			train_dec.write(' '.join(answers[i]) + '\n')
		if i % 10 == 0:
			print(len(range(len(questions))), '处理进度:', i)

	train_enc.close()
	train_dec.close()
	test_enc.close()
	test_dec.close()
convert_seq2seq_files(question1, answer1)
train_encode_file = 'train.enc'
train_decode_file = 'train.dec'
test_encode_file = 'test.enc'
test_decode_file = 'test.dec'
print('开始创建词汇表...')
# 特殊标记，用来填充标记对话
PAD = "__PAD__"
GO = "__GO__"
EOS = "__EOS__"  # 对话结束
UNK = "__UNK__"  # 标记未出现在词汇表中的字符
START_VOCABULART = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
# 参看tensorflow.models.rnn.translate.data_utils

vocabulary_size = 5000


# 生成词汇表文件
def gen_vocabulary_file(input_file, output_file):
	vocabulary = {}
	with open(input_file) as f:
		counter = 0
		for line in f:
			counter += 1
			tokens = [word for word in line.strip()]
			for word in tokens:
				if word in vocabulary:
					vocabulary[word] += 1
				else:
					vocabulary[word] = 1
		vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)
		# 取前5000个常用汉字, 应该差不多够用了(额, 好多无用字符, 最好整理一下. 我就不整理了)
		if len(vocabulary_list) > 5000:
			vocabulary_list = vocabulary_list[:5000]
		print(input_file + " 词汇表大小:", len(vocabulary_list))
		with open(output_file, "w") as ff:
			for word in vocabulary_list:
				ff.write(word + "\n")
gen_vocabulary_file(train_encode_file, "train_encode_vocabulary")
gen_vocabulary_file(train_decode_file, "train_decode_vocabulary")
train_encode_vocabulary_file = 'train_encode_vocabulary'
train_decode_vocabulary_file = 'train_decode_vocabulary'
print("对话转向量...")
# 把对话字符串转为向量形式
def convert_to_vector(input_file, vocabulary_file, output_file):

	tmp_vocab = []
	with open(vocabulary_file, "r") as f:
		tmp_vocab.extend(f.readlines())
	tmp_vocab = [line.strip() for line in tmp_vocab]
	vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
	#{'硕': 3142, 'v': 577, 'Ｉ': 4789, '\ue796': 4515, '拖': 1333, '疤': 2201 ...}
	output_f = open(output_file, 'w')
	with open(input_file, 'r') as f:
		for line in f:
			line_vec = []
			for words in line.strip():
				line_vec.append(vocab.get(words, UNK_ID))
			output_f.write(" ".join([str(num) for num in line_vec]) + "\n")
	output_f.close()


convert_to_vector(train_encode_file, train_encode_vocabulary_file, 'train_encode.vec')
convert_to_vector(train_decode_file, train_decode_vocabulary_file, 'train_decode.vec')

convert_to_vector(test_encode_file, train_encode_vocabulary_file, 'test_encode.vec')
convert_to_vector(test_decode_file, train_decode_vocabulary_file, 'test_decode.vec')
import tensorflow as tf  # 0.12
from tensorflow.models.rnn.translate import seq2seq_model
import os
import numpy as np
import math

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

train_encode_vec = 'train_encode.vec'
train_decode_vec = 'train_decode.vec'
test_encode_vec = 'test_encode.vec'
test_decode_vec = 'test_decode.vec'

# 词汇表大小5000
vocabulary_encode_size = 5000
vocabulary_decode_size = 5000

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
layer_size = 256  # 每层大小
num_layers = 3  # 层数
batch_size = 64


# 读取*dencode.vec和*decode.vec数据（数据还不算太多, 一次读人到内存）
def read_data(source_path, target_path, max_size=None):
	data_set = [[] for _ in buckets]
	with tf.gfile.GFile(source_path, mode="r") as source_file:
		with tf.gfile.GFile(target_path, mode="r") as target_file:
			source, target = source_file.readline(), target_file.readline()
			counter = 0
			while source and target and (not max_size or counter < max_size):
				counter += 1
				source_ids = [int(x) for x in source.split()]
				target_ids = [int(x) for x in target.split()]
				target_ids.append(EOS_ID)
				for bucket_id, (source_size, target_size) in enumerate(buckets):
					if len(source_ids) < source_size and len(target_ids) < target_size:
						data_set[bucket_id].append([source_ids, target_ids])
						break
				source, target = source_file.readline(), target_file.readline()
	return data_set


model = seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_encode_size, target_vocab_size=vocabulary_decode_size,
								   buckets=buckets, size=layer_size, num_layers=num_layers, max_gradient_norm=5.0,
								   batch_size=batch_size, learning_rate=0.5, learning_rate_decay_factor=0.97,
								   forward_only=False)

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # 防止 out of memory

with tf.Session(config=config) as sess:
	# 恢复前一次训练
	ckpt = tf.train.get_checkpoint_state('.')
	if ckpt != None:
		print(ckpt.model_checkpoint_path)
		model.saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		sess.run(tf.global_variables_initializer())

	train_set = read_data(train_encode_vec, train_decode_vec)
	test_set = read_data(test_encode_vec, test_decode_vec)

	train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
	train_total_size = float(sum(train_bucket_sizes))
	train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]

	loss = 0.0
	total_step = 0
	previous_losses = []
	# 一直训练，每过一段时间保存一次模型
	while True:
		random_number_01 = np.random.random_sample()
		bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

		encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
		_, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

		loss += step_loss / 500
		total_step += 1

		print(total_step)
		if total_step % 100 == 0:
			print(model.global_step.eval(), model.learning_rate.eval(), loss)

			# 如果模型没有得到提升，减小learning rate
			if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
				sess.run(model.learning_rate_decay_op)
			previous_losses.append(loss)
			# 保存模型
			model.saver.save(sess, 'Model/chatbot_seq2seq.ckpt', global_step=model.global_step)
			loss = 0.0
			# 使用测试数据评估模型
			for bucket_id in range(len(buckets)):
				if len(test_set[bucket_id]) == 0:
					continue
				encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)
				_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
				eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
				print(bucket_id, eval_ppx)
