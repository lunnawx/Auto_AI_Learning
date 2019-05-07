# coding: utf-8

# 《安娜卡列尼娜》新编——利用TensorFlow构建LSTM模型

import time
from collections import namedtuple
import numpy as np
import tensorflow as tf



'''use this txt-file as training source'''
#open txt
with open('anna.txt', 'r') as f:
    text = f.read()

#build vocabulary
vocab = set (text)

#vocabulary to number & number to vcabulary
vocab_to_int = {c:i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text],dtype=np.int32)

text[:100]
encoded[:100]
len(vocab)



'''fuction part'''
'''mini-batch will be divided in this fuction'''
def get_batches(arr,n_seqs,n_steps):

    batch_size = n_seqs * n_steps

    #get integry
    n_batches = int(len(arr)/batch_size)

    #reserve complete batch, which means drop unmol part
    arr = arr[:batch_size*n_batches]
    arr = arr.reshape((n_seqs,-1))

    #define a generator and return a object of generator
    for n in range(0,arr.shape[1],n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1],y[:, -1] = x[:, 1:],y[:, 0]
        yield x,y

batches = get_batches(encoded, 10, 50)
x, y = next(batches)
print('x\n', x[:10,:10])
print('ny\n', x[:10,:10])

'''define the layer of input'''
def build_inputs(num_seqs,num_steps):
    # num_seqs: number of every sequences
    # num_steps: number of steps in every sequences reminded below

    #placeholder, name,shape
    inputs = tf.placeholder(tf.int32,shape=(num_seqs,num_steps),name='inputs' )
    targets = tf.placeholder(tf.int32,shape=(num_steps,num_seqs),name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob

'''define layer of LSTM'''
def lstm_cell():
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    return tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    # lstm_size: the number of hidden layer in lstm cell
    # num_layers: number of layer in lstm
    #batch_size : size of batch_size = num_seqs*num_step

    #build a lstm cell
#    lstm_cells = []
#    for i in range(num_layers):
    #lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)


    #unit of dropout
    #drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    # drop = tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=keep_prob)

    #weaver the LSTM cell
    # cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
    #cell = tf.nn.rnn_cell.MultiRNNCell([drop for _ in range(num_layers)])
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size,tf.float32)

    return cell, initial_state

'''define layer of output'''
def build_output(lstm_output, in_size, out_size):
    # lstm_output: result of LSTM layer
    # in_size : the size after been modified by LSTM
    # out_size: the size after been modified by softmax

    #concat the seq_output
    seq_output = tf.concat(lstm_output, axis=1)

    x = tf.reshape(seq_output,[-1,in_size])

    #link the softmax layer and output of LSTM layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    #calculate the logits and proporbility
    logits = tf.matmul(x,softmax_w) +softmax_b
    out = tf.nn.softmax(logits, name='predictions')
    return out ,logits

'''define loss'''
def build_loss(logits, targets, lstm_size, num_classes):
    #logits: result of output from build_output
    #targets: target word
    #lstm_size: number of sport of LSTM cell
    #num_classes: vocab_size

    y_one_hot = tf.one_hot(targets,num_classes)
    y_reshaped = tf.reshape(y_one_hot,logits.get_shape())

    #define CrossEntropy
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss

'''define optimizer in case of gradients exploding&disappear'''
def build_optimizer(loss, learning_rate, grad_clip):
    #loss, learning rate

    tvars = tf.trainable_variables()

    grads, _ = tf.clip_by_global_norm(tf.gradients(loss,tvars),grad_clip)
    # grad, _ = tf.gradients(loss, tvars)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads,tvars))

    return optimizer


class CharRNN:
    def __init__(self, num_classes, batch_size=64,
                 num_steps=50, lstm_size=128,
                 num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):

        #if sampling is true, use SGD
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        #input layer
        self.inputs, self.targets, self.keep_prob=build_inputs(batch_size, num_steps)

        #lstm
        cell, self.initial_state = build_lstm(lstm_size, num_layers,
                                              batch_size ,self.keep_prob)

        #encode the input by one-hot
        x_one_hot = tf.one_hot(self.inputs, num_classes)

        #run CharRNN
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

        #Loss and optimizer
        self.loss = build_loss(self.logits, self.targets,lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate = 0.001
keep_prob = 0.5

epochs = 20
save_every_n = 200

model = CharRNN(len(vocab),batch_size=batch_size,num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers,
                learning_rate=learning_rate)
saver = tf.train.Saver(max_to_keep=100)
MODEL_SAVE_PATH = "./checkpoints/"
#MODEL_NAME = "mnist_model"
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    '''
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
    '''
    counter = 0
    for e in range(epochs):
        new_state = sess.run(model.initial_state)
        loss = 0
        for x,y in get_batches(encoded,batch_size,num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets:y,
                    model.keep_prob: keep_prob,
                    model.initial_state:new_state}
            batch_loss, new_state, _ = sess.run([model.loss,
                                                model.final_state,
                                                model.optimizer],
                                                feed_dict=feed)
            end = time.time()

            #control the print lines
            if counter % 100 == 0:
                print('轮数: {}/{}... '.format(e+1, epochs),
                      '训练步数: {}... '.format(counter),
                      '训练误差: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end-start)))

            if (counter %save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))


tf.train.get_checkpoint_state('checkpoint')

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p/np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    #generate the text
    #checkpoint
    #n_sample: longth of new text
    #lstm_size
    #vocab_size
    #prime: start of new sample

    #transfor the input into list
    samples = [c for c in prime]

    #model
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1,1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(vocab))

        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)
            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
    return "".join(samples)
    #return  "".join('%s' %id for id in samples)
    #newStr = [str(x) for x in samples]
    return "".join(newStr)

tf.train.latest_checkpoint('checkpoints')




# In[26]:

# 选用最终的训练参数作为输入进行文本生成
checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="The")
print(samp)


# In[22]:

checkpoint = 'checkpoints/i200_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)


# In[23]:

checkpoint = 'checkpoints/i1000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)


# In[24]:

checkpoint = 'checkpoints/i2000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)


def printresult():
    #name is look like 'checkpoints/i2000_l512.ckpt'
    checkpoint = 'checkpoints/i200_l512.ckpt'
    samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
    print(samp)

