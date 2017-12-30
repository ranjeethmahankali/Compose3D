from ops import *
import sys

# this is for sigmoid cross entropy loss
def sigmoid_loss(m , vTrue):
    epsilon = 1e-9
    cross_entropy = -((vTrue * tf.log(m + epsilon))+((1-vTrue)*tf.log(1-m+epsilon)))
    
    # adding this to summaries
    ce_by_example = tf.reduce_sum(cross_entropy, axis=1, name='cross_entropy')
    summarize(ce_by_example)

    ce_loss = tf.reduce_sum(ce_by_example)

    # now implementing regularizations
    # l2_loss = 0
    # for v in varList:
    #     l2_loss += tf.nn.l2_loss(v)

    # l2_loss *= alpha
    # total_loss = ce_loss + l2_loss

    # with tf.name_scope('loss_params'):
    #     tf.summary.scalar('l2_loss', l2_loss)
    #     tf.summary.scalar('total_loss', total_loss)

    # return cross_entropy
    return ce_loss
    # return total_loss

# this returns the accuracy tensor
def accuracy(v, vTrue):
    difference = tf.abs(v-vTrue)
    correctness = tf.less(difference, 0.05)
    acc_norm = tf.cast(correctness, tf.float32)
    acc = tf.multiply(acc_norm, 100)

    acc_by_example = tf.reduce_mean(acc, axis=1,name='accuracy')

    summarize(acc_by_example)

    return tf.reduce_mean(acc_by_example)

LAYERS = [6000, 4096, 2048, 512, 64, 3]
# creating all the weights
with tf.variable_scope("vars"):
    weights = [
        weightVariable([LAYERS[i], LAYERS[i+1]], name="w%s"%i) for i in range(len(LAYERS)-1)
    ]

    biases = [
        biasVariable([LAYERS[i+1]], name="b%s"%i) for i in range(len(LAYERS)-1)
    ]

# list of vars we care about
all_vars = tf.trainable_variables()
varList = [v for v in all_vars if 'vars' in v.name]

view_placeholder = tf.placeholder(tf.float32, shape=[None, imgSize[0], imgSize[1], 1])
scene_params_placeholder = tf.placeholder(tf.float32, shape=[None, 3])
keep_prob_placeholder = tf.placeholder(tf.float32)

def interpreter(view_t, keep_prob):
    # [-1, 80,75,1] - view
    # [-1, 6000] - h_flat
    # [-1, 4096] - h1
    # [-1, 2048] - h2
    # [-1, 512] - h3
    # [-1, 64] - h4
    # [-1, 3] - h5

    view_flat = tf.reshape(view_t, [-1, 6000])

    h = 0
    for i in range(len(weights)):
        total_layers = len(weights)
        L = view_flat if i == 0 else h
        h = tf.matmul(L, weights[i])+biases[i]
        if i < 4:
            h = tf.nn.relu(h)
        elif i == total_layers - 1:
            h = tf.nn.sigmoid(h)
        else:
            h = tf.nn.tanh(h)
        if i == len(weights)-2:
            h = tf.nn.dropout(h, keep_prob)
        # print(h.get_shape())
    
    return h

def normalize_output(out):
    # out is of shape [batch_size, 3]
    xDim = tf.shape(out)[0]
    angles_raw = tf.slice(out, [0,0], [xDim,1])
    # print(angles_raw.get_shape())
    coords = tf.slice(out, [0,1], [xDim,2])
    # print(coords.get_shape())
    angles = tf.floor(angles_raw*4)/4
    return tf.concat([angles, coords], 1)

# vox = tf.round(m2, name='voxels')
output = interpreter(view_placeholder, keep_prob_placeholder)
scene_params = normalize_output(output)

loss = sigmoid_loss(output, scene_params_placeholder)
# loss = tf.reduce_sum(tf.abs(output - scene_params_placeholder))

optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# optim = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
accTensor = accuracy(scene_params, scene_params_placeholder)

# this is for the summaries during the training
merged = tf.summary.merge_all()