import tensorflow as tf


def res_net(x,
            num_residual_blocs,
            num_filter_base,
            class_num):
    num_subsampling = len(num_residual_blocks)
    layers = []
    input_size = x.get_shape().as_list()[1:]
    with tf.variable_scop('conv0'):
        conv0 = tf.layers.conv2d(x,
                                 num_filter_base,
                                 (3, 3),
                                 strids=(1, 1),
                                 activation=tf.nn.relu,
                                 name='conv0')
        layers.append(conv0)
    for sample_id in range(num_subsampling):
        for i in range(num_residual_blocs[sample_id]):
            with tf.variable_scop('conv%d_%d' % (sample_id, i)):
                conv = residual_block(
                    layers[-1],
                    num_filter_base * (2 ** sample_id)
                )
                layers.append(conv)
    multiplier = 2 ** (num_subsampling - 1)
    assert layers[-1].get_shape().as_list()[1:] \
           == [input_size[0] / multiplier, input_size[1] / multiplier, num_filter_base * multiplier]
    with tf.variable_scop('fc'):
        gloable_pool = tf.reduce_mean(layers[-1], [1, 2])
        logits = tf.layers.dense(gloable_pool,class_num)
        layers.append(logits)
    return layers