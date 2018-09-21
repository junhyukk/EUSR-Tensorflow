from model.base_model import BaseModel
import tensorflow as tf

def create_model(args, parent=False):
    return EUSR(args)

class EUSR(BaseModel):
	def __init__(self, args):
		super(EUSR, self).__init__(args)
		self.build_model()
		self.init_saver()
		num_params = self.count_num_trainable_params()
		print("num_params: %d" % num_params)
		print("EUSR and Saver are initialized!")

	def build_model(self):
		# Initialization
		num_feats = self.args.num_feats
		num_res = self.args.num_res
		mean_shift = self.mean_shift
		conv = self.conv
		res_module = self.res_module
		scale_specific_module = self.scale_specific_module
		scale_specific_upsampler = self.scale_specific_upsampler

		# check scales used in experiment
		scale_list = list(map(lambda x: int(x), self.args.scale.split('+')))

		# Placeholder for input, target, and flag_scale
		self.input = tf.placeholder(tf.float32, [None, None, None, 3])
		self.target = tf.placeholder(tf.float32, [None, None, None, 3])
		self.flag_scale = tf.placeholder(tf.float32, []) 

		# Pre-processing
		in_img = mean_shift(self.input)
		tar_img = mean_shift(self.target)

        # First convolution layer 
		x = conv(in_img, num_feats, [3,3])

        # Scale-specific processing module
		if len(scale_list) > 1:
			x = tf.cond(tf.equal(self.flag_scale, 2), lambda: scale_specific_module(x, num_feats, [5,5]), \
	        lambda: tf.cond(tf.equal(self.flag_scale, 4), lambda: scale_specific_module(x, num_feats, [5,5]), \
	        lambda: scale_specific_module(x, num_feats, [5,5])))

		# Main branch
		x = res_module(x, num_feats, num_res)

		# Scale-speficifc up-sampling
		if len(scale_list) > 1: 
			x = tf.cond(tf.equal(self.flag_scale, 2), lambda: scale_specific_upsampler(x, 2), \
			lambda: tf.cond(tf.equal(self.flag_scale, 4), lambda: scale_specific_upsampler(x, 4), \
			lambda: scale_specific_upsampler(x, 8)))
		else:
			x = scale_specific_upsampler(x, int(self.args.scale))

		# Post-processing
		self.output = mean_shift(x, is_add=True) 

		# Loss & Training options
		with tf.name_scope("loss"):
			self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.target, self.output))
			self.learning_rate = tf.train.exponential_decay(self.args.init_lr, self.global_step, self.args.decay_step, self.args.decay_ratio, staircase=True)
			self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
