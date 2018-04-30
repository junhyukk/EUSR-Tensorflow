import tensorflow as tf
import time, os
import numpy as np
from utils import cal_psnr, chop_forward, save_img, mod_crop

class Trainer():
    def __init__(self, sess, model, data, logger, args):
        self.sess = sess 
        self.model = model
        self.data = data        
        self.logger = logger
        self.args = args
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        if args.is_resume:
            self.model.load(sess)

    def train(self):
        print("Begin training...")
        whole_time = 0 
        init_global_step = self.model.global_step.eval(self.sess)
        for _ in range(init_global_step, self.args.num_iter):
            start_time = time.time()

            # For Track 2, 3, and 4: multi-scale => multi-degradation
            if self.args.degrade:
                self.data.scale_list = self.data.degra_list

            # randomly select scale in scale_list
            idx_scale = np.random.choice(len(self.data.scale_list))
            scale = self.data.scale_list[idx_scale]

            # get batch data and scale
            train_in_imgs, train_tar_imgs = self.data.get_batch(batch_size=self.args.num_batch, idx_scale=idx_scale)

            # train the network  
            feed_dict = {self.model.input: train_in_imgs, self.model.target: train_tar_imgs, self.model.flag_scale: scale}
            _, loss, lr, output, global_step = self.sess.run([self.model.train_op, self.model.loss, self.model.learning_rate, self.model.output, self.model.global_step], \
                                                            feed_dict=feed_dict)

            # check the duration of each iteration
            end_time = time.time()
            duration = end_time - start_time
            whole_time += duration
            mean_duration = whole_time / (global_step - init_global_step)   

            ############################################## print loss and duratin of training  ################################################
            if global_step % self.args.print_freq == 0:
                print('Loss: %d, Duration: %d / %d (%.3f sec/batch)' % (loss, global_step, self.args.num_iter, mean_duration))
            

            ############################################## log the loss, PSNR, and lr of training ##############################################
            if global_step % self.args.log_freq == 0:
                # calculate PSNR
                psnr = 0 
                for (out_img, tar_img) in zip(output, train_tar_imgs):
                    psnr += cal_psnr(out_img, tar_img, scale) / self.args.num_batch

                # write summary
                summaries_dict = {}
                summaries_dict['loss'] = loss
                summaries_dict['PSNR'] = psnr
                summaries_dict['lr'] = lr
                # summaries_dict['input'] = np.array(train_in_imgs)[:3]
                # summaries_dict['output'] = np.clip(np.round(output), 0.0, 255.0)[:3]
                # summaries_dict['target'] = np.array(train_tar_imgs)[:3]            
                self.logger.write(summaries_dict, global_step, is_train=True, idx_scale=idx_scale) 


            ######################################################## save the trained model ########################################################
            if global_step % self.args.save_freq == 0: 
                # save the trained model
                self.model.save(self.sess)  


            ############################################## log the PSNR of validation ############################################## 
            if global_step % self.args.valid_freq == 0: 
                # validation for all scale used
                for idx_scale, scale in enumerate(self.data.scale_list):
                    if self.args.degrade:
                        scale = 4

                    valid_in_imgs = self.data.dataset[idx_scale][-self.args.num_valid:]
                    if self.args.is_degrade:
                        valid_tar_imgs = self.data.dataset[idx_scale+len(self.data.scale_list)][-self.args.num_valid:]
                    else:
                        valid_tar_imgs = self.data.dataset[-1][-self.args.num_valid:]

                    # inference validation images & calculate PSNR
                    psnr = 0
                    for (in_img, tar_img) in zip(valid_in_imgs, valid_tar_imgs):
                        tar_img = mod_crop(tar_img, scale)
                        out_img = chop_forward(in_img, self.sess, self.model, scale=scale, shave=10)
                        psnr += cal_psnr(out_img, tar_img, scale) / self.args.num_valid

                    # write summary
                    summaries_dict = {}
                    summaries_dict['PSNR'] = psnr 
                    self.logger.write(summaries_dict, global_step, is_train=False, idx_scale=idx_scale) 
        print("Training is done!")

    def test(self):
        print("Begin test...")
        # load trained model
        self.model.load(self.sess)

        # inference validation images & calculate PSNR
        mean_runtime= 0
        mean_psnr = 0
        file_name_list = self.data.file_names_for_dirs[0]
        for (in_img, file_name) in zip(self.data.dataset[0], file_name_list):
            # inference 
            start_time = time.time()
            if self.args.degrade or self.args.no_self_ensemble:
                out_img = chop_forward(in_img, self.sess, self.model, scale=self.data.scale_list[0], shave=10)
                
            # Only for Track 1, we use geometric self-ensemble
            else:
                tmp_img = np.zeros([in_img.shape[0]*self.data.scale_list[0], in_img.shape[1]*self.data.scale_list[0], 3])
                for i in range(2):
                    if i == 0:
                        flip_img = np.fliplr(in_img)
                        for j in range(4):
                            rot_flip_img = np.rot90(flip_img, j)
                            out_img = chop_forward(rot_flip_img, self.sess, self.model, scale=self.data.scale_list[0], shave=10)
                            tmp_img += np.fliplr(np.rot90(out_img, 4-j)) 
                    else:
                        for k in range(4):
                            rot_img = np.rot90(in_img, k)
                            out_img = chop_forward(rot_img, self.sess, self.model, scale=self.data.scale_list[0], shave=10)
                            tmp_img += np.rot90(out_img, 4-k)
                out_img = tmp_img / 8
            end_time = time.time()
            mean_runtime += (end_time - start_time) / self.args.num_test

            # save images
            dir_img = os.path.join(self.args.exp_dir, self.args.exp_name, 'results', file_name)
            save_img(out_img, dir=dir_img)

        # write text file for summarize results
        dir_file = os.path.join(self.args.exp_dir, self.args.exp_name, 'results', 'results.txt')
        with open(dir_file, "w") as f:
            f.write("runtime per image [s] : {0:2.2f}\n".format(mean_runtime))
            f.write("CPU[1] / GPU[0] : 0\n")
            f.write("Extra Data [1] / No Extra Data [0] : 0")
        print("Test is done!")



