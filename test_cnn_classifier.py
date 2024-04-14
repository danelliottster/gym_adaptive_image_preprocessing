import sys
sys.path.append('/home/dane/adaptive_pre_proc/gym_adaptive_img_pre_proc/gym_adaptive_img_pre_proc')

from gym_adaptive_img_pre_proc import MnistClassifierAutoCorr, MnistCorrupted

classifier = MnistClassifierAutoCorr( root_dir_in='/home/dane/adaptive_pre_proc/gym_adaptive_img_pre_proc/gym_adaptive_img_pre_proc' )
classifier.train()
classifier.test()