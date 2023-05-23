
import sys
sys.path.append('/home/dane/adaptive_pre_proc/gym_adaptive_img_pre_proc')

import time

from gym_adaptive_img_pre_proc import MnistClassifierAutoCorr, MnistCorrupted

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN

#instantiate the auto correlation classifier
test_auto_corr_classifier = MnistClassifierAutoCorr()

# train the classifier
test_auto_corr_classifier.train()

# plot the mean images
test_auto_corr_classifier.plot_mean_images()

# instantiate the environment
env = MnistCorrupted( test_auto_corr_classifier )

# test rendering
# run reset first so there is something to render
initial_image = env.reset()
fig = env.render()
fig = env.render( fig_in=fig )

# test environment step
test_action = env.action_space.sample()
new_image, reward, done, info = env.step( test_action , verbose=True )
fig = env.render( fig_in=fig )


# run the stable baselines environment checker
check_env( env )

# test the classifier
test_auto_corr_classifier.test()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=150000, log_interval=4)
model.save("test_blah")

obs = env.reset()
fig = env.render()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, info = env.step(action)
    fig = env.render(fig_in=fig)
    time.sleep(1)
    if terminated:
        obs = env.reset()