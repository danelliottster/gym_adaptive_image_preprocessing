import sys
sys.path.append('/home/dane/adaptive_pre_proc/gym_adaptive_img_pre_proc/gym_adaptive_img_pre_proc')

import time

from gym_adaptive_img_pre_proc import MnistClassifierAutoCorr, MnistCorrupted

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback

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

#create a custom callback class
class ShowOffCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(ShowOffCallback, self).__init__(verbose)
        self.iteration = 0
        self._fig = None

    def _on_rollout_start(self) -> bool:

        if self.num_timesteps % 10000 == 0:
            print(f"starting a new rollout, num_timesteps: {self.num_timesteps}")
            obs = self.training_env.reset()
            self._fig = env.render(fig_in=self._fig)
            terminated = False
            while not terminated:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, info = self.training_env.step(action)
                print(f"action: {action}, reward: {reward}")
                self._fig = env.render(fig_in=self._fig)
    
    def _on_training_start(self):
        pass

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass
    
show_off_callback = ShowOffCallback()

model = DQN("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=350000, log_interval=4 , callback=show_off_callback)
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