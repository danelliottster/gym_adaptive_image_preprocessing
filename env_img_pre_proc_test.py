import sys
sys.path.append('/home/dane/adaptive_pre_proc/gym_adaptive_img_pre_proc/gym_adaptive_img_pre_proc')

import time

from gym_adaptive_img_pre_proc import MnistClassifierAutoCorr, MnistCorrupted
from stable_baselines3.common.env_checker import check_env


#instantiate the auto correlation classifier
test_auto_corr_classifier = MnistClassifierAutoCorr()

# train the classifier
test_auto_corr_classifier.train()

# plot the mean images
test_auto_corr_classifier.plot_mean_images()

# instantiate the environment
env = MnistCorrupted( test_auto_corr_classifier , selected_corruptions=[ 'translation', 'rotation' , 'brightness' ] )

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

# test the trial creation ability by resetting the environment several times and rendering each time
print('Testing trial creation ability')
fig = None
for i in range(10):
    env.reset()
    fig = env.render(fig_in=fig)
    time.sleep(1)

# test brightness action
print('Testing shift left action')

fig = env.render()
time.sleep(1)
for i in range(10):
    new_image, reward, done, info = env.step( 0 )
    fig = env.render(fig_in=fig)
    time.sleep(1)

print('Testing shift right action')
env.reset()
fig = env.render()
time.sleep(1)
for i in range(10):
    new_image, reward, done, info = env.step( 1 )
    fig = env.render(fig_in=fig)
    time.sleep(1)

print('Testing shift up action')
env.reset()
fig = env.render()
time.sleep(1)
for i in range(10):
    new_image, reward, done, info = env.step( 2 )
    fig = env.render(fig_in=fig)
    time.sleep(1)

print('Testing shift down action')
env.reset()
fig = env.render()
time.sleep(1)
for i in range(10):
    new_image, reward, done, info = env.step( 3 )
    fig = env.render(fig_in=fig)
    time.sleep(1)

print('Testing rotate counter-clockwise action')
env.reset()
fig = env.render()
time.sleep(1)
for i in range(20):
    new_image, reward, done, info = env.step( 4 )
    fig = env.render(fig_in=fig)
    time.sleep(1)

print('Testing rotate clockwise action')
env.reset()
fig = env.render()
time.sleep(1)
for i in range(20):
    new_image, reward, done, info = env.step( 5 )
    fig = env.render(fig_in=fig)
    time.sleep(1)

print('Testing darken action')
env.reset()
fig = env.render()
time.sleep(1)
for i in range(20):
    new_image, reward, done, info = env.step( 6 )
    fig = env.render(fig_in=fig)
    time.sleep(1)

print('Testing lighten action')
env.reset()
fig = env.render()
time.sleep(1)
for i in range(20):
    new_image, reward, done, info = env.step( 7 )
    fig = env.render(fig_in=fig)
    time.sleep(1)

print('Testing creation of kmeans environment')
env = MnistCorrupted_kmeans( test_auto_corr_classifier , selected_corruptions=[ 'translation', 'rotation' , 'brightness' ] , kmeans=True )