import sys
# sys.path.append('/home/dane/adaptive_pre_proc/gym_adaptive_img_pre_proc/gym_adaptive_img_pre_proc')
# sys.path.append('/home/dane/adaptive_pre_proc/gym_adaptive_img_pre_proc/')

import time
import matplotlib.pyplot as plt
from gym_adaptive_img_pre_proc import MnistCorrupted , MnistCorrupted_kmeans
from stable_baselines3.common.env_checker import check_env

# instantiate the environment
env = MnistCorrupted_kmeans( selected_corruptions=[ 'translation', 'rotation' , 'brightness' ] )

# render the means
fig_means = env.render_means()

# test reward function
env.reset()
observed_image = env.apply_cumulative_actions()
r = env.reward_function( observed_image.squeeze().numpy() )

# test full observation
observation_full = env.build_full_observation( observed_image.numpy() )

# plot the full observation
fig = plt.figure()
for i in range( env._K+1 ) :
    
    ax = fig.add_subplot( 5 , 11 , i+1 )
    ax.imshow( observation_full[ : , : , i ] , cmap='gray' )
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle( 'Full Observation' )
fig.tight_layout()
fig.subplots_adjust( top=0.9 )
fig.show()


# # test rendering
# # run reset first so there is something to render
# initial_image = env.reset()
# fig = env.render()
# fig = env.render( fig_in=fig )

# run the stable baselines environment checker
check_env( env )