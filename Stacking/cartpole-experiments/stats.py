import json
import numpy as np
import matplotlib.pyplot as plt




measures1=['loss', 'mean_squared_error', 'mean_q', 'episode_reward','nb_steps', 'duration' ]
measures=['episode_reward']
models=["SARSA","DQN"]


for measure in measures:
    for i in range(6):
        for model in models:
            with open('log/'+model+'_agent'+str(i)+'_train.json') as f:
                    data = json.load(f)
                
            array=list(np.convolve(data[measure], np.ones(50)/50, mode='valid'))
            plt.plot(array,label= model+"A-"+str(i))
            plt.ylabel("Measure: "+measure)
            plt.xlabel("Episodes")
    plt.title(measure)
    plt.legend()
    
    plt.savefig('graphs/'+measure+'_DQN-SARSA_all')
    plt.close()

# for model in models:
#     for measure in measures:
#         for i in range(6):
#             with open('log/'+model+'_agent'+str(i)+'_train.json') as f:
#                 data = json.load(f)
            
#             #array=list(np.convolve(data[measure], np.ones(50)/50, mode='valid'))
#             plt.plot(data[measure],label="Agent-"+str(i))
#             plt.ylabel("Measure: "+measure)
#             plt.xlabel("Episodes")
#         plt.title(model+" -- "+measure)
#         plt.legend()
        
#         plt.savefig('graphs/'+model+"_"+measure)
#         plt.close()

# measures=[ 'episode_reward','nb_steps']
# models=["SARSA","DQN"]
# for model in models:
#     for measure in measures:
#         for i in range(6):
#             with open('log/'+model+'_agent'+str(i)+'_test.json') as f:
#                 data = json.load(f)
            
#             #array=list(np.convolve(data[measure], np.ones(50)/50, mode='valid'))
#             plt.plot(data[measure],label="Agent-"+str(i))
#             plt.ylabel("Measure: "+measure)
#             plt.xlabel("Episodes")
#         plt.title('Test  '+model+" -- "+measure)
#         plt.legend()
        
#         plt.savefig('graphs/'+model+"_test_"+measure)
#         plt.close()

