import numpy as np
import matplotlib.pyplot as plt
import texas_holdem_mod as texas_holdem
from stable_baselines3.common.monitor import Monitor
from tabulate import tabulate
from scipy.interpolate import make_interp_spline
import os 
 
class graph_metrics():
    """
    A class for creating and plotting various metrics and statistics for multiple graph models.

    Args:
    n_models (int): The number of graph models.
    storage (dict): A dictionary containing various metrics and statistics for training.
    storageB (dict): A dictionary containing additional metrics and statistics for training, or None if not available.
    Used for random opponent benchmark
    figsize (tuple): A tuple specifying the size of the plots (width, height).
    t_steps (int): The number of training steps per generation.
    overlay (bool): If True, overlay multiple plots; otherwise, create separate plots for each model.
    e_steps (int): The number of evaluation steps per generation.
    title (str): The title to be used for saving the plots.
    device (str): The device type, 'mac' or 'pc', to determine the file path for saving the plots.

    Attributes:
    num_graphs (int): The number of graph models.
    storage (dict): A dictionary containing various metrics and statistics for training.
    storageB (dict): A dictionary containing additional metrics and statistics for training .Used for random opponent benchmark
    figsize (tuple): A tuple specifying the size of the plots (width, height).
    n_SP_gens (int): The number of self-play generations.
    t_steps (int): The number of training steps per generation.
    overlay (bool): If True, overlay multiple plots; otherwise, create separate plots for each model.
    e_steps (int): The number of evaluation steps per generation.
    title (str): The title to be used for saving the plots.
    device (str): The device type, 'mac' or 'pc', to determine the file path for saving the plots.

    Methods:
    create_x_y(): Initialize and organize data for plotting.
    plot_rewards(train, eval): Plot rewards for training and evaluation.
    plot_moving_rewards(train, eval): Plot moving rewards for training and evaluation.
    plot_moving_mean(train, eval, trim): Plot moving mean for training and evaluation.

    comb_plot_loss(): Plot combined training loss across selfplay generations.
    comb_plot_rewards(train, eval): Plot combined rewards for training and evaluation.
    comb_plot_moving_total(train, eval): Plot combined moving total rewards for training and evaluation across selfplay generations.
    comb_plot_moving_mean(train, eval): Plot combined moving mean rewards for training and evaluation across selfplay generations
    comb_plot_opt_acts(sing_opt_acts): Plot combined optimal action percentages.
    plot_sims(): Plot similarity, in terms of KLdivergnece data, for various poker hands.
    print_all_graphs(): Print all graphs.
    print_select_graphs(): Print specified graphs.

    """
    def __init__(self, n_models, storage, storageB,figsize, t_steps, overlay, e_steps, title, device ): 
        self.num_graphs = n_models
        self.storage = storage
        self.storageB = storageB
        self.figsize = figsize
        self.n_SP_gens = n_models
        self.t_steps = t_steps
        self.overlay = overlay
        self.e_steps = e_steps
        self.title = title 
        self.device = device 
        
        if self.device == 'mac':
            self.direct = '/Users/rhyscooper/Desktop/MSc Project/Pages/plots3/' + self.title
            if not os.path.exists(self.direct):
                os.makedirs(self.direct)
        elif self.device == 'pc':
            self.direct = os.path.join('S:\\MSC_proj\\plots', self.title )
            if not os.path.exists(self.direct):
                os.makedirs(self.direct)
        
    def create_x_y(self):
        same_keys = self.check_dicts_have_same_keys(self.storage)
        self. dict_attributes = [attr for attr in dir(self.storage) if isinstance(getattr(self.storage, attr), dict)]
        self.storage_ids = list(getattr(self.storage, self.dict_attributes[1]).keys())
        
        self.train_moving_mean = {}
        self.train_moving_total = {}
        # self.train_losses = {}
        self.train_rewards = {}
        
        self.train_rand_op_moving_mean = {}
        self.train_rand_op_moving_total = {}
        self.train_value_losses = [[], []]
        self.train_policy_losses = [[], []]
        self.train_entropy_losses = [[], []]
        self.train_rand_op_rewards = {}
        
        self.eval_moving_mean = {}
        self.eval_moving_total = {}
        self.eval_rewards = {}
        
        self.eval_rand_op_moving_mean = {}
        self.eval_rand_op_moving_total = {}
        self.eval_rand_op_rewards = {}
        
         # combined plot
        self.comb_train_moving_mean = [[], []]
        self.comb_train_moving_total = [[], [0.0]]
        self.comb_train_value_losses = [[], []]
        self.comb_train_policy_losses = [[], []]
        self.comb_train_entropy_losses = [[], []]
        self.comb_train_rewards = [[], []]
        
        self.comb_train_rand_op_moving_mean = [[], []]
        self.comb_train_rand_op_moving_total = [[], [0.0]]
        self.comb_train_rand_op_losses = [[], []]
        self.comb_train_rand_op_rewards = [[], []]
        
        self.comb_eval_moving_mean = [[], []]
        self.comb_eval_moving_total = [[], [0.0]]
        self.comb_eval_rewards = [[], []]
        
        self.comb_eval_rand_op_moving_mean = [[], []]
        self.comb_eval_rand_op_moving_total = [[], [0.0]]
        self.comb_eval_rand_op_rewards = [[], []]
        
        
        self.comb_percentages_ag = [[], []]
        self.sep_percentages_ag = [[], []]
        self.sep_percentages_op = [[], []]
        
        self.HC_sims= [[], []]
        self.STR_sims = [[], []]
        self.RF_sims= [[], []]
        self.PR_sims = [[], []]
        self.FLSH_sims =[[], []]
        self.STR_FLSH_sims=[[], []]
        

     
        for key in self.storage.sims.keys():
            self.HC_sims[1].append(self.storage.sims[key]['HC'])
            self.STR_sims[1].append(self.storage.sims[key]['STR'])
            self.RF_sims[1].append(self.storage.sims[key]['RF'])
            self.PR_sims[1].append(self.storage.sims[key]['PR'])
            self.FLSH_sims[1].append(self.storage.sims[key]['FLSH'])
            self.STR_FLSH_sims[1].append(self.storage.sims[key]['STR_FLSH'])

            
        self.HC_sims[0] = [i for i in range(len(self.HC_sims[1]))]
        self.STR_sims[0] = [i for i in range(len(self.STR_sims[1]))]
        self.RF_sims[0] = [i for i in range(len(self.RF_sims[1]))]
        self.PR_sims[0] = [i for i in range(len(self.PR_sims[1]))]
        self.FLSH_sims[0] = [i for i in range(len(self.FLSH_sims[1]))]
        self.STR_FLSH_sims[0] = [i for i in range(len(self.STR_FLSH_sims[1]))]


                 
        # combine all percentages
        for key in self.storage.percentages_ag.keys():
            self.comb_percentages_ag[1].extend(self.storage.percentages_ag[key])
            # if key == 'PPOop' or 'A2Cop':
            
        for key in self.storage.percentages_ag.keys():
            if key == 'PPO' or key == '1' or key == '2' or key == '3' or key == '4':
                self.sep_percentages_ag[1] = (self.storage.percentages_ag[key])
            if key == 'PPOop' or 'A2Cop'or key == '1' or key == '2' or key == '3' or key == '4':    
        # for sep percentages, eval op round has replaced original dict values     
                self.sep_percentages_op[1] = (self.storage.percentages_op[key])

                
                
            
        # create x for percentages
        self.comb_percentages_ag[0] = [i for i in range(1, len(self.comb_percentages_ag[1])+1)]  
        self.sep_percentages_ag[0] = [i for i in range(1, len(self.sep_percentages_ag[1])+1)]  
        self.sep_percentages_op[0] =  [i for i in range(1, len(self.sep_percentages_op[1])+1)]      

        for key in self.storage_ids:
            self.train_moving_mean[key] = ([i for i in range(1, len(self.storage.gen_train_moving_mean_reward[key])+1)], self.storage.gen_train_moving_mean_reward[key])
            self.train_moving_total[key] = ([i for i in range(1, len(self.storage.gen_train_moving_total[key])+1)], self.storage.gen_train_moving_total[key])
            # self.train_losses[key] = ([i for i in range(1, len(self.storage.gen_train_losses[key])+1)], self.storage.gen_train_losses[key]) 
            self.train_rewards[key] = ([i for i in range(1, len(self.storage.gen_train_rewards[key])+1)], self.storage.gen_train_rewards[key])
            
            if self.storageB is not None:
                self.train_rand_op_moving_mean[key]= ([i for i in range(1, len(self.storageB.gen_train_moving_mean_reward[key])+1)], self.storageB.gen_train_moving_mean_reward[key])
                self.train_rand_op_moving_total[key]= ([i for i in range(1, len(self.storageB.gen_train_moving_total[key])+1)], self.storageB.gen_train_moving_total[key])
                # self.train_rand_op_losses[key]= ([i for i in range(1, len(self.storageB.gen_train_losses[key])+1)], self.storageB.gen_train_losses[key])
                self.train_rand_op_rewards[key]= ([i for i in range(1, len(self.storageB.gen_train_rewards[key])+1)], self.storageB.gen_train_rewards[key])
            
            self.eval_moving_mean[key] = ([i for i in range(1, len(self.storage.gen_eval_moving_mean_reward[key])+1)], self.storage.gen_eval_moving_mean_reward[key])
            self.eval_moving_total[key] = ([i for i in range(1, len(self.storage.gen_eval_moving_total[key])+1)], self.storage.gen_eval_moving_total[key])
            self.eval_rewards[key] = ([i for i in range(1, len(self.storage.gen_eval_rewards[key])+1)], self.storage.gen_eval_rewards[key])

            if self.storageB is not None:
                self.eval_rand_op_moving_mean[key]= ([i for i in range(1, len(self.storageB.gen_eval_moving_mean_reward[key])+1)], self.storageB.gen_eval_moving_mean_reward[key])
                self.eval_rand_op_moving_total[key]= ([i for i in range(1, len(self.storageB.gen_eval_moving_total[key])+1)], self.storageB.gen_eval_moving_total[key])
                self.eval_rand_op_rewards[key]= ([i for i in range(1, len(self.storageB.gen_eval_rewards[key])+1)], self.storageB.gen_eval_rewards[key])
            
            
            self.comb_train_rewards[1].extend(self.train_rewards[key][1])
            self.comb_eval_rewards[1].extend(self.eval_rewards[key][1])

            if self.storage.gen_train_value_losses[key][0] != None:
                self.comb_train_value_losses[1].extend(self.storage.gen_train_value_losses[key])
                self.comb_train_policy_losses[1].extend(self.storage.gen_train_policy_losses[key])
                self.comb_train_entropy_losses[1].extend(self.storage.gen_train_entropy_losses[key])
            
            if self.storageB is not None:
                self.comb_train_rand_op_rewards[1].extend(self.train_rand_op_rewards[key][1])
                self.comb_eval_rand_op_rewards[1].extend(self.eval_rand_op_rewards[key][1])
                # self.comb_train_rand_op_losses[1].extend(self.train_rand_op_losses[key][1])
          
        if len(self.comb_train_rewards[1]) > 1 and self.comb_train_rewards[1][1]!= None:    
            for i in range(0, len(self.comb_train_rewards[1])):
                # self.comb_train_moving_mean[1].append(self.comb_train_rewards[1][i] / i+1)
                self.comb_train_moving_mean[1].append(np.mean(self.comb_train_rewards[1][:i+1]))
                self.comb_train_moving_total[1].append(self.comb_train_moving_total[1][i] + self.comb_train_rewards[1][i])

        if len(self.comb_eval_rewards[1]) > 1 and self.comb_train_rewards[1][1]!= None:
            for i in range(0, len(self.comb_eval_rewards[1])):    
                self.comb_eval_moving_mean[1].append(np.mean(self.comb_eval_rewards[1][:i+1]))
                self.comb_eval_moving_total[1].append(self.comb_eval_moving_total[1][i] + self.comb_eval_rewards[1][i] )
                
        if len(self.comb_train_rand_op_rewards[1]) > 1 and self.comb_train_rewards[1][1]!= None:
            for i in range(0, len(self.comb_train_rand_op_rewards[1])):
                # self.comb_train_moving_mean[1].append(self.comb_train_rewards[1][i] / i+1)
                self.comb_train_rand_op_moving_mean[1].append(np.mean(self.comb_train_rand_op_rewards[1][:i+1]))
                self.comb_train_rand_op_moving_total[1].append(self.comb_train_rand_op_moving_total[1][i] + self.comb_train_rand_op_rewards[1][i])
        
        if len(self.comb_eval_rewards[1]) > 1 and self.comb_train_rewards[1][1]!= None:
            for i in range(0, len(self.comb_eval_rand_op_rewards[1])):    
                self.comb_eval_rand_op_moving_mean[1].append(np.mean(self.comb_eval_rand_op_rewards[1][:i+1]))
                self.comb_eval_rand_op_moving_total[1].append(self.comb_eval_rand_op_moving_total[1][i] + self.comb_eval_rand_op_rewards[1][i] )    
        
            
            

        if self.comb_train_rewards[1][1]!= None:
            self.comb_train_moving_total[1].pop(0)  
            self.comb_eval_moving_total[1].pop(0) 
            self.comb_train_rand_op_moving_total[1].pop(0)
            if len(self.comb_eval_rewards[0]) > 1:
                self.comb_eval_rand_op_moving_total[1].pop(0)                                   
            

        self.comb_train_moving_mean[0] = [i for i in range(1, len(self.comb_train_moving_mean[1])+1)]    
        self.comb_train_moving_total[0]= [i for i in range(1, len(self.comb_train_moving_total[1])+1)]
        self.comb_train_rewards[0] = [i for i in range(1, len(self.comb_train_rewards[1])+1)]
        
        self.comb_train_rand_op_moving_mean[0] = [i for i in range(1, len(self.comb_train_rand_op_moving_mean[1])+1)]    
        self.comb_train_rand_op_moving_total[0]= [i for i in range(1, len(self.comb_train_rand_op_moving_total[1])+1)]
        self.comb_train_rand_op_losses[0] = [i for i in range(1, len(self.comb_train_rand_op_losses[1])+1)]
        self.comb_train_rand_op_rewards[0] = [i for i in range(1, len(self.comb_train_rand_op_rewards[1])+1)]
        
        
        self.comb_eval_moving_mean[0]= [i for i in range(1, len(self.comb_eval_moving_mean[1])+1)]
        self.comb_eval_moving_total[0]= [i for i in range(1, len(self.comb_eval_moving_total[1])+1)]
        self.comb_eval_rewards[0]= [i for i in range(1, len(self.comb_eval_rewards[1])+1)]
        
        self.comb_eval_rand_op_moving_mean[0]= [i for i in range(1, len(self.comb_eval_rand_op_moving_mean[1])+1)]
        self.comb_eval_rand_op_moving_total[0]= [i for i in range(1, len(self.comb_eval_rand_op_moving_total[1])+1)]
        self.comb_eval_rand_op_rewards[0]= [i for i in range(1, len(self.comb_eval_rand_op_rewards[1])+1)]
        
        self.comb_train_value_losses[0] = [i for i in range(1, len(self.comb_train_value_losses[1])+1)]
        self.comb_train_policy_losses[0] = [i for i in range(1, len(self.comb_train_policy_losses[1])+1)]
        self.comb_train_entropy_losses[0] = [i for i in range(1, len(self.comb_train_entropy_losses[1])+1)]

    def plot_rewards(self, train, eval):
        file_path = os.path.join(self.direct, 'sing_rewards.png')
        if not self.overlay:
            fig4, axs4 = plt.subplots(self.num_graphs, 2, sharey= 'col', figsize = self.figsize)
            for i, key in enumerate(self.storage_ids):
                if train:
                    axs4[i, 0].plot(self.train_rewards[key][0], self.train_rewards[key][1], color = 'b', label= 'reward')
                    axs4[i, 0].axhline(y=np.mean(self.train_rewards[key][1]), color='r', linestyle='-', label='mean reward')
                    axs4[i, 0].axhline(y=-0.5, color='g', linestyle='--', label='random')
                    axs4[i, 0].set_title(str(key) + ' training Reward')
                    axs4[i, 0].legend(fontsize='small')
                if eval:
                    axs4[i, 1].plot(self.eval_rewards[key][0], self.eval_rewards[key][1], color = 'b' , label= 'reward')
                    axs4[i, 1].axhline(y=np.mean(self.eval_rewards[key][1]), color='r', linestyle='-', label='mean reward')
                    axs4[i, 1].axhline(y=-0.5, color='g', linestyle='--', label='random')
                    axs4[i, 1].set_title(str(key) + ' evaluation Reward')
                    axs4[i, 1].legend(fontsize='small')
            plt.tight_layout()
            fig4.savefig(file_path)
            plt.show()  
        else:
            fig4, axs4 = plt.subplots(1, 1, sharey= 'col', figsize = self.figsize)
            for key in (self.storage_ids):
                if train:
                    axs4.plot(self.train_rewards[key][0], self.train_rewards[key][1], color = 'b', label= 'reward')
                    axs4.axhline(y=np.mean(self.train_rewards[key][1]), color='r', linestyle='-', label='mean reward')
                    axs4.set_title(' training Reward')

                if eval:
                    if key == 'PPO': 
                        colourP = 'b'
                    else:
                        colourP = 'g'
                    axs4.plot(self.eval_rewards[key][0], self.eval_rewards[key][1], color = colourP , label= str(key) + '_reward')
                    axs4.set_title('evaluation Reward')
    
            axs4.legend(fontsize='small')
            plt.tight_layout()
            fig4.savefig(file_path)
            plt.show()  
            
    def plot_moving_rewards(self, train, eval):
        file_path = os.path.join(self.direct, 'sing_mov_rewards.png')
        if not self.overlay:
            fig2, axs2 = plt.subplots(self.num_graphs, 2, sharey= 'col',figsize = self.figsize) 
        
            for i, key in enumerate(self.storage_ids):
                if train:
                    axs2[i, 0].plot(self.train_moving_total[key][0], self.train_moving_total[key][1],  color = 'b', label= 'moving total')
                    axs2[i, 0].axhline(y=np.mean(self.train_moving_total[key][1]), color='r', linestyle='-', label='mean')
                    if self.train_rand_op_moving_total:
                        axs2[i, 0].plot(self.train_rand_op_moving_total[key][0], self.train_rand_op_moving_total[key][1],  color = 'g', label= 'rand')
                    axs2[i, 0].set_title('train moving Reward')
                    axs2[i, 0].legend(fontsize='small')
                if eval:
                    axs2[i, 1].plot(self.eval_moving_total[key][0], self.eval_moving_total[key][1], color = 'b', label= 'moving total')
                    if self.eval_rand_op_moving_total:
                        axs2[i, 1].plot(self.eval_rand_op_moving_total[key][0], self.eval_rand_op_moving_total[key][1],  color = 'g', label= 'random')
                    axs2[i, 1].set_title('evaluation moving Reward')
                    axs2[i, 1].legend(fontsize='small')
            plt.tight_layout()
            fig2.savefig(file_path)
            plt.show()
            
        else:
            fig4, axs4 = plt.subplots(1, 1, sharey= 'col', figsize = self.figsize)
            for key in (self.storage_ids):
                if train:
                    axs4.plot(self.train_moving_total[key][0], self.train_moving_total[key][1], color = 'b', label= 'reward')
                    axs4.axhline(y=np.mean(self.train_moving_total[key][1]), color='r', linestyle='-', label='mean reward')
                    axs4.set_title(' training Reward')

                if eval:
                    if key == 'PPO': 
                        colourP = 'b'
                    else:
                        colourP = 'g'
                    axs4.plot(self.eval_moving_total[key][0], self.eval_moving_total[key][1], color = colourP , label= str(key) + '_moving reward')
                    axs4.set_title('evaluation moving Reward')
            axs4.legend(fontsize='small')
            plt.tight_layout()
            fig4.savefig(file_path)
            plt.show()                  
    
    def plot_moving_mean(self, train, eval, trim):
        file_path = os.path.join(self.direct, 'sing_mov_mean.png')
        if not self.overlay:
            fig1, axs1 = plt.subplots(self.num_graphs, 2, sharey= 'col', figsize = self.figsize)
            for i, key in enumerate(self.storage_ids):
                if train:
                    axs1[i, 0].plot(self.train_moving_mean[key][0], self.train_moving_mean[key][1],  color = 'b', label= 'moving mean')
                    axs1[i, 0].axhline(y=np.mean(self.train_moving_mean[key][1]), color='r', linestyle='-', label='average')
                    axs1[i, 0].axhline(y=-0.5, color='g', linestyle='--', label='random')
                    axs1[i, 0].set_title(str(key) + ' moving mean')
                    axs1[i, 0].legend(fontsize='small')
                if eval:
                    axs1[i, 1].plot(self.eval_moving_mean[key][0], self.eval_moving_mean[key][1], color = 'b', label= 'moving mean')
                    axs1[i, 1].axhline(y=np.mean(self.eval_moving_mean[key][1]), color='r', linestyle='-', label='average')
                    axs1[i, 1].axhline(y=-0.5, color='g', linestyle='--', label='random')
                    axs1[i, 1].set_title(str(key) + ' moving mean')
                    axs1[i, 1].legend(fontsize='small')   
            plt.tight_layout()
            fig1.savefig(file_path)
            plt.show()

            
        else:
            if not trim:
                trim = 400
                fig4, axs4 = plt.subplots(1, 1, sharey= 'col', figsize = self.figsize)
                for key in (self.storage_ids):
                    if train:
                        axs4.plot(self.train_moving_mean[key][0], self.train_moving_mean[key][1], color = 'b', label= 'reward')
                        axs4.axhline(y=np.mean(self.train_moving_mean[key][1]), color='r', linestyle='-', label='mean reward')
                        axs4.set_title(' training Reward')

                    if eval:
                        if key == 'PPO': 
                            colourP = 'b'
                            colourM = 'lightblue'
                        else:
                            colourP = 'g'
                            colourM = 'lightgreen'
    

                        axs4.plot(self.eval_moving_mean[key][0], self.eval_moving_mean[key][1], color = colourP , label= str(key) + '_mean reward')
                        axs4.axhline(y=np.mean(self.eval_rewards[key][1]), color=colourM, linestyle='--', label=str(key) +'_average: ' + str(np.mean(self.eval_rewards[key][1])))
                        axs4.set_title('evaluation mean reward')
            if trim:
                    fig4, axs4 = plt.subplots(1, 1, sharey= 'col', figsize = self.figsize)
                    for key in (self.storage_ids):
                        if train:
                            axs4.plot(self.train_moving_mean[key][0][trim:] , self.train_moving_mean[key][1][trim:] , color = 'b', label= 'reward')
                            axs4.axhline(y=np.mean(self.train_moving_mean[key][1]), color='r', linestyle='-', label='mean reward')
                            axs4.set_title(' training Reward')
                        if eval:
                            if key == 'PPO': 
                                colourP = 'b'
                                colourM = 'lightblue'
                            else:
                                colourP = 'g'
                                colourM = 'lightgreen'
                            axs4.plot(self.eval_moving_mean[key][0][trim:] , self.eval_moving_mean[key][1][trim:] , color = colourP , label= str(key) + '_mean reward')
                            axs4.axhline(y=np.mean(self.eval_rewards[key][1][trim:] ), color=colourM, linestyle='--', label=str(key) +'_average: ' + str(np.mean(self.eval_rewards[key][1][trim:] )))
                            axs4.set_title('evaluation mean reward')
            axs4.legend(fontsize='small')
            plt.tight_layout()
            fig4.savefig(file_path)
            plt.show()                     
                
    def comb_plot_loss(self):
        file_path = os.path.join(self.direct, 'comb_loss.png')
        fig5, axs5 = plt.subplots(1, 1, sharey='col', figsize = self.figsize)
        axs5.plot(self.comb_train_value_losses[0], self.comb_train_value_losses[1], label='value loss', color = 'red')
        axs5.plot(self.comb_train_policy_losses[0], self.comb_train_policy_losses[1], label='policy loss', color = 'green')
        axs5.plot(self.comb_train_entropy_losses[0], self.comb_train_entropy_losses[1], label='entropy loss', color = 'blue')
        axs5.set_title(' training loss')
        axs5.legend(fontsize='small')    
        plt.tight_layout()
        fig5.savefig(file_path)  
        plt.show()

    def comb_plot_rewards(self, train, eval):
        file_path = os.path.join(self.direct, 'sing_comb_rewards.png')
        fig2, axs2 = plt.subplots(2, 1,figsize = self.figsize) 
        if train:
            axs2[0].axhline(y=np.mean(self.comb_train_rewards[1]), color='r', linestyle='-', label='mean')
            axs2[0].plot(self.comb_train_rand_op_rewards[0], self.comb_train_rand_op_rewards[1],  color = 'g', label= 'rand')
            axs2[0].plot(self.comb_train_rewards[0], self.comb_train_rewards[1],  color = 'b', label= 'rewards')
            axs2[0].set_title(' train Rewards')
            axs2[0].legend(fontsize='small')
        if eval:
            axs2[1].plot(self.comb_eval_rand_op_rewards[0], self.comb_eval_rand_op_rewards[1],  color = 'g', label= 'rand')
            axs2[1].plot(self.comb_eval_rewards[0], self.comb_eval_rewards[1], color = 'b', label= 'rewards')
            axs2[1].axhline(y=np.mean(self.comb_eval_rewards[1]), color='r', linestyle='-', label='mean')
            axs2[1].set_title('eval rewards')
            axs2[1].legend(fontsize='small')
        fig2.savefig(file_path)
        plt.show()
        
    def comb_plot_moving_total(self, train, eval):
        file_path = os.path.join(self.direct, 'mov_comb_rewards.png')
        fig2, axs2 = plt.subplots(2, 1,figsize = self.figsize) 
        if train:
            axs2[0].plot(self.comb_train_moving_total[0], self.comb_train_moving_total[1],  color = 'b', label= 'moving total')
            axs2[0].plot(self.comb_train_rand_op_moving_total[0], self.comb_train_rand_op_moving_total[1],  color = 'g', label= 'rand')
            for j in range(1,self.n_SP_gens+1):
                axs2[0].axvline(x=self.t_steps *j, color='y', linestyle='--')
            axs2[0].set_title(' train moving Reward')
            axs2[0].legend(fontsize='small')
        if eval:
            axs2[1].plot(self.comb_eval_moving_total[0], self.comb_eval_moving_total[1], color = 'b', label= 'moving total')
            axs2[1].plot(self.comb_eval_rand_op_moving_total[0], self.comb_eval_rand_op_moving_total[1],  color = 'g', label= 'rand')
            for j in range(1,self.n_SP_gens+1):
                axs2[1].axvline(x=self.e_steps*j, color='y', linestyle='--')
            axs2[1].set_title('eval moving Reward')
            axs2[1].legend(fontsize='small')
        plt.tight_layout()
        fig2.savefig(file_path)
        plt.show()
        
    def comb_plot_moving_mean(self, train, eval):
        file_path = os.path.join(self.direct, 'comb_mov_mean.png')
        fig2, axs2 = plt.subplots(2, 1,figsize = self.figsize) 
        if train:
            axs2[0].plot(self.comb_train_moving_mean[0], self.comb_train_moving_mean[1],  color = 'b', label= 'moving mean')
            axs2[0].axhline(y=-0.5, color='g', linestyle='--', label='random')
            for j in range(1,self.n_SP_gens+1):
                axs2[0].axvline(x=self.t_steps *j, color='y', linestyle='--')
            axs2[0].set_title(' train moving mean')
            axs2[0].legend(fontsize='small')
        if eval:
            axs2[1].plot(self.comb_eval_moving_mean[0], self.comb_eval_moving_mean[1], color = 'b', label= 'moving mean')
            axs2[1].axhline(y=-0.5, color='g', linestyle='--', label='random')
            for j in range(1,self.n_SP_gens+1):
                axs2[1].axvline(x=self.e_steps *j, color='y', linestyle='--')
            axs2[1].set_title('eval moving mean')
            axs2[1].legend(fontsize='small')
        plt.tight_layout()
        fig2.savefig(file_path)
        plt.show()    

    def comb_plot_opt_acts(self, sing_opt_acts):
        file_path = os.path.join(self.direct, 'opt_acts.png')
        if not self.overlay:
            if sing_opt_acts:           
                fig5, axs5 = plt.subplots(1, 1, sharey='col', figsize = self.figsize)
                axs5.plot(self.sep_percentages_ag[0], self.sep_percentages_ag[1], label='agent optimal action rate')
                axs5.set_title('optimal action percentages')
                axs5.legend(fontsize='small')
                plt.tight_layout()
                fig5.savefig(file_path)  
                plt.show()  
            else:
                fig5, axs5 = plt.subplots(1, 1, sharey='col', figsize = self.figsize)
                axs5.plot(self.comb_percentages_ag[0], self.comb_percentages_ag[1], label='optimal action rate')
                axs5.set_title('optimal action percentages across generations')
                axs5.legend(fontsize='small')
                plt.tight_layout()
                fig5.savefig(file_path)    
                plt.show()
         
        else:
            fig5, axs5 = plt.subplots(1, 1, sharey='col', figsize = self.figsize)
            axs5.plot(self.sep_percentages_ag[0], self.sep_percentages_ag[1], label='agent optimal action rate')
            axs5.plot(self.sep_percentages_op[0], self.sep_percentages_op[1], label='opponent optimal action rate')
            axs5.set_title('optimal action percentages')
            axs5.legend(fontsize='small')
            plt.tight_layout()
            fig5.savefig(file_path)    
            plt.show()    
  
    def plot_sims(self):       
        if len(self.HC_sims[0])>1:
            fig5, axs5 = plt.subplots(1, 1, figsize = self.figsize)
            axs5.plot(self.HC_sims[0], self.HC_sims[1], label='High Card (10)')
            axs5.plot(self.PR_sims[0], self.PR_sims[1], label='Pair (9) ')
            axs5.plot(self.STR_sims[0], self.STR_sims[1], label='Straight (6)')
            axs5.plot(self.FLSH_sims[0], self.FLSH_sims[1], label='Flush (5)')
            axs5.plot(self.STR_FLSH_sims[0], self.STR_FLSH_sims[1], label='Straight Flush (2)')
            axs5.plot(self.RF_sims[0], self.RF_sims[1], label='Royal Flush (1)')
            axs5.set_title('action distribution KL divergence')
            axs5.legend(fontsize='small')
            plt.tight_layout()  
            plt.show()
            
            data = [
            ['High Card'] + self.HC_sims[1],
            ['Pair'] + self.PR_sims[1],
            ['Straight'] + self.STR_sims[1],
            ['Flush'] + self.FLSH_sims[1],
            ['Straight Flush'] + self.STR_FLSH_sims[1],
            ['Royal Flush'] + self.RF_sims[1]
            
            ] 
            headers = ['Category'] + self.HC_sims[0]
            table = tabulate(data, headers=headers, tablefmt='grid')
            print(table)        
        else:
            data = [
            ['High Card'] + self.HC_sims[1],
            ['Pair'] + self.PR_sims[1],
            ['Straight'] + self.STR_sims[1],
            ['Flush'] + self.FLSH.sims[1],
            ['Straight Flush'] + self.STR_FLSH_sims[1],
            ['Royal Flush'] + self.RF_sims[1]
            
            ] 
            headers = ['Category'] + self.HC_sims[0]
            table = tabulate(data, headers=headers, tablefmt='grid')
            print(table)         
   
    def print_all_graphs(self, train, eval, comb, sim, sing_opt_acts, trim):
            print("creating xy")
            self.create_x_y()
            self.plot_rewards(train, eval)
            self.plot_moving_rewards(train, eval)
            self.plot_moving_mean(train, eval, trim)
            if comb:
                self.comb_plot_rewards(train, eval)
                self.comb_plot_moving_total(train, eval)
                self.comb_plot_moving_mean(train, eval)
            if sim:    
                self.plot_sims()
            # comb plot opt acts doesnt have to be in comb group abo
            self.comb_plot_opt_acts(sing_opt_acts)
            self.comb_plot_loss()

    def print_select_graphs(self, rewards, movrews,movmean, loss, combs, sim, opts):
            self.create_x_y()
            if rewards:
                self.plot_rewards(train, eval)
            if movrews:
                self.plot_moving_rewards(train, eval)
            if movmean:
                self.plot_moving_mean(train, eval, trim)
            if loss:
                self.comb_plot_loss()
            if combs:
                self.comb_plot_rewards(train, eval)
                self.comb_plot_moving_total(train, eval)
                self.comb_plot_moving_mean(train, eval)
                self.comb_plot_loss()
            if sim:    
                self.plot_sims()
            # comb plot opt acts doesnt have to be in comb group above
            if opts:
                self.comb_plot_opt_acts(sing_opt_acts)        

class metric_dicts():
    """
    A class for managing and updating various metrics related to training and evaluation of models for each generation.

    This class provides methods to initialize, add keys to, and update metrics dictionaries for both training
    and evaluation phases of selfplay and other procedures. It also stores Kl divergence results (Sims, short for similarity) 
    and percentages for optimal actions. 

    Attributes:
        gen_train_rewards (dict): A dictionary to store training rewards for each generation.
        gen_train_moving_total (dict): A dictionary to store the moving total of training rewards for each generation.
        gen_train_moving_mean_reward (dict): A dictionary to store the moving mean reward during training for each generation.
        gen_train_final_mean_reward (dict): A dictionary to store the final mean reward at the end of training for each generation.

        gen_train_value_losses (dict): A dictionary to store value losses during training for each generation.
        gen_train_policy_losses (dict): A dictionary to store policy losses during training for each generation.
        gen_train_entropy_losses (dict): A dictionary to store entropy losses during training for each generation.

        gen_eval_rewards (dict): A dictionary to store evaluation rewards for each generation.
        gen_eval_moving_total (dict): A dictionary to store the moving total of evaluation rewards for each generation.
        gen_eval_moving_mean_reward (dict): A dictionary to store the moving mean reward during evaluation for each generation.
        gen_eval_final_mean_reward (dict): A dictionary to store the final mean reward at the end of evaluation for each generation.

        n_pairs (int): The number of pairs or generations for which metrics are being tracked.
        sims (dict): A dictionary to store similarities in terms of KL divergence results.

    Methods:
        add_keys_to_metrics_dict(train_storage_dict, eval_storage_dict):
            Adds keys to the metrics dictionaries based on input dictionaries.

        add_keys_to_metrics(keys):
            Adds keys to the metrics dictionaries based on a list of keys.

        update_train_metrics_from_callback(gen, callback_train):
            Updates training metrics from a callback object.

        update_eval_metrics_from_ep_rewards(gen, mean_reward, episode_rewards, percentages_ag, percentages_op):
            Updates evaluation metrics from episode rewards and percentages.

        update_sims(gen, sim_results):
            Updates similarity results, in terms of KL divergence, for a specific generation pair.
    """  
    def __init__(self, n_gens):
        self.gen_train_rewards = {}
        self.gen_train_moving_total = {}
        self.gen_train_moving_mean_reward= {}
        self.gen_train_final_mean_reward= {}

        self.gen_train_value_losses= {}
        self.gen_train_policy_losses= {}
        self.gen_train_entropy_losses= {}

        self.gen_eval_rewards = {}
        self.gen_eval_moving_total = {}
        self.gen_eval_moving_mean_reward= {}
        self.gen_eval_final_mean_reward= {}

        self.n_pairs = int(n_gens/2)
        self.sims= {}

        self.percentages_ag = {}
        self.percentages_op = {}

    def add_keys_to_metrics_dict(self, train_storage_dict, eval_storage_dict):
        """
        Adds keys to the metrics dictionaries for training and evaluation based on input dictionaries.

        This method iterates through the keys in the provided training and evaluation storage dictionaries and
        adds corresponding key entries with initial values (None) to the metrics object for tracking training
        and evaluation metrics.

        Args:
            train_storage_dict (dict): A dictionary containing keys for training metrics.
            eval_storage_dict (dict): A dictionary containing keys for evaluation metrics.

        Note:
            If the input dictionaries do not have the 'keys()' method (AttributeError), it falls back to using
            'list(train_storage_dict.keys())' and 'list(eval_storage_dict.keys())' to iterate through keys.
        """
        try:
            for key in train_storage_dict.keys():
                self.gen_train_rewards[key] = [None]
                self.gen_train_moving_total[key] = [None]
                self.gen_train_moving_mean_reward[key]= [None]
                self.gen_train_final_mean_reward[key]= [None]
        except AttributeError:
            for key in list(train_storage_dict.keys()):
                self.gen_train_rewards[key] = [None]
                self.gen_train_moving_total[key] = [None]
                self.gen_train_moving_mean_reward[key]= [None]
                self.gen_train_final_mean_reward[key]= [None]
        try:
            for key in eval_storage_dict.keys():   
                self.gen_eval_rewards[key] = [None]
                self.gen_eval_moving_total[key] = [None]
                self.gen_eval_moving_mean_reward[key]= [None]
                self.gen_eval_final_mean_reward[key]= [None]
        except AttributeError:
            for key in eval_storage_dict.keys():   
                self.gen_eval_rewards[key] = [None]
                self.gen_eval_moving_total[key] = [None]
                self.gen_eval_moving_mean_reward[key]= [None]
                self.gen_eval_final_mean_reward[key]= [None]

    def add_keys_to_metrics(self, keys):
        """
        Adds a list of keys to the metrics dictionaries for tracking various metrics.

        This method takes a list of keys as input and adds corresponding key entries with initial values (None) to the
        metrics dictionaries for both training and evaluation. These keys are typically associated with metrics such as
        rewards, losses, and mean values for a generative model.

        Args:
            keys (list): A list of keys to be added to the metrics dictionaries.
        """
        for key in keys:
            self.gen_train_rewards[key] = [None]
            self.gen_train_moving_total[key] = [None]
            self.gen_train_moving_mean_reward[key]= [None]
            self.gen_train_final_mean_reward[key]= [None]

            self.gen_train_value_losses[key]= [None]
            self.gen_train_policy_losses[key]= [None]
            self.gen_train_entropy_losses[key]= [None]
  
            self.gen_eval_rewards[key] = [None]
            self.gen_eval_moving_total[key] = [None]
            self.gen_eval_moving_mean_reward[key]= [None]
            self.gen_eval_final_mean_reward[key]= [None]

    def update_train_metrics_from_callback(self, gen, callback_train):
        """
        Update training metrics for a specific generation based on a callback_train object.

        This method updates various training metrics and statistics for a specific generation
        of a reinforcement learning agent based on the information provided in the callback_train
        object.

        Args:
            gen (int): The generation for which the training metrics are being updated.
            callback_train (CallbackTrain): An object containing training-related statistics and
                information for the current generation.

        Notes:
            The `callback_train` object is expected to contain the following attributes:
                - `moving_total`: A list of cumulative rewards or values for each training episode.
                - `moving_mean_reward`: A list of moving average rewards over training episodes.
                - `rewards`: A list of rewards collected during training episodes.
                - `value_losses`: A list of value loss values during training episodes.
                - `entropy_losses`: A list of entropy loss values during training episodes.
                - `policy_losses`: A list of policy loss values during training episodes.

        After calling this method, the updated training metrics for the specified generation will
        be stored in the respective attributes of the current object.
        """
        temp_list = callback_train.moving_total
        temp_list.pop(0)
        callback_train.moving_total = temp_list
        
        self.gen_train_moving_mean_reward[gen] = callback_train.moving_mean_reward
        self.gen_train_rewards[gen] = callback_train.rewards
        self.gen_train_moving_total[gen] = callback_train.moving_total
        self.gen_train_final_mean_reward[gen] = self.gen_train_moving_mean_reward[gen][-1]
        self.gen_train_value_losses[gen] = callback_train.value_losses
        self.gen_train_entropy_losses[gen] = callback_train.entropy_losses
        self.gen_train_policy_losses[gen] = callback_train.policy_losses
          
    def update_eval_metrics_from_ep_rewards(self, gen, mean_reward,episode_rewards, percentages_ag, percentages_op):
        """
        Update evaluation metrics based on episode rewards and percentages.

        This method updates various evaluation metrics and statistics for a specific generation
        of a reinforcement learning agent based on episode rewards and other statistics.

        Args:
            gen (int): The generation for which the evaluation metrics are being updated.
            mean_reward (float): The mean reward achieved in the evaluation episodes.
            episode_rewards (list of float): A list of rewards obtained in individual evaluation
                episodes.
            percentages_ag (list of float): A list of percentages related to the optimal actions taken by the agent.
            percentages_op (list of float): A list of percentages related to the optimal actions taken by the opponent.


        After calling this method, the updated evaluation metrics for the specified generation
        will be stored in the respective attributes of the current object
        """
        self.gen_eval_moving_mean_reward[gen].append(mean_reward)
        self.gen_eval_rewards[gen] = episode_rewards
        for i in range(0, len(self.gen_eval_rewards[gen])):
            if i == 0:
                self.gen_eval_moving_total[gen][i]= self.gen_eval_rewards[gen][i]
                self.gen_eval_moving_mean_reward[gen][i] = self.gen_eval_rewards[gen][i]
                self.gen_eval_moving_total[gen][i]= self.gen_eval_rewards[gen][i]
            else:    
                self.gen_eval_moving_total[gen].append(self.gen_eval_moving_total[gen][i-1 ] + self.gen_eval_rewards[gen][i])
                self.gen_eval_moving_mean_reward[gen].append(np.mean(self.gen_eval_rewards[gen][0:i+1]))
            
        self.gen_eval_final_mean_reward[gen]= mean_reward
        self.percentages_ag[gen] = percentages_ag
        self.percentages_op[gen] = percentages_op
        
    def update_sims(self, gen, sim_results):
        """
        Update similarity, in terms of KL divergence, for a specific generation pair. gen is the oppponent and gen+1 is the agent.

        Args:
            gen (int): The generation for which KL divergence results are being updated.
            sim_results (object): The  KL divergence results associated with the specified
                generation pair.

        Notes:
            After calling this method, the simulation results for the specified generation will
            be stored in the 'sims' attribute of the current object, using a key that combines
            the agent and oppponents generation number for reference.
        """
        self.sims['sim_ep' + str(gen) + str(gen+1)]= sim_results

                