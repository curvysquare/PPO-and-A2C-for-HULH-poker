import numpy as np
import matplotlib.pyplot as plt
import texas_holdem_mod as texas_holdem
from stable_baselines3.common.monitor import Monitor
from tabulate import tabulate
from scipy.interpolate import make_interp_spline
 
class graph_metrics():
    def __init__(self, n_models, storage, storageB,figsize, t_steps, overlay, e_steps): 
        self.num_graphs = n_models
        self.storage = storage
        self.storageB = storageB
        self.figsize = figsize
        self.n_SP_gens = n_models
        self.t_steps = t_steps
        self.overlay = overlay
        self.e_steps = e_steps
 
    def create_x_y(self):
        same_keys = self.check_dicts_have_same_keys(self.storage)
        self. dict_attributes = [attr for attr in dir(self.storage) if isinstance(getattr(self.storage, attr), dict)]
        self.storage_ids = list(getattr(self.storage, self.dict_attributes[1]).keys())
        
        self.train_moving_mean = {}
        self.train_moving_total = {}
        self.train_losses = {}
        self.train_rewards = {}
        
        self.train_rand_op_moving_mean = {}
        self.train_rand_op_moving_total = {}
        self.train_rand_op_losses = {}
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
        self.comb_train_losses = [[], []]
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
        
        if 1 in self.storage.sims:
            del self.storage.sims[1]
        if 2 in self.storage.sims:
            del self.storage.sims[2]    
            
        for key in self.storage.sims.keys():
            self.HC_sims[1].append(round(self.storage.sims[key]['HC'], 2)*100)
            self.STR_sims[1].append(round(self.storage.sims[key]['STR'], 2) *100)
            self.RF_sims[1].append(round(self.storage.sims[key]['RF'], 2) *100)
            
        self.HC_sims[0] = [i for i in range(len(self.HC_sims[1]))]
        self.STR_sims[0] = [i for i in range(len(self.STR_sims[1]))]
        self.RF_sims[0] = [i for i in range(len(self.RF_sims[1]))]
            

                 
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
            self.train_losses[key] = ([i for i in range(1, len(self.storage.gen_train_losses[key])+1)], self.storage.gen_train_losses[key]) 
            self.train_rewards[key] = ([i for i in range(1, len(self.storage.gen_train_rewards[key])+1)], self.storage.gen_train_rewards[key])
            
            if self.storageB is not None:
                self.train_rand_op_moving_mean[key]= ([i for i in range(1, len(self.storageB.gen_train_moving_mean_reward[key])+1)], self.storageB.gen_train_moving_mean_reward[key])
                self.train_rand_op_moving_total[key]= ([i for i in range(1, len(self.storageB.gen_train_moving_total[key])+1)], self.storageB.gen_train_moving_total[key])
                self.train_rand_op_losses[key]= ([i for i in range(1, len(self.storageB.gen_train_losses[key])+1)], self.storageB.gen_train_losses[key])
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
            self.comb_train_losses[1].extend(self.train_losses[key][1])
            
            if self.storageB is not None:
                self.comb_train_rand_op_rewards[1].extend(self.train_rand_op_rewards[key][1])
                self.comb_eval_rand_op_rewards[1].extend(self.eval_rand_op_rewards[key][1])
                self.comb_train_rand_op_losses[1].extend(self.train_rand_op_losses[key][1])
          
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
            
        #     self.comb_train_moving_mean[1].append(self.train_moving_mean[key][1])
        #     self.comb_train_moving_total[1].extend(self.train_moving_total[key][1])
        #     self.comb_train_losses[1].append(self.train_losses[key][1])
        #     self.comb_train_rewards[1].append(self.train_rewards[key][1])
        #     self.comb_eval_moving_mean[1].append(self.eval_moving_mean[key][1])
        #     self.comb_eval_moving_total[1].append(self.eval_moving_total[key][1])
        #     self.comb_eval_rewards[1].append(self.eval_rewards[key][1])
            
        self.comb_train_moving_mean[0] = [i for i in range(1, len(self.comb_train_moving_mean[1])+1)]    
        self.comb_train_moving_total[0]= [i for i in range(1, len(self.comb_train_moving_total[1])+1)]
        self.comb_train_losses[0] = [i for i in range(1, len(self.comb_train_losses[1])+1)]
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
        
        
         
    def plot_rewards(self, train, eval):
        if not self.overlay:
            fig4, axs4 = plt.subplots(self.num_graphs, 2, sharey= 'col', figsize = self.figsize)
            for i, key in enumerate(self.storage_ids):
                if train:
                    axs4[i, 0].plot(self.train_rewards[key][0], self.train_rewards[key][1], color = 'b', label= 'reward')
                    axs4[i, 0].axhline(y=np.mean(self.train_rewards[key][1]), color='r', linestyle='-', label='mean reward')
                    axs4[i, 0].axhline(y=-0.5, color='g', linestyle='--', label='random')
                    # axs4[i, 0].plot(self.train_rand_op_rewards[key][0], self.train_rand_op_rewards[key][1], color = 'g', linestyle = '--',label= 'rand')
                    axs4[i, 0].set_title(str(key) + ' training Reward')
                    axs4[i, 0].legend(fontsize='small')
                if eval:
                    axs4[i, 1].plot(self.eval_rewards[key][0], self.eval_rewards[key][1], color = 'b' , label= 'reward')
                    axs4[i, 1].axhline(y=np.mean(self.eval_rewards[key][1]), color='r', linestyle='-', label='mean reward')
                    # axs4[i, 1].plot(self.eval_rewards[key][0], self.train_rand_op_rewards[key][1][:len(self.eval_rewards[key][0])], color = 'g', linestyle = '--',label= 'rand')
                    axs4[i, 1].axhline(y=-0.5, color='g', linestyle='--', label='random')
                    axs4[i, 1].set_title(str(key) + ' evaluation Reward')
                    axs4[i, 1].legend(fontsize='small')
            plt.tight_layout()
            plt.show()  
        else:
            fig4, axs4 = plt.subplots(1, 1, sharey= 'col', figsize = self.figsize)
            for key in (self.storage_ids):
                if train:
                    axs4.plot(self.train_rewards[key][0], self.train_rewards[key][1], color = 'b', label= 'reward')
                    axs4.axhline(y=np.mean(self.train_rewards[key][1]), color='r', linestyle='-', label='mean reward')
        
                    # axs4[i, 0].plot(self.train_rand_op_rewards[key][0], self.train_rand_op_rewards[key][1], color = 'g', linestyle = '--',label= 'rand')
                    axs4.set_title(' training Reward')

                if eval:
                    if key == 'PPO': 
                        colourP = 'b'
                    else:
                        colourP = 'g'
                    axs4.plot(self.eval_rewards[key][0], self.eval_rewards[key][1], color = colourP , label= str(key) + '_reward')
                    # axs4.axhline(y=np.mean(self.eval_rewards[key][1]), color='r', linestyle='-', label='mean reward')
                    axs4.set_title('evaluation Reward')
    
            axs4.legend(fontsize='small')
            plt.tight_layout()
            plt.show()  
            
                        

   
             
    def plot_moving_rewards(self, train, eval):
        if not self.overlay:
            fig2, axs2 = plt.subplots(self.num_graphs, 2, sharey= 'col',figsize = self.figsize) 
        
            for i, key in enumerate(self.storage_ids):
                if train:
                    axs2[i, 0].plot(self.train_moving_total[key][0], self.train_moving_total[key][1],  color = 'b', label= 'moving total')
                    axs2[i, 0].axhline(y=np.mean(self.train_moving_total[key][1]), color='r', linestyle='-', label='mean')
                    if self.train_rand_op_moving_total:
                        axs2[i, 0].plot(self.train_rand_op_moving_total[key][0], self.train_rand_op_moving_total[key][1],  color = 'g', label= 'rand')
                    # axs2[i, 0].axhline(y=-0.5, color='g', linestyle='--', label='random')
                    axs2[i, 0].set_title('train moving Reward')
                    axs2[i, 0].legend(fontsize='small')
                if eval:
                    axs2[i, 1].plot(self.eval_moving_total[key][0], self.eval_moving_total[key][1], color = 'b', label= 'moving total')
                    # axs2[i, 1].axhline(y=np.mean(self.eval_moving_total[key][1]), color='r', linestyle='-', label='mean')
                    if self.eval_rand_op_moving_total:
                        axs2[i, 1].plot(self.eval_rand_op_moving_total[key][0], self.eval_rand_op_moving_total[key][1],  color = 'g', label= 'random')
                    # axs2[i, 1].axhline(y=-0.5, color='g', linestyle='--', label='random')
                    axs2[i, 1].set_title('evaluation moving Reward')
                    axs2[i, 1].legend(fontsize='small')
            plt.tight_layout()
            plt.show()
            
        else:
            fig4, axs4 = plt.subplots(1, 1, sharey= 'col', figsize = self.figsize)
            for key in (self.storage_ids):
                if train:
                    axs4.plot(self.train_moving_total[key][0], self.train_moving_total[key][1], color = 'b', label= 'reward')
                    axs4.axhline(y=np.mean(self.train_moving_total[key][1]), color='r', linestyle='-', label='mean reward')
        
                    # axs4[i, 0].plot(self.train_rand_op_rewards[key][0], self.train_rand_op_rewards[key][1], color = 'g', linestyle = '--',label= 'rand')
                    axs4.set_title(' training Reward')

                if eval:
                    if key == 'PPO': 
                        colourP = 'b'
                    else:
                        colourP = 'g'
                    axs4.plot(self.eval_moving_total[key][0], self.eval_moving_total[key][1], color = colourP , label= str(key) + '_moving reward')
                    axs4.set_title('evaluation moving Reward')
                    
            # if self.eval_rand_op_moving_total:
            #     axs4.plot(self.eval_rand_op_moving_total['PPO'][0], self.eval_rand_op_moving_total['PPO'][1],  color = 'r', label= 'random')
            axs4.legend(fontsize='small')
            plt.tight_layout()
            plt.show()                  
    
    def plot_moving_mean(self, train, eval):
        if not self.overlay:
            fig1, axs1 = plt.subplots(self.num_graphs, 2, sharey= 'col', figsize = self.figsize)
            for i, key in enumerate(self.storage_ids):
                if train:
                    axs1[i, 0].plot(self.train_moving_mean[key][0], self.train_moving_mean[key][1],  color = 'b', label= 'moving mean')
                    axs1[i, 0].axhline(y=np.mean(self.train_moving_mean[key][1]), color='r', linestyle='-', label='average')
                    # axs1[i, 0].plot(self.train_rand_op_moving_mean[key][0], self.train_rand_op_moving_mean[key][1],  color = 'g', label= 'rand')
                    axs1[i, 0].axhline(y=-0.5, color='g', linestyle='--', label='random')
                    axs1[i, 0].set_title(str(key) + ' moving mean')
                    axs1[i, 0].legend(fontsize='small')
                if eval:
                    axs1[i, 1].plot(self.eval_moving_mean[key][0], self.eval_moving_mean[key][1], color = 'b', label= 'moving mean')
                    # axs1[i, 1].plot(self.eval_rand_op_moving_mean[key][0], self.eval_rand_op_moving_mean[key][1], color = 'g', label= 'rand')
                    axs1[i, 1].axhline(y=np.mean(self.eval_moving_mean[key][1]), color='r', linestyle='-', label='average')
                    axs1[i, 1].axhline(y=-0.5, color='g', linestyle='--', label='random')
                    axs1[i, 1].set_title(str(key) + ' moving mean')
                    axs1[i, 1].legend(fontsize='small')   
            plt.tight_layout()
            plt.show()
            
        else:
            fig4, axs4 = plt.subplots(1, 1, sharey= 'col', figsize = self.figsize)
            for key in (self.storage_ids):
                if train:
                    axs4.plot(self.train_moving_mean[key][0], self.train_moving_mean[key][1], color = 'b', label= 'reward')
                    axs4.axhline(y=np.mean(self.train_moving_mean[key][1]), color='r', linestyle='-', label='mean reward')
        
                    # axs4[i, 0].plot(self.train_rand_op_rewards[key][0], self.train_rand_op_rewards[key][1], color = 'g', linestyle = '--',label= 'rand')
                    axs4.set_title(' training Reward')

                if eval:
                    if key == 'PPO': 
                        colourP = 'b'
                    else:
                        colourP = 'g'
                    if key == 'random':
                        pass
                    else:     
                        axs4.plot(self.eval_moving_mean[key][0], self.eval_moving_mean[key][1], color = colourP , label= str(key) + 'mean reward')
                        # axs4.axhline(y=np.mean(self.eval_rewards[key][1]), color='r', linestyle='-', label='mean reward')
                        axs4.set_title('evaluation mean reward')
                    
            axs4.axhline(y=-0.5, color='r', linestyle='--', label='random')
            axs4.legend(fontsize='small')
            plt.tight_layout()
            plt.show()                     
                
    
    def plot_loss(self):
        fig5, axs5 = plt.subplots(self.num_graphs, 1, sharey='col', figsize = self.figsize)
        for i, key in enumerate(self.storage_ids):
            axs5[i].plot(self.train_losses[key][0], self.train_losses[key][1], label='training loss')
            axs5[i].set_title(str(key) + 'loss')
            
        plt.tight_layout()  
        plt.show()
    
    def comb_plot_rewards(self, train, eval):
        fig2, axs2 = plt.subplots(2, 1,figsize = self.figsize) 
        if train:
            axs2[0].axhline(y=np.mean(self.comb_train_rewards[1]), color='r', linestyle='-', label='mean')
            axs2[0].plot(self.comb_train_rand_op_rewards[0], self.comb_train_rand_op_rewards[1],  color = 'g', label= 'rand')
            axs2[0].plot(self.comb_train_rewards[0], self.comb_train_rewards[1],  color = 'b', label= 'rewards')
            # axs2[0, 0].axhline(y=-0.5, color='g', linestyle='--', label='random')
            axs2[0].set_title(' train Rewards')
            axs2[0].legend(fontsize='small')
        if eval:
            axs2[1].plot(self.comb_eval_rand_op_rewards[0], self.comb_eval_rand_op_rewards[1],  color = 'g', label= 'rand')
            axs2[1].plot(self.comb_eval_rewards[0], self.comb_eval_rewards[1], color = 'b', label= 'rewards')
            axs2[1].axhline(y=np.mean(self.comb_eval_rewards[1]), color='r', linestyle='-', label='mean')
            # axs2[1].plot(self.comb_eval_rand_op_rewards[0], self.comb_eval_rand_op_rewards[1],  color = 'g', label= 'rand')
            # axs2[0, 1].axhline(y=-0.5, color='g', linestyle='--', label='random')
            axs2[1].set_title('eval rewards')
            axs2[1].legend(fontsize='small')
        # plt.tight_layout()
        plt.show()
        
    def comb_plot_moving_total(self, train, eval):
        fig2, axs2 = plt.subplots(2, 1,figsize = self.figsize) 
        if train:
            axs2[0].plot(self.comb_train_moving_total[0], self.comb_train_moving_total[1],  color = 'b', label= 'moving total')
            # axs2[0].axhline(y=np.mean(self.comb_train_moving_total[1]), color='r', linestyle='-', label='mean')
            axs2[0].plot(self.comb_train_rand_op_moving_total[0], self.comb_train_rand_op_moving_total[1],  color = 'g', label= 'rand')
            for j in range(1,self.n_SP_gens+1):
                axs2[0].axvline(x=self.t_steps *j, color='y', linestyle='--')
            axs2[0].set_title(' train moving Reward')
            axs2[0].legend(fontsize='small')
        if eval:
            axs2[1].plot(self.comb_eval_moving_total[0], self.comb_eval_moving_total[1], color = 'b', label= 'moving total')
            # axs2[1].axhline(y=np.mean(self.comb_eval_moving_total[1]), color='r', linestyle='-', label='mean')
            axs2[1].plot(self.comb_eval_rand_op_moving_total[0], self.comb_eval_rand_op_moving_total[1],  color = 'g', label= 'rand')
            # axs2[0, 1].axhline(y=-0.5, color='g', linestyle='--', label='random')
            for j in range(1,self.n_SP_gens+1):
                axs2[1].axvline(x=self.e_steps*j, color='y', linestyle='--')
            axs2[1].set_title('eval moving Reward')
            axs2[1].legend(fontsize='small')
        plt.tight_layout()
        plt.show()
        
    def comb_plot_moving_mean(self, train, eval):
        fig2, axs2 = plt.subplots(2, 1,figsize = self.figsize) 
        if train:
            axs2[0].plot(self.comb_train_moving_mean[0], self.comb_train_moving_mean[1],  color = 'b', label= 'moving mean')
            # axs2[0].axhline(y=np.mean(self.comb_train_moving_mean[1]), color='r', linestyle='-', label='mean')
            # axs2[i, 0].plot(self.train_rand_op_moving_total[key][0], self.train_rand_op_moving_total[key][1],  color = 'g', label= 'rand')
            axs2[0].axhline(y=-0.5, color='g', linestyle='--', label='random')
            for j in range(1,self.n_SP_gens+1):
                axs2[0].axvline(x=self.t_steps *j, color='y', linestyle='--')
            axs2[0].set_title(' train moving mean')
            axs2[0].legend(fontsize='small')
        if eval:
            axs2[1].plot(self.comb_eval_moving_mean[0], self.comb_eval_moving_mean[1], color = 'b', label= 'moving mean')
            # axs2[1].axhline(y=np.mean(self.comb_eval_moving_mean[1]), color='r', linestyle='-', label='mean')
            # axs2[i, 1].plot(self.eval_rand_op_moving_total[key][0], self.eval_rand_op_moving_total[key][1],  color = 'g', label= 'rand')
            axs2[1].axhline(y=-0.5, color='g', linestyle='--', label='random')
            for j in range(1,self.n_SP_gens+1):
                axs2[1].axvline(x=self.e_steps *j, color='y', linestyle='--')
            axs2[1].set_title('eval moving mean')
            axs2[1].legend(fontsize='small')
        plt.tight_layout()
        plt.show()    

    def comb_plot_opt_acts(self, sing_opt_acts):
        if not self.overlay:
            if sing_opt_acts:           
                fig5, axs5 = plt.subplots(1, 1, sharey='col', figsize = self.figsize)
                axs5.plot(self.sep_percentages_ag[0], self.sep_percentages_ag[1], label='agent optimal action rate')
                axs5.set_title('optimal action percentages')
                axs5.legend(fontsize='small')
                plt.tight_layout()  
                plt.show()  
            else:
                fig5, axs5 = plt.subplots(1, 1, sharey='col', figsize = self.figsize)
                axs5.plot(self.comb_percentages_ag[0], self.comb_percentages_ag[1], label='optimal action rate')
                axs5.set_title('optimal action percentages across generations')
                axs5.legend(fontsize='small')
                # for j in range(1,self.n_SP_gens+1):
                #     axs5.axvline(x=self.e_steps *j, color='y', linestyle='--', label = 'generation update')
                plt.tight_layout()  
                plt.show()
         
        else:
            fig5, axs5 = plt.subplots(1, 1, sharey='col', figsize = self.figsize)
            axs5.plot(self.sep_percentages_ag[0], self.sep_percentages_ag[1], label='agent optimal action rate')
            axs5.plot(self.sep_percentages_op[0], self.sep_percentages_op[1], label='opponent optimal action rate')
            axs5.set_title('optimal action percentages')
            axs5.legend(fontsize='small')
            plt.tight_layout()  
            plt.show()    
  
            
    def plot_sims(self):
        if len(self.HC_sims[0])>1:
            fig5, axs5 = plt.subplots(1, 1, figsize = self.figsize)
            axs5.plot(self.HC_sims[0], self.HC_sims[1], label='High Card')
            axs5.plot(self.STR_sims[0], self.STR_sims[1], label='Straight')
            axs5.plot(self.RF_sims[0], self.RF_sims[1], label='Royal Flush')
            # for j in range(1,self.n_SP_gens+1):
            #     axs5.axvline(x=self.t_steps *j, color='y', linestyle='--')
            axs5.set_title('action distribution cosine similarity')
            axs5.legend(fontsize='small')
            plt.tight_layout()  
            plt.show()
            
            data = [['High Card'] + self.HC_sims[1],['Straight'] + self.STR_sims[1],['Royal Flush'] + self.RF_sims[1]]
            headers = ['Category'] + self.HC_sims[0]
            table = tabulate(data, headers=headers, tablefmt='grid')
            print(table)        
        else:
            data = [['High Card'] + self.HC_sims[1],['Straight'] + self.STR_sims[1],['Royal Flush'] + self.RF_sims[1]]
            headers = ['Category'] + self.HC_sims[0]
            table = tabulate(data, headers=headers, tablefmt='grid')
            print(table)         


    #non plots below     
    def check_dicts_have_same_keys(self, class_instance):
    # Get the list of attributes that are dictionaries
        dict_attributes = [attr for attr in dir(class_instance) if isinstance(getattr(class_instance, attr), dict)]
        
        if len(dict_attributes) == 0:
            print("No dictionary attributes found in the class.")
            return False
        
        # Get the keys of the first dictionary
        reference_keys = list(getattr(class_instance, dict_attributes[1]).keys())
        
        for attr in dict_attributes[1:]:
            current_keys = list(getattr(class_instance, attr).keys())
            if current_keys != reference_keys:
                print(f"Keys in '{attr}' are different from the reference dictionary.")
                print(reference_keys, current_keys)
                return False
        
        # print("All dictionary attributes have the same keys.")
        return True      
    
    def print_all_graphs(self, train, eval, comb, sim, sing_opt_acts):
            print("creating xy")
            self.create_x_y()
            self.plot_rewards(train, eval)
            self.plot_moving_rewards(train, eval)
            self.plot_moving_mean(train, eval)
            self.plot_loss()
            if comb:
                self.comb_plot_rewards(train, eval)
                self.comb_plot_moving_total(train, eval)
                self.comb_plot_moving_mean(train, eval)
            if sim:    
                self.plot_sims()
            # comb plot opt acts doesnt have to be in comb group abo
            self.comb_plot_opt_acts(sing_opt_acts)
            
class obs_type_envs():
    def __init__(self):
        train_env_124 = texas_holdem.env(obs_type = '124', render_mode = 'rgb_array')
        train_env_72 = texas_holdem.env(obs_type = '72', render_mode = 'rgb_array')
        train_env_72_plus = texas_holdem.env(obs_type = '72+', render_mode = 'rgb_array')
        
        eval_env_124 = Monitor(texas_holdem.env(obs_type = '124', render_mode = 'rgb_array'))
        eval_env_72 = Monitor(texas_holdem.env(obs_type = '72', render_mode = 'rgb_array'))
        eval_env_72_plus = Monitor(texas_holdem.env(obs_type = '72+', render_mode = 'rgb_array'))

        self.train_envs = {'124': train_env_124, '72':train_env_72, '72+': train_env_72_plus}
        self.eval_envs = {'124': eval_env_124, '72':eval_env_72, '72+': eval_env_72_plus}

class metric_dicts():
    def __init__(self, n_gens):

        
        self.gen_train_rewards = {}
        self.gen_train_moving_total = {}
        self.gen_train_moving_mean_reward= {}
        self.gen_train_final_mean_reward= {}
        self.gen_train_losses= {}
        
        # self.gen_train_rand_op_rewards = {}
        # self.gen_train_rand_op_moving_total = {}
        # self.gen_train_rand_op_moving_mean_reward = {}
        
        self.gen_eval_rewards = {}
        self.gen_eval_moving_total = {}
        self.gen_eval_moving_mean_reward= {}
        self.gen_eval_final_mean_reward= {}
        
        # self.gen_eval_rand_op_rewards = {}
        # self.gen_eval_rand_op_moving_total = {}
        # self.gen_eval_rand_op_moving_mean_reward = {}
        
        self.n_pairs = int(n_gens/2 - 1)
        self.sims= {}
        for i in range(1, self.n_pairs+1):
            self.sims[i] = None
              
        self.percentages_ag = {}
        self.percentages_op = {}
    def add_keys_to_metrics_dict(self, train_storage_dict, eval_storage_dict):
        for key in train_storage_dict.keys():
            self.gen_train_rewards[key] = [None]
            self.gen_train_moving_total[key] = [None]
            self.gen_train_moving_mean_reward[key]= [None]
            self.gen_train_final_mean_reward[key]= [None]
            self.gen_train_losses[key]= [None]
            
            # self.gen_train_rand_op_rewards[key] = [None]
            # self.gen_train_rand_op_moving_total[key] = [None]
            # self.gen_train_rand_op_moving_mean_reward[key] = [None]
            
        for key in eval_storage_dict.keys():   
            self.gen_eval_rewards[key] = [None]
            self.gen_eval_moving_total[key] = [None]
            self.gen_eval_moving_mean_reward[key]= [None]
            self.gen_eval_final_mean_reward[key]= [None]
            
            # self.gen_eval_rand_op_rewards = [None]
            # self.gen_eval_rand_op_moving_total = [None]
            # self.gen_eval_rand_op_moving_mean_reward = [None]
            
    def add_keys_to_metrics(self, keys):
        for key in keys:
            self.gen_train_rewards[key] = [None]
            self.gen_train_moving_total[key] = [None]
            self.gen_train_moving_mean_reward[key]= [None]
            self.gen_train_final_mean_reward[key]= [None]
            self.gen_train_losses[key]= [None]
            
            # self.gen_train_rand_op_rewards[key] = [None]
            # self.gen_train_rand_op_moving_total[key] = [None]
            # self.gen_train_rand_op_moving_mean_reward[key] = [None]
  
            self.gen_eval_rewards[key] = [None]
            self.gen_eval_moving_total[key] = [None]
            self.gen_eval_moving_mean_reward[key]= [None]
            self.gen_eval_final_mean_reward[key]= [None]
            
            # self.gen_eval_rand_op_rewards = [None]
            # self.gen_eval_rand_op_moving_total = [None]
            # self.gen_eval_rand_op_moving_mean_reward = [None]        
     
    def update_train_metrics_from_callback(self, gen, callback_train):
        temp_list = callback_train.moving_total
        temp_list.pop(0)
        callback_train.moving_total = temp_list
        
        self.gen_train_moving_mean_reward[gen] = callback_train.moving_mean_reward
        self.gen_train_rewards[gen] = callback_train.rewards
        self.gen_train_moving_total[gen] = callback_train.moving_total
        self.gen_train_final_mean_reward[gen] = self.gen_train_moving_mean_reward[gen][-1]
        self.gen_train_losses[gen] = callback_train.losses
        
        # self.gen_train_rand_op_rewards[gen] = callback_train.op_rewards
        # self.gen_train_rand_op_moving_total[gen] = callback_train.op_moving_total
        # self.gen_train_rand_op_moving_mean_reward[gen] = callback_train.op_moving_mean_reward
        
         
    def update_eval_metrics_from_ep_rewards(self, gen, mean_reward,episode_rewards, percentages_ag, percentages_op):
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
        self.sims['sim_ep' + str(gen) + str(gen-1)]= sim_results

                