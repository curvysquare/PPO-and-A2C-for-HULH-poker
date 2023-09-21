class train_convergence_search():
    """
    Class to perform a search for convergence-related hyperparameters in a PPO training setup.

    This class initializes a search for convergence-related hyperparameters using Optuna, specifically
    for training a PPO agent in a Texas Hold'em environment. It defines a custom Optuna callback
    to stop training based on convergence criteria. It also tracks trial results and provides
    optimization parameters for the Optuna study.

    Args:
    - verbose (bool): Whether to display progress bars during training.
    - obs_type (str): Type of observation space in the environment.

    Attributes:
    - verbose (bool): Whether to display progress bars during training.
    - obs_type (str): Type of observation space in the environment.
    - model (str): Type of RL model being trained (default: 'PPO').
    - trial_results (dict): Dictionary to store trial results.

    Methods:
    - init_trained_op(): Initialize the trained opponent models.
    - callback(trial, study): Custom Optuna callback for tracking trial results.
    - optimize_cb_params(trial): Define optimization parameters for the convergence search.
    - optimize_cb(trial): Perform the convergence search using Optuna.
    """

    def __init__(self, verbose, obs_type, trial_training_steps, cb_frequency):
        self.verbose = verbose
        self.obs_type = obs_type
        self.model = 'PPO'
        self.trial_results = {}
        self.trial_training_steps = trial_training_steps
        self.cb_frequency = cb_frequency
    def init_trained_op(self, n_training_steps):
        """
        Initialize the trained opponent models using selfplay for the range of network architectures.
        """
        
        self.na_gen_0_dict = {}
        self.na = [{'pi': [64], 'vf': [64]}]
        for na_key in self.na:
            sp = self_play(0, n_training_steps, 1, obs_type = self.obs_type, tag = 19, model = self.model, na_key = na_key)
            sp.run(False)
            self.na_gen_0_dict[str(na_key)] = sp.gen_lib[0]

    def callback(self,trial, study):
        trail = trial
        study = study
        self.trial_results[study.number] = {
        'params': study.params,
        'value': study.value,
    }
        
    def optimize_cb_params(self, trial):
        return {
        'max_no_improvement_evals': trial.suggest_categorical('max_no_improvement_evals', [3000]),    
        'min_evals': trial.suggest_categorical('min_evals',  [2, 3])
        } 
    
    def optimize_cb(self, trial):
        """
        Perform the convergence search using Optuna.
        """
        cb_params = self.optimize_cb_params(trial)
        
        env = texas_holdem.env(self.obs_type, render_mode = "rgb_array")
        env.OPPONENT.policy = 'PPO'
        env.AGENT.policy = 'PPO'
        env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
        env.OPPONENT.model.set_parameters(self.na_gen_0_dict[str(self.na[0])])
        
        self.env = env

        Eval_env = texas_holdem.env(self.obs_type, render_mode = "rgb_array")
        Eval_env = Monitor(Eval_env)
        Eval_env.OPPONENT.policy = 'random'
        Eval_env.AGENT.policy = 'PPO'

        cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals= cb_params['max_no_improvement_evals'], min_evals=cb_params['min_evals'], verbose =1)
        cb = EvalCallback(Eval_env, eval_freq=self.cb_frequency, callback_after_eval=cb, verbose=0, n_eval_episodes= cb_params['max_no_improvement_evals']) 
           
        model = PPO('MultiInputPolicy', self.env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
        model.learn(total_timesteps=40000, dumb_mode = False, progress_bar=self.verbose, callback= cb)
    
        return cb.callback.parent.best_mean_reward
            
    def run(self, print_graphs):
        if __name__ == '__main__':
            study = optuna.create_study(direction= 'maximize')
        try:
            study.optimize(self.optimize_cb, n_trials=8, n_jobs=1, show_progress_bar= True, callbacks = [self.callback])
        except KeyboardInterrupt:
            print('Interrupted by keyboard.')
        print(study.best_params)
        if print_graphs == True:
            optuna.visualization.plot_param_importances(study).show()
            optuna.visualization.plot_optimization_history(study).show() 
    
    def plot_dictionary(self, dict ):
        dictionary = dict
        mean_dict = {}
    
        # Calculate the mean values for duplicate keys
        for key, value in dictionary.items():
            if key in mean_dict:
                mean_dict[key].append(value)
            else:
                mean_dict[key] = [value]
        
        for key in mean_dict:
            mean_dict[key] = sum(mean_dict[key]) / len(mean_dict[key])
        
        # Extract keys and values for plotting
        x_values = list(mean_dict.keys())
        y_values = list(mean_dict.values())
        
        # Create a bar graph
        plt.bar(x_values, y_values)
        plt.xlabel('No improvement steps')
        plt.ylabel('Mean Values')
        plt.title('Number of steps vs mean reward')

        plt.show()