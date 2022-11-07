from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class Tboard():
    def __init__(self, config):
        self.logdir = config["tensorboard"]["logdir"]
        self.name = config["name"] + "_" + str(datetime.utcnow())
        self.writer = SummaryWriter(os.path.join(self.logdir, self.name))

def make_image_plot(generated_data, pdata, Integral, maxlen):
    """
    utility function to make image like plots to compare Generated and Real Vectors.
    Some of the factors are hard coded for ease.
    """
    fig = plt.subplots(nrows = 5, ncols = 5, figsize = (15, 10), sharex = True, sharey = True)
    for (i,j), gen in np.ndenumerate(np.arange(0, 25).reshape(5, 5)):
        fig[1][i, j].plot(np.arange(0, maxlen) + 15, generated_data[i, j, :]*Integral, 'bo-', label = "Generated data (fake)")
        fig[1][i, j].plot(np.arange(0, maxlen) + 15, pdata[i, j, :]*Integral, 'ro-', label = "True data (Real)")
        #ax[i, j].set_xlabel("Age", fontsize = 12)
        #ax[i, j].set_ylabel("Norm Counts", fontsize = 12)
        #ax[i, j].legend(fontsize = 12)
    fig[1][0, 0].legend(fontsize = 15)

    return fig

def Write_FullInfo(config, writer):

    Summary = f'''
    The Experiment \' {config['name']}\' was performed on a {config['device']}.
    Here is a summary of the experiment
    TITLE : {config['plot_title']}
    SEED : {config['seed']}
    DATASET :
            ROOTFILE : {config['dataset']['root_file']}
            DOWNSAMPLE FRACTION : {1. - config['dataset']['frac']}
            TRAINING_SAMPLES : {config['dataset']['maxlength']}
    TEST DATASET :
            ROOTFILE : {config['test_dataset']['root_file']}
            DOWNSAMPLE FRACTION : {1. - config['test_dataset']['frac']}
            TRAINING_SAMPLES : {config['test_dataset']['maxlength']}
    TRAINING INFO :
            LATENT DIM : {config['training']['latent_dim']}
            INPUT DATA LENGTH : {config['training']['input_data_len']}
            NUM EPOCHS : {config['training']['num_epochs']}
    OUTPUT LOCATION :
            DIRECTORY : {config['output']['dir']}
    '''

    writer.add_text('Detailed Information', Summary)
    writer.flush()
