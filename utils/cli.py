import argparse

def cli(): 
    parser = argparse.ArgumentParser(description='Argument Parser for experiments with metadata')
    parser.add_argument('-r', '--root_dir', type=str, default = '../../data/CHASEDB1',
                        help="Root directory of data") 
    parser.add_argument('-d', '--datafile', type=str, default = '../../data/retina_data.csv',
                        help="Full path to csv file with dataset description") 
    parser.add_argument('-f', '--folder', type=str,
                       default='saved_models',
                       help='Subfolder in the experiment directory to save model checkpoints') 
    parser.add_argument('-s', '--save_name', type=str,
                       default='best_model',
                       help='The name of model checkpoints') 
    parser.add_argument('-b', '--batch_size', type=int,
                       default=4,
                       help='The batch size for training, validation and evaluation') 
    parser.add_argument('-j', '--random_seed', type=int,
                       default=42,
                       help='Random seed used for reducing non-deterministic behaviour')
    parser.add_argument('-l', '--lr', type=float,
                       default=0.0002,
                       help='Starting learning rate used for optimizer and cycle scheduler') 
    parser.add_argument('-u', '--up_lr', type=float,
                       default=0.0008,
                       help='Upper point of learning rate used for optimizer and cycle scheduler')
    parser.add_argument('-n', '--num_epochs', type=int,
                       default=100,
                       help='Number of epochs for training') 
    parser.add_argument('-c', '--cycles', type=int,
                       default=15,
                       help='Number of cycles (half-cycles of changing direction) in training')
    parser.add_argument('-m', '--milestone', type=int,
                       default=10,
                       help='Number of epochs from which chekpoints are saved') 
    parser.add_argument('-e', '--loss_log', type=str, default = 'losses',
                        help="File name for the losses log (.csv)")
    parser.add_argument('-i', '--lr_log', type=str, default = 'lr_rates',
                        help="File name for the learning rate log (.csv)")                                
    parser.add_argument('-g', '--savefig', type=str, default = 'rate',
                        help="File name for the saved plots (.png and .pdf)")
    parser.add_argument('-k', '--savemetrics', type=str, default = 'metrics',
                        help="File name for the saved metrics (.csv)")              

    args = parser.parse_args() 
    return args 