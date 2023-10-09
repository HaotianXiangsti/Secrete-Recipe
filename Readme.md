The Secrete-Recipe.

Two thing needs pip:

`pip install tensorboardX`

`pip install torch_geometric`

Run the script like this:

PEMS04

`python main.py --config configurations/PEMS04_astgcn.conf` 


PEMS08

`python main.py --config configurations/PEMS08_astgcn.conf` 

SafeGraph

`python main.py --config configurations/NYC_new_astgcn.conf`

========================================================

Evaluation for PEMS04

`python Evaluation.py --config configurations/PEMS04_astgcn.conf` 

Evaluation for PEMS08

`python Evaluation.py --config configurations/PEMS08_astgcn.conf` 

Evaluation for SafeGraph

`python Evaluation.py --config configurations/NYC_new_astgcn.conf` 

Careful with config epochs_starting_reconstruct_loss, better set it into 10000 (a ver large number). 

Now calculating reconstructing loss may cause CUDA memeory exceed
