The Secrete-Recipe.

Two thing needs pip:

`pip install tensorboardX`

`pip install torch_geometric`

Run the script like this:

PEMS04

`python main.py --config configurations/PEMS04_astgcn.conf` 


PEMS08

`python main.py --config configurations/PEMS08_astgcn.conf` 


========================================================

Evaluation for PEMS04

`python Evaluation.py --config configurations/PEMS04_astgcn.conf` 

Evaluation for PEMS08

`python Evaluation.py --config configurations/PEMS08_astgcn.conf` 


========================================================

Classification for PEMS04

`python joint_train.py --config configurations/PEMS04_astgcn.conf` 

Classification for PEMS08

`python joint_train.py --config configurations/PEMS08_astgcn.conf` 


