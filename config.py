from utils import get_now

config = {
    'epochs': 10,
    'lr': 1e-2,
    'backbone': 'wide_resnet_50',
    'batch_size': 64
}

sweep_configuration = {
    'method': 'bayes',
    'name': f'HierarchyCNN_sweep_{get_now(True)}',
    'project': 'Classification_2022',
    'entity': 'ljh415',
    'metric': {
        'goal': 'minimize',
        'name' : 'validation_loss'
    },
    'parameters':{
        'batch_size': {'values':[16, 32, 64]},
        'epochs': {'values': [10, 15, 20]},
        'lr': {'max': 0.1, 'min':0.001},
        'fine_const': {'max': 3.0, 'min': 1.0},
        'coarse_const': {'max': 2.0, 'min': 1.0}
    }
}