from dataclasses import dataclass, field
import os
from typing import List, Tuple, Optional, Union
import numpy as np

from misc import Timer

timer = Timer()
timer.time(print_time=False)

from torchvision import datasets, transforms
import torch

from general.model_tester import test_model, train_model, TrainingParameters
from general.models import SimpleClassifier
from general.xp_data import XpData

from general import image_data, analyse_data
from general.analyse_data import Parameters, save_errors
from mnist import visualization


timer.time(msg='finish imports')

class FashionMnistData(image_data.ImageData):

    def __init__(self, data_root_dir: str = ''):
        """
        Leave the data_root_dir empty to load the data (if needed) into the directory
        of this file.
        Otherwise, it loads into data_root_dir.
        For example, you can use data_root_dir = './data' to load it into the subdirectory
        data inside the directory of the file from which the script is running.
        """
        transform = transforms.Compose([
            transforms.ToTensor()  # Convert to tensor and normalize to [0, 1]
        ])

        # Load dataset
        if data_root_dir == '':
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_root_dir = os.path.join(script_dir, "data")

        train_dataset = datasets.FashionMNIST(root=data_root_dir, train=True, transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(root=data_root_dir, train=False, transform=transform, download=True)

        super().__init__(
            train_dataset=train_dataset,        # size = 60000
            test_dataset=test_dataset,          # size = 10000
            label_names=train_dataset.classes   # = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                                #    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        )

params = Parameters(
    seed=1,
    up_to_label=2,
    learning_rate=0.001,
    hidden=(),
    batch_size=64,
    epochs=2)

# <editor-fold desc=" ------------------------ Prepare data ------------------------">

database = FashionMnistData()
# database.plot_examples()

train_data, test_data = database.restricted_data(up_to_label=params.up_to_label)
train_size = len(train_data)
train_loader = database.train_loader(up_to_label=params.up_to_label, batch_size=params.batch_size)
test_loader = database.test_loader(up_to_label=params.up_to_label)

# Prepare model
torch.manual_seed(params.seed)
model = SimpleClassifier(dimensions=[28 * 28] + list(params.hidden) + [params.up_to_label])

# </editor-fold>

# <editor-fold desc=" ------------------------ Training ------------------------">

# train_model(model=model, data_loader=train_loader, train_size=train_size,
#             epochs=params.epochs, learning_rate=params.learning_rate)
#
# print('Train data:')
# train_error_rate = test_model(model, train_loader)
# print('Test data:')
# test_error_rate = test_model(model, test_loader)

# exit(0)

# </editor-fold>

# image_data.plot_first_layer_images(model = model, title='Fashion filters')

run_scatter_animation = False
if run_scatter_animation:
    """
    Create scatter animation of the outputs of all the linear layers going into dimension 2.
    The frames corresponds to each learning step
    """

    with torch.no_grad():
        for X_test, Y_test in test_loader:
            break  # because I hate python!!!

    #                                         n x 1 x 28 x 28               n
    collector = image_data.OutputProgressCollector(model=model, test_input=X_test, test_output=Y_test)

    train_model(model=model, data_loader=train_loader, train_size=train_size,
                epochs=params.epochs, learning_rate=params.learning_rate,
                train_step_callback=collector.progress_callback)

    print('Train data:')
    train_error_rate = test_model(model, train_loader)
    print('Test data:')
    test_error_rate = test_model(model, test_loader)

    hidden_str = '_'.join(str(d) for d in params.hidden)
    collector.generate_animation()  # save_animation_file=f'media/labels_{params.up_to_label}_hidden_{hidden_str}')


up_to_label = 2
running_data = XpData(
    file_path=f'fashion_data{up_to_label}.csv',
    data_cls=Parameters,
    values = dict(train_error_rate=float, test_error_rate=float))


# save_errors(
#     running_data=running_data, test_loader=test_loader,
#     model_generator = lambda: SimpleClassifier.generate_from_parameters(
#         seeds                   =[1,10,100,1000],
#         linear_layer_dimensions =[(28*28,)+hidden+(up_to_label,) for hidden in [()]]),
#     data_generator = lambda: database.train_loaders_from_parameters(
#         up_to_label             =[up_to_label],
#         batch_sizes             =[64]
#     ),
#     training_generator = lambda: TrainingParameters.from_parameters(
#         learning_rates          =[0.001],
#         epoch_list              =[5])
# )
#
# analyse_data.errors_per_epoch(
#     running_data,
#     # seed = 10,
#     learning_rate=0.001,
#     hidden=(),
#     batch_size=64
# )


# for X_test, Y_test in test_loader:
#     break  # because I hate python!!!
# Y_test = Y_test.detach().numpy()
# X_test = X_test.detach().numpy()
#
# arr = np.squeeze(X_test[Y_test == 0])
#
# visualization.pca_visualize(arr[:20], 10)
