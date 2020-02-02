from algorithms import get_b0_backbone, connect_simple_head
from generators import get_data_generators
from training import train_model

def train_b0_simple():
    train_generator, valid_generator = get_data_generators('split1', 64)
    train_model(train_generator, valid_generator, get_b0_backbone, connect_simple_head, 'results/b0_simple')
