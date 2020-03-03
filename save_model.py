from algorithms import get_b0_backbone, connect_simple_head


model = connect_simple_head(*get_b0_backbone())
model.load_weights('results/train_full.h5')

model.save("results/model.h5")
