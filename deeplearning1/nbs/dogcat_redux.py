from utils import *
from vgg16 import Vgg16

root_path = './data/redux/'
test_path = root_path + 'test/'
results_path = root_path +'results/'
train_path = root_path + 'train/'
# train_path = root_path + 'sample/'
is_sample = ''
valid_path = root_path + 'valid/'

vgg = Vgg16()
batch_size = 64
no_of_epochs = 3

batches = vgg.get_batches(train_path, batch_size=batch_size)
val_batches = vgg.get_batches(valid_path, batch_size=2*batch_size)
batches.nb_class = batches.num_class
batches.nb_sample = batches.samples
val_batches.nb_class = val_batches.num_class
val_batches.nb_sample = val_batches.samples

vgg.finetune(batches)

vgg.model.optimizer.lr = 0.01

latest_weights_filename = None
for epoch in range(no_of_epochs):
    print("Running epoch: %d" % epoch)
    vgg.model.fit_generator(batches, steps_per_epoch=batches.samples // batch_size, epochs=1, validation_data=val_batches, validation_steps=val_batches.samples // (2*batch_size))
    # vgg.fit(batches, val_batches, nb_epoch=1)
    latest_weights_filename = '%sft%d.h5' % (is_sample, epoch)
    print("Saving %s" % latest_weights_filename)
    vgg.model.save_weights(results_path+latest_weights_filename)
print("Completed %s fit operations" % no_of_epochs)

