from vgg16 import Vgg16
import numpy as np

is_testing = False
root_path = '/data/redux/'
if is_testing:
    batch_size = 8
    root_path = root_path + 'sample/'
else:
    batch_size = 64

test_path = root_path + 'test/'
results_path = '/output/'
train_path = root_path + 'train/'
valid_path = root_path + 'valid/'

vgg = Vgg16()
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
vgg.model.fit_generator(batches, steps_per_epoch=batches.samples // batch_size, epochs=no_of_epochs, validation_data=val_batches, validation_steps=val_batches.samples // (2*batch_size))
latest_weights_filename = 'ft.h5'
print("Saving %s" % latest_weights_filename)
vgg.model.save_weights(results_path+latest_weights_filename)
print("Completed %s fit operations" % no_of_epochs)
batches, preds = vgg.test(test_path, batch_size = batch_size*2)
filenames = batches.filenames
isdog = preds[:,1]
isdog = isdog.clip(min=0.05, max=0.95)
filenames = batches.filenames
submission_file_name = 'submission.csv'
with open(results_path+'filenames.txt', 'w') as f:
    for i in filenames:
        f.write(i+"\n")
with open(results_path+'preds.txt', 'w') as f:
    for i in isdog.astype(str):
        f.write(i+"\n")
