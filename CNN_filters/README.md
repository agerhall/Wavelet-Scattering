# Filter coefficients in the CNN

This work is based on the convolutional neural network built by S. Agerhäll with Keras. The main document is a Jupyter notebook called "cnn_filters.ipynb". It requires matplotlib, seaborn and compCNN2, which is provided in the same directory.

## Parameters

- img_x, img_y: image size; shouldn't be changed
- num_classes: number of classes; shouldn't be changed
- batch_size: used for back-propagation
- epochs: number of gradient steps during learning phase

- nb_out_chan: number of output channels in the first convolution layer
- kernel_size: size of the convolution filter in the first layer

## Training the model

Once the parameters have been set, the model must be trained. This is done by running the cell in section "Training the model".

## Plot filter coefficients

It is possible to visualize the filter coefficients for both the first and the second layer. They are plotted as heatmaps (red for positive values, blue for negative values).
