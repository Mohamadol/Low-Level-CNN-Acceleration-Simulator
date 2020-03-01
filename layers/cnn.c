#include "cnn.h"



/*
	Initializes a convolution layer
	Expects input size as [in_w, in_h, in_c]
	Expects kernel size as [k_w, k_h, k_input_channel, k_output_channel]
	returns pointer to layer 
	To be Added:
	1- Padding
	2- Stride
*/
cnn * cnn_init(int *input_size, int *kernel_size, double *input, double *kernel, int stride, int padding) {

	cnn *layer = (cnn *)malloc(sizeof(cnn));
	if (layer == NULL) {
		printf("Problem allocating dynamic memory in cnn\n");
		return 0;
	}

	layer->stride = stride;
	layer->padding = padding;

	layer->inputs = input;
	layer->input_c = *(input_size+2);
	layer->input_h = *(input_size+1) + 2 * padding;
	layer->input_w = *(input_size) + 2 * padding;
	cnn_pad(layer);
	
	layer->kernel = kernel;
	layer->kernel_co = *(kernel_size + 3);
	layer->kernel_ci = *(kernel_size + 2);
	layer->kernel_h =  *(kernel_size + 1);
	layer->kernel_w =  *(kernel_size);

	
	layer->output_c = *(kernel_size + 3);
	layer->output_h = ( (layer->input_h - layer->kernel_h ) / stride) + 1;
	layer->output_w = ( (layer->input_w - layer->kernel_w ) / stride) + 1;
	layer->output = (double *)malloc((layer->output_w * layer->output_h * layer->output_c) * sizeof(double));
	if (layer->output == NULL) {
		printf("couldn't allocate memory in CNN\n");
		return 0;
	}

}

void cnn_pad(cnn * layer) {

	int new_input_offset, old_input_offset;
	int channel, width, height;
	double * padded_input = (double *)malloc((layer->input_w * layer->input_h * layer->input_c) * sizeof(double));

	if (layer->padding == 0)
		return;

	for (channel = 0; channel < layer->input_c; channel++) {
		for (height = 0; height < layer->input_h; height++) {
			for (width = 0; width < layer->input_w; width++) {

				new_input_offset = channel * layer->input_h * layer->input_w   +   height * layer->input_w + width;

				if (height == 0 || width == 0 || height == layer->input_h - 1 || width == layer->input_w - 1) {
					*(padded_input + new_input_offset) = 0;
				}
				else {
					old_input_offset = channel * (layer->input_h - 2) * (layer->input_w - 2) + (height - 1) * (layer->input_w - 2) + (width - 1);
					*(padded_input + new_input_offset) = *(layer->inputs + old_input_offset);
				}
			}
		}
	}

	layer->inputs = padded_input;

}




void cnn_convolve(cnn *layer) {


	//loop variables
	int w_kernel, h_kernel, c_in, c_out, w_fmap, h_fmap;
	//partial sum holder
	double partial_sum;
	//size of one input channel
	int input_channel_size = layer->input_h * layer->input_w;
	int input_offset = 0;
	//size of one full kernel
	int kernel_set_size = layer->kernel_w * layer->kernel_h * layer->kernel_ci;
	int kernel_size = layer->kernel_w *layer->kernel_h;
	int weight_offset = 0;
	//size of one output channel
	int output_channle_size = layer->output_h * layer->output_w;
	int output_offset = 0;
	
	double weight = 0, ifmap = 0;

	int tst = 0;
	double tmp = 0.0;
	

	//for output channels
	for (c_out = 0 ; c_out < layer->output_c ; c_out++) {
		//for each row of one output channel
		for (h_fmap = 0 ; h_fmap < layer->output_h ; h_fmap++) {
			//for each column of one output channe;
			for (w_fmap = 0 ; w_fmap < layer->output_w ; w_fmap++) {

				//initialize the partial sum
				partial_sum = 0.0;

				//for input channels
				for (c_in = 0 ; c_in < layer->kernel_ci ; c_in++) {
					//for each row of one input channel
					for (h_kernel = 0 ; h_kernel < layer->kernel_h ; h_kernel++) {
						//for each column of input channel
						for (w_kernel = 0 ; w_kernel < layer->kernel_w ; w_kernel++) {
							
							//begin of input array + number of channels passed * size of channel + number of rows passed * size of one whole row
							// + number of columns passed 
							input_offset = c_in * input_channel_size + (h_fmap * layer->stride + h_kernel) * layer->input_w + (w_fmap * layer->stride + w_kernel);
							ifmap = *( layer->inputs +   input_offset);

							//begin of weights + number of sets passed * set size + number of input channels passed * kernel size + 
							// + number of kernel rows passed * one row of kernel size + number of kernel columns passed
							weight_offset = c_out * kernel_set_size + c_in * kernel_size + h_kernel * layer->kernel_w + w_kernel;
							weight = *( layer->kernel  +  weight_offset);

							partial_sum += ifmap * weight;
							//printf("%0.3f by %0.3f = %0.3f \n", weight, ifmap, ifmap * weight);
						}
					}
				}
				//printf("%.6f", partial_sum);
				tst++;

				//save in output begin address + output channel number * size of one output channel + output row number * one output row size
				//+ output column
				output_offset = c_out * output_channle_size + h_fmap * layer->output_w + w_fmap;
				*(  layer->output  +  output_offset) = partial_sum;

			}
		}
	}

	return;
}