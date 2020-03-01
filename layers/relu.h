#ifndef RELU_H 
#define RELU_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Types Declaration
typedef struct {
	double	*input;
	int		size;
	double  *output;
}relu;

//Function Declartion
relu * relu_init(int input_size, double *input);
void relu_process(relu *layer);

relu * relu_init(int input_size, double *input) {

	relu *layer = (relu *)malloc(sizeof(relu));
	if (layer == NULL) {
		printf("Problem allocating dynamic memory in relu\n");
		return 0;
	}
	layer->input = input;
	layer->size =  input_size;
	layer->output = input;
	return layer;
}

void relu_process(relu *layer) {
	double tmp;
	int index;

	for (index = 0; index < layer->size; index++) {
		tmp = *(layer->input + index);
		if (tmp < 0)
			*(layer->input + index) = 0;
	}
}


#endif