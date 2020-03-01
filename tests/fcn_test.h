#ifndef FCN_TEST_H 
#define FCN_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fcn.h"


double weights[12] = { 0.68470613, -1.53172995, -0.45762036,
						-1.0022496, -0.71529916, -0.0905519,
						-0.91515473, -0.24742825, -2.02612031,
						-1.38199277, -0.03705897, 1.91535371 };

double inputs[3] = { 1.5611388, 1.15466152, -0.18685829 };

void fcn_test();

void fcn_test(){
	int i;
	fcn *layer1 = fcn_init(3, 4, inputs, weights);
	fcn_process(layer1);

	for (i = 0; i < 4; i++) {
		printf("Result %d is %.6f\n", i, *(layer1->output + i));
	}

	return;
}


#endif 