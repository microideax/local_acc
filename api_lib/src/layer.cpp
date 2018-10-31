#include "layer.h"
#include "acc_ctrl.h"
#include <iostream>
using namespace std;
uint32_t para_list[16] = {0};

void layer_construct(uint32_t 	input_num,
					 uint32_t 	output_num,
					 uint32_t 	kernel_size,
					 uint32_t 	feature_in_size,
					 uint32_t 	feature_out_size,
					 uint32_t 	stride,
					 uint32_t 	padding,
					 uint32_t 	act,	
					 short int  weight[][32],
					 int     	weight_length,
					 short int  feature_in[][32],
					 int 		feature_in_length,
					 short int  feature_out[][32],
					 int        feature_out_length
					 )
{
	para_list[0] = input_num;
	para_list[1] = kernel_size;
	para_list[2] = output_num;
	para_list[3] = 320;
	para_list[4] = 320;
	para_list[5] = 320;
	para_list[6] = 320;
	para_list[7] = stride;
	para_list[8] = padding;
	para_list[9] = act;
	para_list[10] = WEIGHT_OFFSET;
	para_list[11] = BIAS_OFFSET;
	para_list[12] = DATA_IN_OFFSET;
	para_list[13] = DATA_OUT_OFFSET;
	para_list[14] = 0;
	para_list[15] = 0;

	quick_start(input_num,
				output_num,
			    kernel_size,
			    feature_in_size,
		        para_list,
				weight,
				weight_length,
				feature_in,
				feature_in_length,
				feature_out,
				feature_out_length
			   );
}




