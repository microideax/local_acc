#ifndef LAYER_H
#define LAYER_H
#include <stdint.h>
#include <iostream>
void conv_layer_construct(uint32_t 	input_num,
						 uint32_t 	output_num,
						 uint32_t 	kernel_size,
						 uint32_t 	feature_in_size,
						 uint32_t 	feature_out_size,
						 uint32_t 	stride,
						 uint32_t 	padding,
						 uint32_t 	act,	
						 short int  feature_out[][32],
						 int        feature_out_length,
						 uint32_t   weight_offset,
						 uint32_t   bias_offset,
						 uint32_t   data_in_offset,
						 uint32_t   data_out_offset,
						 uint32_t   data_out_mem
						 );
void pooling_layer_construct(uint32_t feature_in_size,
							uint32_t  input_num,
							uint32_t kernel_size,
							uint32_t feature_out_size,
							uint32_t stride,
							uint32_t padding,
							uint32_t act,
							short int  feature_out[][32],
						 	int        feature_out_length,
						 	uint32_t   data_in_offset,
						 	uint32_t   data_out_offset,
						 	uint32_t   data_out_mem
							);
#endif