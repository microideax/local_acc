#ifndef ACC_CTRL_H
#define ACC_CTRL_H
#include <stdint.h>
//==============board level interface============//
#define DEVICE_H2C      "/dev/xdma0_h2c_1"
#define DEVICE_C2H      "/dev/xdma0_c2h_1"
#define DEVICE_CTRL		"/dev/xdma0_user"

#define MAP_SIZE (32*1024UL)


//==============address==================//

#define PARA_MEM_CONV	0x00000000
#define PARA_MEM_POOL   0xC4000000

#define CHECK_TIME 1
#define CTRL_REG 		0x0000



void write_para(uint32_t* para_list,int type);
void write_weight(short int weight[][32],int weight_length,uint32_t weight_mem);
void write_bias(int* bias,int bias_length,uint32_t bias_mem);
void write_data(short int feature[][32],int feature_length,uint32_t data_in_mem);
void start_process();
void read_data_conv(short int feature[][32],int feature_length,uint32_t data_out_mem);
void read_data_pooling(short int feature[][32],int feature_length,uint32_t pooling_out_mem);
void write_weight_bias(short int weight[][32],int weight_length,int *bias,int bias_length,uint32_t weight_mem,uint32_t bias_mem);
void quick_start_conv(
					 uint32_t*  para_list,
					 short int  feature_out[][32],
					 int 	    feature_out_length,
					 uint32_t   data_out_mem
					 );
void quick_start_pooling(
						 uint32_t*  para_list,
						 short int  feature_out[][32],
					 	 int 	    feature_out_length,
					 	 uint32_t   data_out_mem
						);
void disp_performance( uint32_t input_num,
					   uint32_t output_num,
					   uint32_t kernel_size,
					   uint32_t conv_out_size
					);
#endif
