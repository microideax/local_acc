#include "acc_ctrl.h"
#include "cl_tsc.h"
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <assert.h>
using namespace std;

const char *device_h2c = DEVICE_H2C;
const char *device_c2h = DEVICE_C2H;
const char *device_ctrl = DEVICE_CTRL;
void *map_base, *virt_addr;

int rc;
int rc_time;
off_t off;
int fpga_fd;
int fpga_ctrl_fd;

uint64_t time1,time2,time3,time4,sum_time;


void write_para(uint32_t* para_list,int type)     // type   0: conv  1:pooling
{
	fpga_fd= open(device_h2c,O_RDWR);
	assert(fpga_fd >= 0);
	if(type == 0)   //conv
	{
		off = lseek(fpga_fd,PARA_MEM_CONV,SEEK_SET);
		rc = write(fpga_fd,para_list,16*4);
		off = lseek(fpga_fd,PARA_MEM_POOL,SEEK_SET);
		rc = write(fpga_fd,para_list,16*4);
	}
	else	
	{
		//pooling
		off = lseek(fpga_fd,PARA_MEM_POOL,SEEK_SET);
		rc = write(fpga_fd,para_list,16*4);
	}		

	assert(rc == 16*4);
	close(fpga_fd); 
}

void write_weight(short int weight[][32],int weight_length,uint32_t weight_mem)
{
	fpga_fd= open(device_h2c,O_RDWR);
	assert(fpga_fd >= 0);
	off = lseek(fpga_fd,weight_mem,SEEK_SET);
	rc = write(fpga_fd, weight, weight_length);
	assert(rc == weight_length);
	close(fpga_fd); 
}
void write_bias(int* bias,int bias_length,uint32_t bias_mem)
{
	fpga_fd= open(device_h2c,O_RDWR);
	assert(fpga_fd >= 0);
	off = lseek(fpga_fd,bias_mem,SEEK_SET);
	rc = write(fpga_fd, bias, bias_length);
	assert(rc == bias_length);
	close(fpga_fd); 
}

void write_weight_bias(short int weight[][32],int weight_length,int *bias,int bias_length,uint32_t weight_mem,uint32_t bias_mem)
{
	fpga_fd= open(device_h2c,O_RDWR);
	assert(fpga_fd >= 0);

	off = lseek(fpga_fd,weight_mem,SEEK_SET);
	rc = write(fpga_fd, weight, weight_length);
	assert(rc == weight_length);

	off = lseek(fpga_fd,bias_mem,SEEK_SET);
	rc = write(fpga_fd, bias, bias_length);
	assert(rc == bias_length);

	close(fpga_fd); 
}


void write_data(short int feature[][32],int feature_length,uint32_t data_in_mem)
{
	fpga_fd= open(device_h2c,O_RDWR);
	assert(fpga_fd >= 0);
	off = lseek(fpga_fd,data_in_mem,SEEK_SET);
	rc = write(fpga_fd, feature, feature_length);
	assert(feature_length);
	close(fpga_fd); 
}

void start_process()
{
	fpga_ctrl_fd = open(device_ctrl, O_RDWR | O_SYNC);
	assert(fpga_ctrl_fd >= 0);
	map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fpga_ctrl_fd, 0);
	virt_addr = (uint8_t*)map_base + CTRL_REG;

	

	(*((uint32_t *) virt_addr)) = 0x00000001;
	while(((*((uint32_t *) virt_addr)) & 0x00000002)!=0x00000002);

	
	
	close(fpga_ctrl_fd);
}

void read_data_conv(short int feature[][32],int feature_length,uint32_t data_out_mem)
{
	fpga_fd = open(device_c2h, O_RDWR | O_NONBLOCK);
	assert(fpga_fd >= 0);
	off = lseek(fpga_fd,data_out_mem,SEEK_SET);
	rc = read(fpga_fd, feature, feature_length);
	assert(rc == feature_length);
	close(fpga_fd); 
}
void read_data_pooling(short int feature[][32],int feature_length,uint32_t pooling_out_mem)
{
	fpga_fd = open(device_c2h, O_RDWR | O_NONBLOCK);
	assert(fpga_fd >= 0);
	off = lseek(fpga_fd,pooling_out_mem,SEEK_SET);
	rc = read(fpga_fd, feature, feature_length);
	assert(rc == feature_length);
	close(fpga_fd); 
}


void quick_start_conv(uint32_t*  para_list,
					 short int  feature_out[][32],
					 int 	    feature_out_length,
					 uint32_t   data_out_mem
					 )
{	
    // write the params with xdma
	write_para(para_list,0);
    // start the layer processing
    time1 = cycles_to_nanoseconds(ticks());
	start_process();
	time2 = cycles_to_nanoseconds(ticks());


	read_data_conv(feature_out,feature_out_length,data_out_mem);
	sum_time = time2 - time1;
	

	//cout << "**********t1:       :"<<setprecision(5)<<((time4 - time3)/1000000.0) << " ms"<<endl;
}



void quick_start_pooling(uint32_t*  para_list,
						 short int  feature_out[][32],
					 	 int 	    feature_out_length,
					 	 uint32_t   data_out_mem
						)
{
	// write the params with xdma
	write_para(para_list,1);
	time3 = cycles_to_nanoseconds(ticks());
	start_process();
	time4 = cycles_to_nanoseconds(ticks());

	read_data_pooling(feature_out,feature_out_length,data_out_mem);

	sum_time = time4 - time3 + sum_time;
	

	


	//sum_time = (time2 - time1) + (time4 - time3);
}

void disp_performance( uint32_t input_num,
					   uint32_t output_num,
					   uint32_t kernel_size,
					   uint32_t conv_out_size
					)
{
	cout <<"============================================"<<endl;
	cout << "FPGA_ACC.Latency    :"<<setprecision(5)<<(sum_time/1000000.0) << " ms"<<endl;
	cout << "FPGA_ACC.Perf       :"<<setprecision(5)<<1000/(sum_time/1000000.0) << " fps" << endl;
	cout << "FPGA_ACC.Throughput :"<<setprecision(5)<<1000/(sum_time/1000000.0) * input_num * output_num * kernel_size * kernel_size * conv_out_size * conv_out_size / 1000000000.0 << " GOPS"<<endl;
}