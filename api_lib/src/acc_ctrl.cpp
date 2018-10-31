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

#if CHECK_TIME == 1
	uint64_t time1,time2;
#endif

void write_para(uint32_t* para_list)
{
	fpga_fd= open(device_h2c,O_RDWR);
	assert(fpga_fd >= 0);
	off = lseek(fpga_fd,PARA_MEM,SEEK_SET);
	rc = write(fpga_fd,para_list,16*4);
	assert(rc == 16*4);
	close(fpga_fd); 
}

void write_weight(short int weight[][32],int weight_length)
{
	fpga_fd= open(device_h2c,O_RDWR);
	assert(fpga_fd >= 0);
	off = lseek(fpga_fd,WEIGHT_MEM,SEEK_SET);
	rc = write(fpga_fd, weight, weight_length);
	assert(rc == weight_length);
	close(fpga_fd); 
}

void write_data(short int feature[][32],int feature_length)
{
	fpga_fd= open(device_h2c,O_RDWR);
	assert(fpga_fd >= 0);
	off = lseek(fpga_fd,DATA_IN_MEM,SEEK_SET);
	rc = write(fpga_fd, feature, feature_length);
	assert(feature_length);
	close(fpga_fd); 
}

void start_process(uint32_t input_num,uint32_t output_num,uint32_t kernel_size,uint32_t feature_in_size)
{
	fpga_ctrl_fd = open(device_ctrl, O_RDWR | O_SYNC);
	assert(fpga_ctrl_fd >= 0);
	map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fpga_ctrl_fd, 0);
	virt_addr = (uint8_t*)map_base + CTRL_REG;


	#if CHECK_TIME == 1
		time1 = cycles_to_nanoseconds(ticks());
	#endif

	(*((uint32_t *) virt_addr)) = 0x00000001;
	while(((*((uint32_t *) virt_addr)) & 0x00000002)!=0x00000002);


	#if CHECK_TIME == 1
		time2 = cycles_to_nanoseconds(ticks());
		cout <<"============================================"<<endl;
		cout << "FPGA_ACC.Perf       :"<<setprecision(5)<<1000/((time2 - time1)/1000000.0) << " fps" << endl;
		cout << "FPGA_ACC.Throughput :"<<setprecision(5)<<1000/((time2 - time1)/1000000.0) * input_num * output_num * kernel_size * kernel_size * feature_in_size * feature_in_size / 1000000000.0 << " GOPS"<<endl;
		cout << "FPGA_ACC.Latency    :"<<setprecision(5)<<((time2 - time1)/1000000.0) << " ms"<<endl;
	#endif
	close(fpga_ctrl_fd);
}

void read_data(short int feature[][32],int feature_length)
{
	fpga_fd = open(device_c2h, O_RDWR | O_NONBLOCK);
	assert(fpga_fd >= 0);
	off = lseek(fpga_fd,DATA_OUT_MEM,SEEK_SET);
	rc = read(fpga_fd, feature, feature_length);
	assert(rc == feature_length);
	close(fpga_fd); 
}

void quick_start(uint32_t 	input_num,
				 uint32_t 	output_num,
				 uint32_t 	kernel_size,
				 uint32_t   feature_in_size,
				 uint32_t* para_list,
				 short int weight[][32],
				 int       weight_length,
				 short int feature_in[][32],
				 int       feature_in_length,
				 short int feature_out[][32],
				 int 	   feature_out_length
				 )
{
	fpga_fd= open(device_h2c,O_RDWR);
	assert(fpga_fd >= 0);
	off = lseek(fpga_fd,PARA_MEM,SEEK_SET);
	rc = write(fpga_fd,para_list,16*4);
	assert(rc == 16*4);

	off = lseek(fpga_fd,WEIGHT_MEM,SEEK_SET);
	rc = write(fpga_fd, weight, weight_length);
	assert(rc == weight_length);

	off = lseek(fpga_fd,DATA_IN_MEM,SEEK_SET);
	rc = write(fpga_fd, feature_in, feature_in_length);
	assert(rc == feature_in_length);

	close(fpga_fd); 
	start_process(input_num,output_num,kernel_size,feature_in_size);
	read_data(feature_out,feature_out_length);
}