CXX = g++ 
CCFLAGS = -std=c++11 -pthread -g -O0 -D CPU_ONPY

TARGET = filter
LIBS = `pkg-config opencv --libs`
INC += -Icaffe
INC += -I/home/yaochen/work/caffe/include
INC += -I../../api_lib/include
INC += -I.`pkg-config opencv --cflags`
SRCS = RprotoW.cpp ../../api_lib/src/layer.cpp  ../../api_lib/src/acc_ctrl.cpp
LDLIBS = -lprotobuf
OBJS = $(SRCS:.cpp=.o)


$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS) $(LDLIBS)

%.o: %.cpp 
	$(CC) -c $< -o $@ $(INC)

clean:
	rm -f *.o ../../api_lib/src/*.o $(TARGET)
