CC = g++
TARGET = filter
LIBS = `pkg-config opencv --libs`
INC = -I../../api_lib/include
INC += -I.`pkg-config opencv --cflags`
SRCS = filter.cpp ../../api_lib/src/layer.cpp  ../../api_lib/src/acc_ctrl.cpp
OBJS = $(SRCS:.cpp=.o)


$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(LIBS)

%.o: %.cpp 
	$(CC) -c $< -o $@ $(INC)

clean:
	rm -f *.o ../../api_lib/src/*.o $(TARGET)
