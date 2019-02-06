CXXFLAGS += -std=c++11 -O3 -Os -Wall

libtemporal-cse.so: brepr.h schedule.h schedules.h temporal-cse.cpp
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $@ $(filter %.cpp, $^)
