add_library(griddemo SHARED
        feature.cpp
        )
target_link_libraries(griddemo ${OpenCV_LIBS})