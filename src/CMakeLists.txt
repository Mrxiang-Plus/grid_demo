add_library(griddemo SHARED
        feature.cpp
        camera.cpp
        )
target_link_libraries(griddemo ${OpenCV_LIBS} ${CERES_LIBRARIES})