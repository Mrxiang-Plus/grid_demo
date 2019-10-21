add_library(griddemo SHARED
        feature.cpp
        camera.cpp
        )
target_link_libraries(griddemo ${CERES_LIBRARIES} ${OpenCV_LIBS} )