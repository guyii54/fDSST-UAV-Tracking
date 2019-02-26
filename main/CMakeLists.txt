#fDSST_gra/main

#file(GLOB SOURCE_FILES *.hpp *.cpp)
add_executable(fDSST_gra main.cpp)

target_link_libraries(fDSST_gra
                                    ${OpenCV_LIBS}
                                    tracker)
