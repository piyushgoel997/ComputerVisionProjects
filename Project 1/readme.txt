OS used - Windows 10
IDE - Visual Studio 19 with Visual C++

Note: To run the code in the imgDisplay.cpp file, comment out the code in the vidDispaly.cpp file and uncomment the code of the imgDisplay.cpp file and then compile and run.

Simply compile the code (with opencv) by keeping all the .cpp and .h files in one folder.
After compiling run the executeable and it would launch a small window which displays the view captured by the laptop's camera.
Then the user can press keys to toggle on/off the filters and effects to the images. The user can toggle on as many filters/effects as they like, though the processing time would keep increasing as the number of on filters/effects increase. The key to effect/filter mappings are written below.

q -> quit the program
s -> save the current frame
g -> greyscale effect
b -> apply the gaussian blur
x -> magnitude of the SobelX filter
y -> magnitude of the SobelY filter
m -> gradient of the Sobel filter
l -> blur quantize filter with 15 levels of quantization. (can be easily changed by changing the line 80 of vidDisplay.cpp)
c -> apply the cartoon filter with 15 levels of quantization and a threshold of 20. (can be easily changed by changing the line 86 of vidDisplay.cpp)
7 -> negative of the red channel
8 -> negative of the green channel
9 -> negative of the blue channel
1 -> decrease contrast by 0.1 (can't go below 0, initial value is 1)
2 -> increase contrast by 0.1 (can't go above 4, initial value is 1)
3 -> decrease brightness by 10 (can't go below -260, initial value is 0)
4 -> increase brightness by 10 (can't go above 260, , initial value is 0)
p -> apply the laplace filter
5 -> decrease the mixing ratio of the image named my_image.jpg (has to be present within the same folder) by 0.1 (can't go below 0, initial value 0)
6 -> increase the mixing ratio of the image named my_image.jpg (has to be present within the same folder) by 0.1 (can't go above 1, initial value 0)
e -> apply the sepia effect
f -> mirror the video
u -> flip the video upside down
a -> rotate the video 90 degree anticlockwise (just once)
c -> rotate the video 90 degree clockwise (just once)
z -> increase the filter size of the median blur by two toggles between (1, 3, 5, 7)