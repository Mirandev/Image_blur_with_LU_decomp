1. open the python file in a text editor
2. change the line 77, image = cv2.imread('gray_smallest.jpg'), with the desired file name for the image
3. comment out line 82 or 83 depending on the blur type you want to use
4. open a terminal and go to the directory that the python file is in
5. type 'python 3sk3_project2.py' to run the program
6. three images should be displayed when the program finishes. The original image, the blurred image, and the image resulting from the LU decomposition solution

NOTE: the program runs very slowly on images 50x50 or above. 25x25 is recommended
NOTE: the output image is not entirely correct