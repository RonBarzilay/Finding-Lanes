import cv2
import numpy as np
import matplotlib.pyplot as plt


# Functions Declaration
def canny(image):
    """
    canny is a function that:
    Coloring input image in Gray color
    GaussianBlur the Gray image
    Finally, removes pixels under 50 and keeps pixels above 150,
    for white pixels only
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # GaussianBlur on our image for a sharper gray-color difference
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny on our image marks low-high color changing difference
    # Low - wont be presented
    # High - will be presented
    # Between - depends on if its noticeable color changing
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_intrest(image):
    """
    This function should get a canny(picture):
    Read height dimension of image
    Creates array of polygon points: depends on X & Y in image
    Creates a mask image of zeros with the same shape and type as a given array
    Fill the mask with polygon color 255 (White)
    Bitwise_and for keeping only white pixels (255) in mask as in input image
    """
    height = image.shape[0]
    # About lane ABC(x,y)
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    # Black image pixels - zero like
    mask = np.zeros_like(image)
    # fill mask with triangle by color white
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    """
    This function creates a mask image (for input image) of zeros with the same shape and type as a given array
    As long as lines is not empty list:
    Create an array of 2 points (x1, y1) AND (x2, y2) for line in lines
    Create Line in mask (black) by 2 points, blue colored the line, thickness 10
    :param image:
    :param lines:
    :return: line_image (mask)
    """
    mask = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # print(x1, ", ", y1, ",",  x2, ", ", y2)
            cv2.line(mask, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return mask


def make_cordinates(image, line_parameters):
    """
    This function returns A(x, y1), B(x, y2) cordinates for given line_parameters[m, b]
    The returned A, B cordinates depend on entered y1,y2 (exmpl: y1=bottom_img_y, y2=mid_img_y)
    Then, calculate x1,x2 based on given line_parameters[m, b] and y1,y2
    :param image: lane_image
    :param line_parameters: left[avg_m, avg_intercept], right[avg_m, avg_intercept]
    :return:returns A(x, y1), B(x, y2) cordinates for given line_parameters[m, b]
    """
    slope, intercept = line_parameters
    # Where image is 700 px - y start
    y1 = image.shape[0]
    # Around mid-image - y end
    y2 = int(y1*(3/5))

    # y1 = mx1+b => y1 = slope*x1+intercept => x1 = [y1-intercept]/[slope]
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])


def averege_slope_intercept(image, lines):
    """
    This function gets line_image & lines A(x1, y1)B(x2, y2)...:
    From each (x1, y1) (x2, y2), we create a line by 2 points: {start_x, start_y} {end_x, end_y} of line
    The result is for each line in for loop there is: [slope, intercept] => L1[m1, b1], L2[m2, b2] ...
    Negative m/slope is a left_fit line while positive m/slope is a right_fit line

    Finally, average each left and right fit to a left_fit_average & right_fit_average for an
    average of left[avg_m, avg_b] as well as right[avg_m, avg_b]

    :param image: lane_image
    :param lines: lines_image
    :return:
    """
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # This method creates line by 2 points
        # in 1st[0] place: m, in 2nd[1] place: b
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # slope = m - שיפוע הגרף
        slope = parameters[0]
        intercept = parameters[1]
        # if slope is less than 0
        # the line is on the left side (in picture, left has a negative slope)
        if slope < 0:
            left_fit.append((slope, intercept))
        # the line is on the right side (in picture, right has a positive slope)
        else:
            right_fit.append((slope, intercept))

        # axis = 0 - for vertically average of: [[slop1, inter1], [slop2, inter2], [slop3, inter3]]
        # left_fit_average[0] = negative avg m for all left lines
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = make_cordinates(image, left_fit_average)
        # right_fit_average[0] = positive avg m for all right lines
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = make_cordinates(image, right_fit_average)

    print("left lines average (m, b): ", "m", left_fit_average[0], "b", left_fit_average[1])
    print("right lines average (m, b): ", "m", right_fit_average[0], "b", right_fit_average[1])
    return np.array([left_line, right_line])


'''
For lane image detection 
'''
# image = cv2.imread('finding-lanes/test_image.jpg')
# lane_image = np.copy(image)
#
# canny_image = canny(lane_image)
# cropped_image = region_of_intrest(canny_image)
# # Lane line's under 40px AND line gap above 5 is rejected
# # This function should get more documentation
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# # After HoughLinesP Created lots of detected lines - we want to average the lines to one left & right line
# average_lines = averege_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, average_lines)
# # Blend line_image in original lane_image (input image)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow("title here", combo_image)
# cv2.waitKey(0)

'''
For lane video detection 
'''
cap = cv2.VideoCapture('./test2.mp4')
while cap.isOpened():
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_intrest(canny_image)
    '''
    HoughLinesP - Hough Transform Algorithm:
    Assuming we have 2 points, through each of the points, there are infinite number of lines that can go through
    The line that connects 2 points is the line that has the EXACT SAME slope(m) & intercept(b) for both points
    
    Assuming we create All-Possibility-Lines graph from each point in Lane Image & graph when x: slope(m) y: intercept(b)
    and the point in graph that has the most meetings is the most common slope & intercept for all points in Lane Image
    *which means its the most common line - with the most points from Lane Image on it*
    
    Lane line's under 40px AND line gap above 5px is rejected
    '''
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # After HoughLinesP Created lots of detected lines - we want to average the lines to one left & right line
    average_lines = averege_slope_intercept(frame, lines)
    line_image = display_lines(frame, average_lines)

    # Blend lines from HoughLinesP in original lane image/video
    # giving each frame 0.8 weight while 1 weight for line_image
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("title here", combo_image)
    if cv2.waitKey(1) and 0xFF is ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

