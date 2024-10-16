# Built in packages
import math
import sys

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# Define constant and global variables
TEST_MODE = False    # Please, DO NOT change this variable!

def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array


###########################################
### You can add your own functions here ###
###########################################

#step 1 functions
def convertToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b):
    greyscale_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            greyscale_value = int(0.3 * px_array_r[i][j] + 0.6 * px_array_g[i][j] + 0.1 * px_array_b[i][j])
            greyscale_array[i][j] = greyscale_value
    return greyscale_array

def contrastStretching(image_width, image_height, greyscale_array):
    flat_array = [pixel for row in greyscale_array for pixel in row]
    f_min = sorted(flat_array)[int(len(flat_array) * 0.05)]
    f_max = sorted(flat_array)[int(len(flat_array) * 0.95)]
    
    stretched_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            stretched_value = (greyscale_array[i][j] - f_min) * 255 / (f_max - f_min)
            stretched_array[i][j] = max(0, min(255, int(stretched_value)))
    return stretched_array

#step 2 functions
def applyScharrFilter(image_width, image_height, greyscale_array, scharr_filter):
    filtered_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            pixel_value = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    pixel_value += greyscale_array[i + k][j + l] * scharr_filter[k + 1][l + 1]
            filtered_array[i][j] = pixel_value
    return filtered_array

def computeEdgeMagnitude(image_width, image_height, edge_x, edge_y):
    edge_magnitude = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            edge_magnitude[i][j] = abs(edge_x[i][j]) + abs(edge_y[i][j])
    return edge_magnitude

#step 3 functions
def applyMeanFilter(image_width, image_height, greyscale_array):
    mean_filter = [
        [1/25, 1/25, 1/25, 1/25, 1/25],
        [1/25, 1/25, 1/25, 1/25, 1/25],
        [1/25, 1/25, 1/25, 1/25, 1/25],
        [1/25, 1/25, 1/25, 1/25, 1/25],
        [1/25, 1/25, 1/25, 1/25, 1/25]
    ]
    
    filtered_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(2, image_height - 2):
        for j in range(2, image_width - 2):
            pixel_value = 0
            for k in range(-2, 3):
                for l in range(-2, 3):
                    pixel_value += greyscale_array[i + k][j + l] * mean_filter[k + 2][l + 2]
            filtered_array[i][j] = abs(pixel_value)
    return filtered_array

def applyMeanFilterThreeTimes(image_width, image_height, greyscale_array):
    result_array = greyscale_array
    for _ in range(3):
        result_array = applyMeanFilter(image_width, image_height, result_array)
    return result_array

#Step 4 fucntions 
def applyThresholding(image_width, image_height, greyscale_array, threshold):
    thresholded_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if greyscale_array[i][j] < threshold:
                thresholded_array[i][j] = 0
            else:
                thresholded_array[i][j] = 255
    return thresholded_array

#Step 5 functions
circular_kernel = [
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
]

def dilate(image_width, image_height, greyscale_array, kernel):
    kernel_size = len(kernel)
    offset = kernel_size // 2
    dilated_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            max_value = 0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    if kernel[ki][kj] == 1:
                        ni = i + ki - offset
                        nj = j + kj - offset
                        if 0 <= ni < image_height and 0 <= nj < image_width:
                            max_value = max(max_value, greyscale_array[ni][nj])
            dilated_array[i][j] = max_value
    return dilated_array

def erode(image_width, image_height, greyscale_array, kernel):
    kernel_size = len(kernel)
    offset = kernel_size // 2
    eroded_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            min_value = 255
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    if kernel[ki][kj] == 1:
                        ni = i + ki - offset
                        nj = j + kj - offset
                        if 0 <= ni < image_height and 0 <= nj < image_width:
                            min_value = min(min_value, greyscale_array[ni][nj])
            eroded_array[i][j] = min_value
    return eroded_array

def applyDilationAndErosion(image_width, image_height, greyscale_array, kernel, num_steps):
    result_array = greyscale_array
    for _ in range(num_steps):
        result_array = dilate(image_width, image_height, result_array, kernel)
    for _ in range(num_steps):
        result_array = erode(image_width, image_height, result_array, kernel)
    return result_array

#Step 6 functions:
def connectedComponentLabeling(image_width, image_height, binary_array):
    label = 0
    label_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if binary_array[i][j] == 255 and label_array[i][j] == 0:
                label += 1
                stack = [(i, j)]
                while stack:
                    ci, cj = stack.pop()
                    if label_array[ci][cj] == 0:
                        label_array[ci][cj] = label
                        for ni in range(ci-1, ci+2):
                            for nj in range(cj-1, cj+2):
                                if 0 <= ni < image_height and 0 <= nj < image_width:
                                    if binary_array[ni][nj] == 255 and label_array[ni][nj] == 0:
                                        stack.append((ni, nj))
    return label_array

#Step 7 functions
def extractBoundingBoxes(image_width, image_height, label_array, min_size=50):
    bounding_boxes = []
    max_label = max(max(row) for row in label_array)
    for label in range(1, max_label + 1):
        min_x, min_y = image_width, image_height
        max_x, max_y = 0, 0
        for i in range(image_height):
            for j in range(image_width):
                if label_array[i][j] == label:
                    min_x = min(min_x, j)
                    max_x = max(max_x, j)
                    min_y = min(min_y, i)
                    max_y = max(max_y, i)
        width = max_x - min_x
        height = max_y - min_y
        if width > min_size and height > min_size:
            # Adjust the bounding box to make it tighter
            min_x = min_x + 8
            min_y = min_y + 8
            max_x = max_x - 8
            max_y = max_y - 8
            bounding_boxes.append([min_x, min_y, max_x, max_y])
    return bounding_boxes


# This is our code skeleton that performs the coin detection.
def main(input_path, output_path):
    # This is the default input image, you may change the 'image_name' variable to test other images.
    image_name = 'easy_case_6'
    input_filename = f'./Images/easy/{image_name}.png'
    if TEST_MODE:
        input_filename = input_path

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)
    
    ###################################
    ### STUDENT IMPLEMENTATION Here ###
    ###################################
    
    #Step 1
    greyscale_array = convertToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)
    stretched_array = contrastStretching(image_width, image_height, greyscale_array)
    
    #Step 2
    scharr_x = [
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3]
    ]
    
    scharr_y = [
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3]
    ]
    
    edge_x = applyScharrFilter(image_width, image_height, stretched_array, scharr_x)
    edge_y = applyScharrFilter(image_width, image_height, stretched_array, scharr_y)
    edge_magnitude = computeEdgeMagnitude(image_width, image_height, edge_x, edge_y)
    
    #Step 3
    blurred_array = applyMeanFilterThreeTimes(image_width, image_height, edge_magnitude)
    
    #Step 4
    thresholded_array = applyThresholding(image_width, image_height, blurred_array, threshold=200)

    #Step 5
    num_dilation_steps = 3  
    num_erosion_steps = 3 
    final_array = applyDilationAndErosion(image_width, image_height, thresholded_array, circular_kernel, num_dilation_steps)

    #Step 6
    label_array = connectedComponentLabeling(image_width, image_height, final_array)

    #step 7
    bounding_box_list = extractBoundingBoxes(image_width, image_height, label_array)

    color_image = []
    for i in range(image_height):
        color_row = []
        for j in range(image_width):
            color_row.append([px_array_r[i][j], px_array_g[i][j], px_array_b[i][j]])
        color_image.append(color_row)

    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(color_image, aspect='equal')

    ############################################
    ### Bounding box coordinates information ###
    ### bounding_box[0] = min x
    ### bounding_box[1] = min y
    ### bounding_box[2] = max x
    ### bounding_box[3] = max y
    ############################################
    
    # Loop through all bounding boxes
    for bounding_box in bounding_box_list:
        bbox_min_x = bounding_box[0]
        bbox_min_y = bounding_box[1]
        bbox_max_x = bounding_box[2]
        bbox_max_y = bounding_box[3]
        
        bbox_xy = (bbox_min_x, bbox_min_y)
        bbox_width = bbox_max_x - bbox_min_x
        bbox_height = bbox_max_y - bbox_min_y
        rect = Rectangle(bbox_xy, bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
        axs.add_patch(rect)
        
    pyplot.axis('off')
    pyplot.tight_layout()
    default_output_path = f'./output_images/{image_name}_with_bbox.png'
    if not TEST_MODE:
        # Saving output image to the above directory
        pyplot.savefig(default_output_path, bbox_inches='tight', pad_inches=0)
        
        # Show image with bounding box on the screen
        pyplot.show()
    else:
        # Please, DO NOT change this code block!
        pyplot.savefig(output_path, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    num_of_args = len(sys.argv) - 1
    
    input_path = None
    output_path = None
    if num_of_args > 0:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        TEST_MODE = True
    
    main(input_path, output_path)
    