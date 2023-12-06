import os
import shutil
import img2line
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pdf2image import convert_from_path
import jax.numpy as jnp
from jax import jit
import tensorflow as tf
import asyncio
import cupy as cp
from numba import cuda

tf.config.run_functions_eagerly(True)

cuda.detect()

# Đọc file pdf và chuyển thành ảnh
def read_pdf(input_path, output_path):
    # Store Pdf with convert_from_path function
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Add the Poppler bin directory to the PATH
    # https://github.com/Belval/pdf2image
    os.environ["PATH"] += os.pathsep + r'poppler/Library/bin'

    images = convert_from_path(input_path)

    for i in range(len(images)):
        # Save pages as images in the pdf
        images[i].save(f"{output_path}/page_{i}.jpg", 'JPEG')


def load_image(input_path):
    # Đọc ảnh và chuyển sang ảnh xám
    image = Image.open(input_path).convert('L')

    # Tính tỷ lệ tăng kích thước
    scale_factor = 6  # Thay đổi tỷ lệ theo ý muốn

    # Tăng kích thước ảnh theo tỷ lệ
    image = image.resize((image.width * scale_factor, image.height * scale_factor), Image.LANCZOS)

    return image

def preprocess_image(image):
    image_array = list(image.getdata())

    image_array = [image_array[offset:offset + image.width] for offset in
                   range(0, image.width * image.height, image.width)]

    # Thang độ xám
    threshold = 128  # Ngưỡng để tách nền và ký tự
    binary_image = [[0 if pixel < threshold else 255 for pixel in row] for row in image_array]

    return binary_image

def merge_overlapping_boxes(boxes):
    # Sort the boxes based on their x-coordinate
    sorted_boxes = sorted(boxes, key=lambda x: x[0])

    merged_boxes = []
    current_box = sorted_boxes[0]

    for box in sorted_boxes[1:]:
        # Check if box is completely inside current_box
        if current_box[0] <= box[0] and current_box[1] <= box[1] and current_box[2] >= box[2] and current_box[3] >= box[
            3]:
            # Box is completely inside, update the current_box to encompass both
            current_box = (
                min(current_box[0], box[0]),
                min(current_box[1], box[1]),
                max(current_box[2], box[2]),
                max(current_box[3], box[3])
            )
        else:
            # No complete containment, add the current merged box to the result
            merged_boxes.append(current_box)
            current_box = box

    # Add the last merged box
    merged_boxes.append(current_box)

    return merged_boxes



def find_character_boxes(image_array, width, height):
    contours = []
    for y in range(height):
        for x in range(width):
            if image_array[y][x] == 0:  # Điểm đen (ký tự)
                x0, x1, y0, y1 = x, x, y, y
                stack = [(x, y)]

                while stack:
                    px, py = stack.pop()
                    if px >= 0 and px < width and py >= 0 and py < height and image_array[py][px] == 0:
                        image_array[py][px] = 255
                        x0, x1 = min(x0, px), max(x1, px)
                        y0, y1 = min(y0, py), max(y1, py)
                        stack.extend(((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)))

                if x1 - x0 > 10 and y1 - y0 > 10:  # Điều kiện để loại bỏ các vùng nhỏ
                    contours.append((x0, 0, x1, height))
    res = merge_overlapping_boxes(contours)
    return res


def find_spaces(binary_image, new_width, new_height):
    character_spaces = []
    boxes = []  # Danh sách chứa toạ độ của box

    for x in range(new_width):
        is_space = True
        for y in range(new_height):
            if binary_image[y][x] == 0:  # Điểm đen (ký tự)
                is_space = False
                break

        if is_space:
            character_spaces.append(x)
        else:
            if character_spaces:
                start_x = character_spaces[0]
                end_x = character_spaces[-1]
                boxes.append((start_x, 0, end_x, new_height))
            character_spaces = []

    return boxes


def classify_spaces(spaces):
    # Calculate the space widths and compute the mean and standard deviation
    space_widths = [x1 - x0 for x0, _, x1, _ in spaces[1:-1:1]]
    mean_width = np.mean(space_widths)
    std_deviation = np.std(space_widths)

    # Set the threshold as a multiple of the standard deviation away from the mean
    threshold = mean_width + 1.15 * std_deviation  # You can adjust the multiplier as needed

    # space between characters in same word
    type_1_spaces = []

    # space between words
    type_2_spaces = []

    for contour in spaces:
        x0, y0, x1, y1 = contour
        space_width = x1 - x0

        if space_width <= threshold:
            type_1_spaces.append(contour)
        else:
            type_2_spaces.append(contour)

    spaces_between_words = type_2_spaces[1::]
    return spaces_between_words

def preprocess_and_crop(image, output_path):

    # Apply Gaussian blur to reduce noise
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=2))

    # Enhance brightness
    enhancer = ImageEnhance.Brightness(blurred_image)
    brightened_image = enhancer.enhance(1.5)  # You can adjust the enhancement factor

    # Find the bounding box of characters
    bbox = find_character_bbox(brightened_image)

    # Crop the image to the bounding box
    cropped_image = brightened_image.crop(bbox)

    # Save the preprocessed and cropped image
    cropped_image.save(output_path)

def find_character_bbox(image):
    # Convert the image to grayscale
    grayscale_image = image.convert('L')

    # Apply a binary threshold to get black and white image
    threshold = 200  # You can adjust the threshold based on your image
    binary_image = grayscale_image.point(lambda p: p < threshold and 255)

    # Get the bounding box of non-white pixels
    bbox = binary_image.getbbox()

    return bbox

def prepare_data(input_pdf, output_path, font_name):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    out_page_path = f"{output_path}/{font_name}_char_collection"
    out_line_path = f"{output_path}/{font_name}_line_img"
    out_char_path = f"{output_path}/{font_name}_char_img"

    read_pdf(input_pdf, out_page_path)

    img2line.process_image(f"{out_page_path}/page_0.jpg", out_line_path)

    i = 0

    # Create the output directory if it doesn't exist
    if not os.path.exists(out_char_path):
        os.makedirs(out_char_path)

    for line in os.listdir(out_line_path):
        line_img_path = f"{out_line_path}/{line}"

        line_image = load_image(line_img_path)

        line_binary_image = preprocess_image(line_image)

        boxes = find_character_boxes(line_binary_image, line_image.width, line_image.height)

        for box in boxes:
            char_image = line_image.crop(box)  # Tạo ảnh con chứa ký tự hoặc khoảng trắng
            preprocess_and_crop(char_image, f"{out_char_path}/ky_tu_{i}.png")
            i += 1
    shutil.rmtree(out_line_path)
    # draw = ImageDraw.Draw(image)
    # for box in contours:
    #     draw.rectangle(box, outline='green')
    # image.show()


def box_to_images(image, combined_boxes, output_path):
    # Lưu từng ký tự và khoảng trắng thành ảnh riêng biệt theo thứ tự
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i, (x0, y0, x1, y1) in enumerate(combined_boxes):
        char_image = image.crop((x0, y0, x1, y1))  # Tạo ảnh con chứa ký tự hoặc khoảng trắng
        preprocess_and_crop(char_image, f'{output_path}/ky_tu_{i}.png')

def box_char_and_space(image, binary_image):
    # char_boxes = find_character_boxes(binary_image, image.width, image.height)
    space_boxes = find_spaces(binary_image, image.width, image.height)
    spaces_between_word_boxes = classify_spaces(space_boxes)

    # Gộp và sắp xếp lại thành một danh sách duy nhất
    combined_boxes = sorted(find_character_boxes(binary_image, image.width, image.height) + spaces_between_word_boxes,
                            key=lambda box: box[0])

    # draw = ImageDraw.Draw(image)
    # for box in combined_boxes:
    #     draw.rectangle(box, outline='green')
    # image.show()

    return combined_boxes


def clear_folder_contents(folder_path):
    # Iterate over the files and subdirectories in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isfile(item_path):
            # If it's a file, remove it
            os.remove(item_path)


def resize_to_match(img1, img2):
    width1, height1 = img1.size
    # width2, height2 = img2.size

    # Resize the smaller image to match the dimensions of the larger image
    # if width1 * height1 < width2 * height2:
    #     img1 = img1.resize((width2, height2))
    # else:
    img2 = img2.resize((width1, height1))

    return img1, img2

@jit
def compare_matrices(image1, image2):
    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same dimensions.")

    # Constants for SSIM calculation
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Compute mean of images
    mu1 = cp.mean(image1)
    mu2 = cp.mean(image2)

    # Compute variance of images
    sigma1_sq = cp.var(image1)
    sigma2_sq = cp.var(image2)

    # Compute covariance
    sigma12 = jnp.cov(image1.flatten(), image2.flatten())[0, 1]

    # SSIM calculation
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_index = numerator / denominator

    return 1 - ssim_index  # Convert SSIM to a distance measure


def extract_char_number(file_name):
    return int(file_name.split("_")[2].split(".")[0])


def extract_line_number(file_name):
    return int(file_name.split("_")[1].split(".")[0])

def get_name(name):
    if len(name) <= 2:
        return name[0]
    elif len(name) > 2:
        if name == 'slash':
            return '/'
        if name == 'colon':
            return ':'
        if name == 'asterisk':
            return '*'
        if name == 'question':
            return '?'
        if name == 'dot':
            return '.'
        if name == 'hyphen':
            return '-'
        if name == 'plus':
            return '+'
        if name == 'comma':
            return ','
        if name == 'semicolon':
            return ';'
        if name == 'parenleft':
            return '('
        if name == 'parenright':
            return ')'
        if name == 'exclam':
            return '!'

def small_process(char_data, data_path, char_img, char_img_arr, ssim_values):
    if char_data.endswith('.png'):
        data_image_path = os.path.join(data_path, char_data)
        data_img = Image.open(data_image_path).convert('L')
        # data_img_arr = list(data_img.getdata())

        char_img, data_img = resize_to_match(char_img, data_img)

        # char_img_arr = np.array(char_img.getdata())
        data_img_arr = np.array(data_img.getdata())
        ssim = compare_matrices(char_img_arr, data_img_arr)
        image_name = os.path.splitext(char_data)[0]
        ssim_values[image_name] = ssim

async def wrapper_function(args):
    char_data, data_path, char_img, char_img_arr, ssim_values = args
    return small_process(char_data, data_path, char_img, char_img_arr, ssim_values)

async def process_data(input_data):
    # Run all tasks concurrently using asyncio.gather
    await asyncio.gather(*(wrapper_function(args) for args in input_data))

def small_process2(char_to_check, data_path, ssim_values, f):
    # page_path = "pages"
    # line_path = "lines"
    char_path = "chars"
    full_char = os.listdir(data_path)
    char_img_path = f"{char_path}/{char_to_check}"
    char_img = Image.open(char_img_path).convert('L')
    char_img_arr = list(char_img.getdata())
    char_img_arr = np.array(char_img.getdata())
    if np.mean(char_img_arr) > 240:
        f.write(" ")
        return
    # for char_data in full_char:
    #     small_process(char_data, data_path, char_img, array)
    
    # Find the character with the highest correlation for the current line
    data_chunks = [(char_data, data_path, char_img, char_img_arr, ssim_values) for char_data in full_char]

    asyncio.run(process_data(data_chunks))
    if ssim_values:
        best_match_name = min(ssim_values, key=ssim_values.get)
        name_to_write = get_name(best_match_name)
        f.write(name_to_write)
    ssim_values.clear()

def small_process3(line, f, data_path):
    # page_path = "pages"
    line_path = "lines"
    char_path = "chars"
    line_img_path = f"{line_path}/{line}"
    line_image = load_image(line_img_path)
    line_binary_image = preprocess_image(line_image)

    boxes = box_char_and_space(line_image, line_binary_image)
    box_to_images(line_image, boxes, char_path)

    char_files = sorted(os.listdir(char_path), key=extract_char_number)
    ssim_values = {}
    for char_to_check in char_files:
        small_process2(char_to_check, data_path, ssim_values, f)
    f.write('\n')
    clear_folder_contents(char_path)

def small_process4(page, f, data_path):
    page_path = "pages"
    line_path = "lines"
    # char_path = "chars"
    page_img_path = f"{page_path}/{page}"
    img2line.process_image(page_img_path, line_path)
    line_files = sorted(os.listdir(line_path), key=extract_line_number)
    for line in line_files:
        small_process3(line, f, data_path)

    f.write('\n')

def big_process(input_path, data_path):
    # Hàm tổng hợp và thực hiện toàn bộ quy trình xử lý ảnh
    page_path = "pages"
    # line_path = "lines"
    # char_path = "chars"
    read_pdf(input_path, page_path)
    f = open("demofile2.txt", "a")

    for page in os.listdir(page_path):
        small_process4(page, f, data_path)

import time
print("Start processing: ")
start_time = time.time()
big_process("input_pdf_tnr.pdf", "data/times_new_roman_char_img")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")
