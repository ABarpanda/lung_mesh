import cv2

def binarise(file_path, output_path, threshold_value = 127):
    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    _, binary_img = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

    cv2.imwrite(output_path, binary_img)

if __name__ == "__main__":
    binarise("dataset/toy_dataset/JPCLN003.png", "dataset/toy_dataset/output_binary_image.png", threshold_value=150)