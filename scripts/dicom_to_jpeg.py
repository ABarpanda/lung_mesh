def dicom_to_jpeg(input_dir='./dataset/RSNA', output_dir='./dataset/RSNA_jpeg'):
    """
    Convert DICOM files to JPEG format.
    """
    import os
    import pydicom
    from PIL import Image

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all DICOM files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.dcm'):
            dicom_path = os.path.join(input_dir, filename)
            dicom_data = pydicom.dcmread(dicom_path)

            # Convert DICOM pixel data to a PIL Image
            image = Image.fromarray(dicom_data.pixel_array)

            # Save the image as JPEG
            jpeg_filename = os.path.splitext(filename)[0] + '.jpeg'
            jpeg_path = os.path.join(output_dir, jpeg_filename)
            image.save(jpeg_path)

            print(f'Converted {filename} to {jpeg_filename}')

if __name__ == '__main__':
    dicom_to_jpeg()