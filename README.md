<!-- markdownlint-disable MD033 -->
# LUNG MESH

## Objective:-

Using deep learning to create a preprocessing module that aligns 2D X-rays to a standard geometric frame by aligning them to a fixed-resolution canonical frame reducing input variability and improve training for downstream models.

### Project Title

End-to-End Geometric Canonicalization for 2D Chest X-Ray Preprocessing and Downstream Robustness

#### Problem Statement

Inter-patient anatomical variance, posture shifts, and framing discrepancies in 2D chest X-rays introduce high input noise, degrading the performance and generalization of downstream computer-aided diagnosis (CAD) models. Current preprocessing relies on heuristic resizing or rigid cropping, which fails to correct rotational, scaling, and structural alignment inconsistencies.

#### Proposed Solution

Develop a modular, multi-stage geometric canonicalization pipeline that decouples anatomical feature localization, aspect-preserving closed-form geometric registration, and quality verification to map raw 2D radiographs into a standardized canonical reference frame.

#### Motivation

Disease and anomaly detection using UNet is a key use case of AI in the medical field. We need to train the AI models as quickly as possible with the highest accuracy. Using a good structured dataset is a pivotal factor in model training. The X-ray can be performed on infants, and on elders above the age of 60. It is possible that their body shape is slightly deviated from the ideal outline of a chest.  
For infants, the size of the chest and lungs will be much smaller with respect to the frame of the image. It is also possible that the test subject is slightly off-center or rotated in the frame. These small errors compound while training the AI model. In this paper, we propose an algorithm to perform preprocessing on the dataset that will be used for training the AI model.

## Baseline

Objective: To set up a baseline of the error experienced by machine learning models, such as diffusion models, while training on raw X-ray images before canonicalization.

## Phase 1

To develop Lung Mesh, we take inspiration from the Mediapipe Face Mesh. It is a machine learning system developed by Google for real-time facial landmark detection. It identifies 468 three-dimensional landmarks across the human face using a single camera without requiring a depth sensor. The system first detects the face and then estimates the position of these landmarks to construct a 3D facial mesh.

### Images

<img src="public/face_mesh.png" alt="https://share.google/9EDpLnTtkCjRlcbp0" width="300" height="300">

<img src="public/face_mask.png" alt="https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png" width="300" height="300">

### Papers

[Lung Segmentation in Chest Radiographs Using Anatomical Atlases With Nonrigid Registration](https://www.researchgate.net/publication/258635659_Lung_Segmentation_in_Chest_Radiographs_Using_Anatomical_Atlases_With_Nonrigid_Registration)

[Development of a Digital Image Database for Chest Radiographs With and Without a Lung Nodule](https://www.ajronline.org/doi/pdf/10.2214/ajr.174.1.1740071)

### Outline

<!-- - Create a baseline of the training error with a un-normalized dataset. -->

Steps to proceed -

1. Figure out how to implement a thorax detector (similar to a face-detector)
2. Build bounding rectangles
3. Basic outline landmarks

### Binarization

To detect thorax and to create a bounding box, first we try to binarize the black-and-white X-ray image.

#### Problem faced -

1. Different X-ray images are of different brightness, so it is difficult to set a constant threshold value for binarization.
2. Even if we are able to binarize the image, the patient's left lung (right lung when viewed from our side) is skewed because of the presence of the heart. Hence, it is impossible to get an exact boundary between the heart and lungs at the junction.
3. Even for the position of the lungs with respect to the lateral walls of the chest, it is difficult to capture the position of the lungs after binarization.

_**Hence, the decision was taken to not use binarization or similar techniques before finding the bounding box.**_

### Edge detection

We can use edge detection techniques by using various kernels and/or Canny edge detectors to find the edges in the X-ray image. This might help us in getting a better idea about where the boundaries of the lungs and the other organs are situated within the chest.

#### Challenges faced -

1. We cannot use Canny or Sobel edge detection algorithms because they rely on a binary image. Since we were unable to binarize the image, Canny and Sobel filters perform extremely badly.
2. The X-ray images are continuous and real-world images, which are grayscale, not black and white. The gradient is very low between the black and white portions. The gradual shading between the black and white parts, with no well defined boundary, poses a major challenge for edge detection.

#### Solution proposed -

1. We perform preprocessing on the raw X-ray image to make the gradients more pronounced. We used the function -

$$
I'(x)=255-\left|255-2I(x)\right|
$$

where I(x) is the original grayscale intensity and I'(x) is the transformed intensity.
This custom transformation was found to be more effective for the specific objective of enhancing Canny edge detection in the evaluated X-ray images. It was motivated by the hypothesis that amplifying intensity transitions would improve the detection of anatomical boundaries.

## Phase 2

### Steps to proceed

The process will include three major steps:

1. Rotation
2. Horizontal Alignment
3. Scaling

For phase 2, we will focus mainly on alignment and scaling.

### Horizontal Alignment

The objective of this code is to Center align the X-ray image. It is possible that the image is off-center, so we will use this score to bring the image to the center.
We do this by making the chest X-ray is symmetric about the vertical axis. That means when we flip the image left to right, the image should be more or less symmetrical.  
  
#### How it works:-

We iteratively remove two columns of pixels from the right on each turn, and then we will flip the image left to right and compare its deviation (that is, the difference between the image before and after getting flipped). Then we will add a single column of black pixels to the right and a single column of black pixels to the left. We continue this until the error is less than twice* the minimum error achieved yet.
We repeat the same process for the left side as well.  
  
(* Can be adjusted to reduce the compute time. If it is kept too high, suppose 3 or 4 times, then we might end up checking all the pixels in the image. If we keep it too low, suppose 1.01 times, then we might end up assuming zero movement to either left or right as the best option.)

#### Importance

This step becomes especially important to remove the unwanted pixels to the left and right of the region of interest. These unwanted dark regions near the armpits can sometimes confuse the U-Net model and cause it to detect some additional patches other than the lungs. Hence, horizontal alignment not only improves the downstream model accuracy but also the efficiency of the scaling process.

<!-- Try k=1.5, 2, 3 for JPCNN003 -->

### Scaling

#### Bounding box

Often, we find X-ray images where the target might be an infant, and the lungs of the patient are located towards the upper half of the image. Hence, we can perform scaling to bring uniformity amongst all the X-ray images, irrespective of the size or shape of the lungs.

For the scaling, we will first use a 2D U-net model to identify the exact position of lungs in the X-ray image. Following are the sources -

* [Kevin Scott Mader - Training U-Net on TB Images to Segment Lungs on Kaggle](https://www.kaggle.com/code/kmader/training-u-net-on-tb-images-to-segment-lungs/)  
* [Eduardo Mineo - U-Net lung segmentation (Montgomery + Shenzhen) on Kaggle](https://www.kaggle.com/code/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen)

Instead of training a new model, we can directly use the segmentation model trained by them. The UNet lung segmentation model was trained on two small datasets: the Montgomery dataset and the Shenzhen frontal X-ray dataset. These datasets are generally used for tuberculosis-related abnormalities detection. This model will be used as a foundation to identify the lung as well as to build a bounding box around our organs of interest.  
After creating the bounding box for the target image, we can perform scaling such that the position and proportion of the area of interest (lungs in particular) is similar for all images in the dataset.

#### Choice of proportion -

After performing alignment, we have already fixed the horizontal position of the image. To fix the vertical position of the image, we will try to make sure that the lung covers 75% of the vertical height of the whole image. <!-- However, the volume of the lungs in the image is also a point of concern, but it will be dealt with in phase 3. -->

* Case 1 - Lungs are smaller than required
  * If, in the original image, the vertical height of the lungs is less than 75% of the total height, then we will scale the image by an equivalent proportion.
  * This will result in enlargement of the original image with respect to the 2048x2048 frame that we are working with. This means some of the outer edges of the original image will be trimmed off.  
  <img src="public/zoomed_in.png" alt="Zoomed in lung image" width="300" height="300">
* Case 2 - Lungs are larger than required
  * Similarly, for images with lung sizes greater than 75% of the total vertical height of the image, we will shrink the image down.
  * If we shrink the image, then we will end up with an image whose dimensions are smaller than our original frame, so we add the image with black (or 0-value pixels) to conform to the frame standard.  
  <img src="public/zoomed_out.png" alt="Zoomed out lung image" width="300" height="300">

To perform the scaling operation, we use an affine matrix. An affine matrix gives a mathematical representation of how an image should be transformed. It allows us to resize an image, shift its position, and apply geometric transformations while preserving the relative structure of the objects in the image.  
Similar to the general affine matrix structure, we construct the following matrix for the transformation.

$$
A=
\begin{bmatrix}
s & 0 & \frac{W_{\text{out}}}{2}-s\,c_x\\[6pt]
0 & s & \frac{H_{\text{out}}}{2}-s\,c_y
\end{bmatrix}
$$

Where `s` is the scaling factor and `c_x` and `c_y` are the horizontal and vertical centers of the bounding box. `s` is calculated as follows

$$
s=\frac{rH_{\text{out}}}{h}
$$
Where `r` is the target height ratio (set at 0.75) and `h` is the bounding box height.

#### Problems faced -

1. Resolution - If the original image is of 64 x 64 dimensions and we try to zoom in on a particular section of the image; the image may get pixelated.
2. Asymmetry - It is possible that the lungs are not symmetrical. Biology dictates that the right lung on the X-ray is slightly bulged inwards due to the position of the heart. This may cause the left lung to elongate downwards or may not be in the ideal shape.

#### Solution Proposed -

1. Since we are working on large images of dimensions 2048 x 2048, the image can be zoomed. However, the model that we are using was trained on a 512 x 512 image dataset. So even if we take a 1/16th part of the image, our model will be able to successfully segment the lungs apart from the background X-ray.

### The Pipeline

The final output of this phase is to produce a pipeline for preprocessing the dataset before using it to train the model. The proposed pipeline consists of four main modules: image alignment, lung segmentation, region-of-interest extraction, and geometric normalisation. The image alignment module corrects horizontal misalignment in the input chest X-ray to establish a consistent orientation for subsequent processing. The lung segmentation module uses a trained deep learning model to identify the lung regions and generate corresponding binary masks, separating the lungs from surrounding anatomical structures. The region-of-interest extraction module derives a bounding box from the segmented lung mask to identify the spatial extent of the lungs. Subsequently, the geometric normalisation module crops and transforms the identified region to achieve a standardised lung-region representation with a predefined aspect ratio. The normalised image is then saved for downstream tasks, such as feature extraction and disease classification. Additionally, the pipeline records alignment information and the affine transformation parameters to support traceability and consistency across processed images. If lung segmentation fails, the pipeline reports the absence of a valid lung region.

## Phase 3

For phase 3, we would venture into anatomical landmark detection and semantic parsing to create a more robust, uniform coordinate grid. We will use a transformation matrix `T` via bicubic or Lanczos interpolation.
