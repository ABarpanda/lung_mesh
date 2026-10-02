# LUNG MESH

## Objective:-

Using deep learning to create a preprocessing module that aligns 2D X-rays to a standard geometric frame by aligning them to a fixed-resolution canonical frame reducing input variability and improve training for downstream models.

### Project Title

End-to-End Geometric Canonicalization for 2D Chest X-Ray Preprocessing and Downstream Robustness

#### Problem Statement

Inter-patient anatomical variance, posture shifts, and framing discrepancies in 2D chest X-rays introduce high input noise, degrading the performance and generalization of downstream computer-aided diagnosis (CAD) models. Current preprocessing relies on heuristic resizing or rigid cropping, which fails to correct rotational, scaling, and structural alignment inconsistencies.

#### Proposed Solution

Develop a deep learning-based spatial alignment module that automatically maps raw 2D radiographs into a standardized, fixed-resolution canonical frame. Using a differentiable Spatial Transformer Network (STN) guided by key anatomical landmarks, the module normalizes orientation, scale, and translation dynamically.

## Baseline

Objective: To set up a baseline of the error experienced by machine learning models, such as diffusion models, while training on raw X-ray images before canonicalization.

## Phase 1

### Images

[Canonical face model UV visualization](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)

[Face landmarks mesh map](https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/mesh_map.jpg)

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

1. We cannot use Canny or Sobel edge detection algorithms because they rely on a binary image. Since we were unable to binarize the image, canyon soil filters perform extremely badly.
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
2. Alignment
3. Scaling

For phase 2, we will focus mainly on scaling and alignment

#### Alignment

The objective of this code is to Center align the X-ray image. It is possible that the image is off-center, so we will use this score to bring the image to the center.
We do this by making the chest X-ray is symmetric about the vertical axis. That means when we flip the image left to right, the image should be more or less symmetrical.  
  
How it works:  
We iteratively remove two columns of pixels from the right on each turn, and then we will flip the image left to right and compare its deviation (that is, the difference between the image before and after getting flipped). Then we will add a single column of black pixels to the right and a single column of black pixels to the left. We continue this until the error is lesser than twice* the minimum error achieved yet.
We repeat the same process for the left side as well  
  
(* Can be adjusted to reduce the compute time. If it is kept too high, suppose 3 or 4 times, then we might end up checking all the pixels in the image. If we keep it too low, suppose 1.01 times, then we might end up assuming zero movement to either left or right as the best option. )
