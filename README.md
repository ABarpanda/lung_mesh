# LUNG MESH

## Objective:-

Using deep learning to create a preprocessing module that aligns 2D X-rays to a standard geometric frame by aligning them to a fixed-resolution canonical frame reducing input variability and improve training for downstream models.

#### Project Title

End-to-End Geometric Canonicalization for 2D Chest X-Ray Preprocessing and Downstream Robustness

#### Problem Statement

Inter-patient anatomical variance, posture shifts, and framing discrepancies in 2D chest X-rays introduce high input noise, degrading the performance and generalization of downstream computer-aided diagnosis (CAD) models. Current preprocessing relies on heuristic resizing or rigid cropping, which fails to correct rotational, scaling, and structural alignment inconsistencies.

#### Proposed Solution

Develop a deep learning-based spatial alignment module that automatically maps raw 2D radiographs into a standardized, fixed-resolution canonical frame. Using a differentiable Spatial Transformer Network (STN) guided by key anatomical landmarks, the module normalizes orientation, scale, and translation dynamically.

## Phase 1

### Images

[Canonical face model UV visualization](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)

[Face landmarks mesh map](https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/mesh_map.jpg)

### Papers

[Lung Segmentation in Chest Radiographs Using Anatomical Atlases With Nonrigid Registration](https://www.researchgate.net/publication/258635659_Lung_Segmentation_in_Chest_Radiographs_Using_Anatomical_Atlases_With_Nonrigid_Registration)

[JSRT Dataset](https://www.ajronline.org/doi/pdf/10.2214/ajr.174.1.1740071)

### Outline

- Create a baseline of the training error with a un-normalized dataset.

Steps to proceed -

1. Figure out how to implement a thorax detector (similar to a face-detector)
2. Build bounding rectangles
3. Basic outline landmarks
