# Estimating Pose of Quadrotor from Stereo Correspondences

## Goal:
Estimate the pose of a flying robot from a set of stereo correspondences. Portions of the EuRoc dataset collected at ETH Zurich are used here. 

## Methodology:

### 1. Estimate rotation matrix and translational vector from stereo correspondences

Given n pairs of homogenous point correspondences between two frames (say frame 1 and frame 2) from two consecutive timesteps (say t1 and t2, t2>t1) and the depths of points from frame 2. Solve the following least square problem. At least three pairs of point correspondences are requried to solve it. 

### 2. Use RANSAC to eliminate outliers in stereo correspondences
Repeat for k iterations
- Choose a minimal sample set: 3 pairs of point correspondences here
- Build a model based on the sample set: Estimate rotation matrix and translational vector from stereo correspondences
- Count the inliers for the model: count the number of point correspondences that make the LHS and RHS of P1 = R @ P2 + T almost equal
- Keep the maximal number of inliners and the best model
