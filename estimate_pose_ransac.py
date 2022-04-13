# Imports

import numpy as np
from scipy.spatial.transform import Rotation


# %%

def estimate_pose(uvd1, uvd2, pose_iterations, ransac_iterations, ransac_threshold):
    """
    Estimate Pose by repeatedly calling ransac

    :param uvd1:
    :param uvd2:
    :param pose_iterations:
    :param ransac_iterations:
    :param ransac_threshold:
    :return: Rotation, R; Translation, T; inliers, array of n booleans
    """

    R = Rotation.identity()

    for i in range(0, pose_iterations):
        w, t, inliers = ransac_pose(uvd1, uvd2, R, ransac_iterations, ransac_threshold)
        R = Rotation.from_rotvec(w.ravel()) * R

    return R, t, inliers

def solve_w_t(uvd1, uvd2, R0):
    """
    solve_w_t core routine used to compute best fit w and t given a set of stereo correspondences

    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1, [X/Z, Y/Z, 1/Z]
    :param uvd2: 3xn ndarray : normailzed stereo results from frame 2
    :param R0: Rotation type - base rotation estimate
    :return: w, t : 3x1 ndarray estimate for rotation vector that rotates from frame 2 to frame 1, 3x1 ndarray estimate for translation
    """

    # TODO Your code here replace the dummy return value with a value you compute
    R0 = R0.as_matrix()

    n = uvd1.shape[1]
    d2_prime = uvd2[2]
    u2_prime = uvd2[0]
    v2_prime = uvd2[1]
    u1_prime = uvd1[0]
    v1_prime = uvd1[1]
    y = R0 @ np.vstack((u2_prime, v2_prime, np.ones(n))) # (3,3) @ (3,n) = (3,n)
    for i in range(n):
        a1 = np.hstack((
            np.eye(2), 
            np.array([-u1_prime[i], -v1_prime[i]]).reshape(-1,1)
        ))
        y1 = y[0,i]
        y2 = y[1,i]
        y3 = y[2,i]
        d2 = d2_prime[i]
        a2 = np.array([
            [0, y3, -y2, d2, 0 ,0],
            [-y3, 0, y1, 0, d2, 0],
            [y2, -y1, 0, 0, 0, d2]
        ])
        b_ = -a1 @ y[:,i].reshape(-1,1) # (2,1)
        A_ = a1 @ a2
        if i == 0:
            A = A_
            b = b_
        else:
            A = np.vstack((A,A_))
            b = np.vstack((b,b_))
    # the constructed A should be of shape (2n, 6), b is of shape (2n, 1)

    # solve Ax = b using least sqaures
    x = np.linalg.lstsq(A, b, rcond = None)[0]
    w = x[0:3]
    t = x[3:6]
    return w, t


def find_inliers(w, t, uvd1, uvd2, R0, threshold):
    """

    find_inliers core routine used to detect which correspondences are inliers

    :param w: ndarray with 3 entries angular velocity vector in radians/sec
    :param t: ndarray with 3 entries, translation vector
    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2:  3xn ndarray : normailzed stereo results from frame 2
    :param R0: Rotation type - base rotation estimate
    :param threshold: Threshold to use
    :return: ndarray with n boolean entries : Only True for correspondences that pass the test
    """
    n = uvd1.shape[1]
    diff_norm = np.zeros(n)
    d2_prime = uvd2[2]
    u2_prime = uvd2[0]
    v2_prime = uvd2[1]
    u1_prime = uvd1[0]
    v1_prime = uvd1[1]

    R0 = R0.as_matrix()
    w = w.flatten()
    w_hat = np.array([
        [0,     -w[2], w[1]],
        [w[2], 0,     -w[0]],
        [-w[1], w[0],    0]
    ])
    IplusW = np.eye(3) + w_hat
    T = t.reshape(-1,1)
    for i in range(n):
        a1 = np.hstack((
            np.eye(2), 
            np.array([-u1_prime[i], -v1_prime[i]]).reshape(-1,1)
        ))
        p2 = np.array([
            [u2_prime[i]],
            [v2_prime[i]],
            [1]
        ])
        a2 = IplusW @ R0 @ p2 + d2_prime[i] * T # (3,1)

        diff = a1 @ a2
        diff_norm[i] = np.linalg.norm(diff)
    
    is_inliers = diff_norm < threshold
    return is_inliers


def ransac_pose(uvd1, uvd2, R0, ransac_iterations, ransac_threshold):
    """

    ransac_pose routine used to estimate pose from stereo correspondences

    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2: 3xn ndarray : normailzed stereo results from frame 1
    :param R0: Rotation type - base rotation estimate
    :param ransac_iterations: Number of RANSAC iterations to perform
    :ransac_threshold: Threshold to apply to determine correspondence inliers
    :return: w, t : 3x1 ndarray estimate for rotation vector, 3x1 ndarray estimate for translation
    :return: ndarray with n boolean entries : Only True for correspondences that are inliers

    """
    n = uvd1.shape[1]
    best_w = None
    best_t = None
    best_num_inliers = -1
    best_is_inliers = None
    if ransac_iterations == 0:
        sample_indices = np.random.choice(n,3,replace = False)
        sample_uvd1 = uvd1[:,sample_indices]
        sample_uvd2 = uvd2[:,sample_indices]
        best_w, best_t = solve_w_t(sample_uvd1, sample_uvd2, R0)
        best_is_inliers = np.full(n,True)
    for i in range(ransac_iterations):
        # choose random 3 pairs 
        sample_indices = np.random.choice(n,3,replace = False)
        sample_uvd1 = uvd1[:,sample_indices]
        sample_uvd2 = uvd2[:,sample_indices]
        # Build a model based on the sample set
        w, t= solve_w_t(sample_uvd1, sample_uvd2, R0)
        # Count the inliers for the model
        is_inliers = find_inliers(w, t, uvd1, uvd2, R0, ransac_threshold)
        num_inliers = np.sum(is_inliers)
        # Keep the maximal number of inliners and the best model
        if num_inliers > best_num_inliers:
            best_w = w
            best_t = t
            best_is_inliers = is_inliers
            best_num_inliers = num_inliers
    return best_w, best_t, best_is_inliers
