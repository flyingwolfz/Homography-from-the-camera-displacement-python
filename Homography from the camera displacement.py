import cv2
import numpy as np

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, 0:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) *0.025

camera_matrix=np.array([[535.915733961632, 0, 342.2831547330837],
                        [0, 535.915733961632, 235.5708290978817],
                        [0, 0, 1]])
dist=np.array([-2.6637260909660682e-01, -3.8588898922304653e-02,1.7831947042852964e-03, -2.8122100441115472e-04,2.3839153080878486e-01])

img1 = cv2.imread("left02.jpg")
img2 = cv2.imread("left01.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret1, corner1 = cv2.findChessboardCorners(gray1, (9, 6), None)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret2, corner2 = cv2.findChessboardCorners(gray2, (9, 6), None)


H, _ = cv2.findHomography(corner1, corner2)
print("H:\n",H)

retval,rvec1,tvec1 = cv2.solvePnP(objp, corner1, camera_matrix, dist)
retval,rvec2,tvec2 = cv2.solvePnP(objp, corner2, camera_matrix, dist)

R1=cv2.Rodrigues(rvec1)[0]
R2=cv2.Rodrigues(rvec2)[0]

R_1to2=np.dot(R2,R1.T)
T_1to2=np.dot(R2,np.dot(-R1.T,tvec1))+tvec2
normal=np.array([[0 ],[0] ,[1]])
normal1=np.dot(R1,normal)
origin=np.array([[0 ],[0 ],[0]])
origin1=np.dot(R1,origin) + tvec1

d_inv1 = 1.0 / np.sum(np.multiply(normal1, origin1))

R_1to2=R_1to2+ np.dot(d_inv1,np.dot(T_1to2, normal1.T))


H_compute=np.dot(camera_matrix,np.dot(R_1to2,np.linalg.inv(camera_matrix)))
H_compute = H_compute/H_compute[2, 2]
print("H_compute:\n",H_compute)

img1_warp_compute = cv2.warpPerspective(img1, H_compute, (img1.shape[1], img1.shape[0]))
cv2.imwrite('img_warp_H.jpg', img1_warp_compute)

img1_warp_compute = cv2.warpPerspective(img1, H_compute, (img1.shape[1], img1.shape[0]))
cv2.imwrite('img_warp_H_compute.jpg', img1_warp_compute)