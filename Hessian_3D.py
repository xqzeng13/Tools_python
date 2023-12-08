import cv2 as cv
import SimpleITK as sitk
import cv2
import itk
import os
import numpy as np
from scipy import ndimage


# import SimpleITK as sitk

# from vesselness2d import *
# 这里使用的是MSD数据集中的肝脏血管分割数据集，并且只用已训练好的肝脏分割模型对其进行分割，
# 只保留肝脏区域，图像的灰度范围是[0,200]，血管相较于背景为白色

class vesselness2d:
    def __init__(self, image, sigma, spacing, tau):
        super(vesselness2d, self).__init__()
        # image 为numpy类型，表示n * m 的二维矩阵。
        self.image = image
        # sigma 为list 类型，表示高斯核的尺度。
        self.sigma = sigma
        # spacing 为list类型，表示.nii文件下某一切面下的体素的二维尺寸。如果输入图像本身为二维图像，则为[1,1],如果为三维图像，则为对应的space。
        self.spacing = spacing
        # tau 为float类型，表示比例系数。
        self.tau = tau
        # 图像尺寸
        self.size = image.shape

    # 使用特定的特定sigma尺寸下的高斯核对图像滤波
    # 这里作者并没有使用n*n的卷积核，而是分别使用n*1，1*n的卷积对图像进行x和y方向上的卷积，
    # 并且使用的是最原始的计算高斯函数得到卷积核，而不是直接用现成的高斯卷积核，
    # 通过证明可以发现在两方面的结果是等价的。
    def gaussian(self, image, sigma):
        siz = sigma * 6  # 核的尺寸

        # x轴方向上的滤波
        temp = round(siz / self.spacing[0] / 2)
        x = [i for i in range(-temp, temp + 1)]
        x = np.array(x)
        H = np.exp(-(x ** 2 / (2 * ((sigma / self.spacing[0]) ** 2))))
        H = H / np.sum(H)
        Hx = H.reshape(len(H), 1, 1)
        I = ndimage.filters.convolve(image, Hx, mode='nearest')
        # scipy.ndimage
        # y轴方向上的滤波
        temp = round(siz / self.spacing[1] / 2)
        x = [i for i in range(-temp, temp + 1)]
        x = np.array(x)
        H = np.exp(-(x ** 2 / (2 * ((sigma / self.spacing[1]) ** 2))))
        H = H / np.sum(H[:])
        Hy = H.reshape(1, len(H), 1)
        I = ndimage.filters.convolve(I, Hy, mode='nearest')
        return I

    # 求图像的梯度
    def gradient2(self, F, option):
        k = self.size[0]
        l = self.size[1]
        D = np.zeros(F.shape)
        if option == "x":
            D[0, :] = F[1, :] - F[0, :]
            D[k - 1, :] = F[k - 1, :] - F[k - 2, :]
            # take center differences on interior points
            D[1:k - 2, :] = (F[2:k - 1, :] - F[0:k - 3, :]) / 2
        else:
            D[:, 0] = F[:, 1] - F[:, 0]
            D[:, l - 1] = F[:, l - 1] - F[:, l - 2]
            D[:, 1:l - 2] = (F[:, 2:l - 1] - F[:, 0:l - 3]) / 2
        return D

    # 求海森矩阵中所需要的二阶偏导数
    def Hessian2d(self, image, sigma):
        image = self.gaussian(image, sigma)
        # image = ndimage.gaussian_filter(image, sigma, mode = 'nearest')
        Dy = self.gradient2(image, "y")
        Dyy = self.gradient2(Dy, "y")

        Dx = self.gradient2(image, "x")
        Dxx = self.gradient2(Dx, "x")
        Dxy = self.gradient2(Dx, 'y')
        return Dxx, Dyy, Dxy

    # 求解海森矩阵的两个特征值
    # 这里作者使用求根公式，将二阶海森矩阵展开，a=1,b=-(Ixx+Iyy),c=(Ixx*Iyy-Ixy*Ixy)
    # 首先计算 sqrt(b^2 - 4ac),通过化简得到tmp
    # 最后得到两个特征值mu1，mu2，根据大小关系，大的为mu2，小的为mu1
    def eigvalOfhessian2d(self, Dxx, Dyy, Dxy):
        tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * (Dxy ** 2))
        # compute eigenvectors of J, v1 and v2
        mu1 = 0.5 * (Dxx + Dyy + tmp)
        mu2 = 0.5 * (Dxx + Dyy - tmp)
        # Sort eigen values by absolute value abs(Lambda1) < abs(Lambda2)
        indices = (np.absolute(mu1) > np.absolute(mu2))
        Lambda1 = mu1
        Lambda1[indices] = mu2[indices]

        Lambda2 = mu2
        Lambda2[indices] = mu1[indices]
        return Lambda1, Lambda2

    def imageEigenvalues(self, I, sigma):
        hxx, hyy, hxy = self.Hessian2d(I, sigma)
        c = sigma ** 2
        hxx = -c * hxx
        hyy = -c * hyy
        hxy = -c * hxy

        # 为了降低运算量，去掉噪声项的计算
        B1 = -(hxx + hyy)
        B2 = hxx * hyy - hxy ** 2
        T = np.ones(B1.shape)
        T[(B1 < 0)] = 0
        T[(B1 == 0) & (B2 == 0)] = 0
        T = T.flatten()
        indeces = np.where(T == 1)[0]

        hxx = hxx.flatten()
        hyy = hyy.flatten()
        hxy = hxy.flatten()

        hxx = hxx[indeces]
        hyy = hyy[indeces]
        hxy = hxy[indeces]

        lambda1i, lambda2i = self.eigvalOfhessian2d(hxx, hyy, hxy)
        lambda1 = np.zeros(self.size[0] * self.size[1], )
        lambda2 = np.zeros(self.size[0] * self.size[1], )

        lambda1[indeces] = lambda1i
        lambda2[indeces] = lambda2i

        # 去掉噪声
        lambda1[(np.isinf(lambda1))] = 0
        lambda2[(np.isinf(lambda2))] = 0

        lambda1[(np.absolute(lambda1) < 1e-4)] = 0
        lambda1 = lambda1.reshape(self.size)

        lambda2[(np.absolute(lambda2) < 1e-4)] = 0
        lambda2 = lambda2.reshape(self.size)
        return lambda1, lambda2

    # 血管强化
    def vesselness2d(self):
        for j in range(len(self.sigma)):
            lambda1, lambda2 = self.imageEigenvalues(self.image, self.sigma[j])
            lambda3 = lambda2.copy()
            new_tau = self.tau * np.min(lambda3)
            lambda3[(lambda3 < 0) & (lambda3 >= new_tau)] = new_tau
            different = lambda3 - lambda2
            response = ((np.absolute(lambda2) ** 2) * np.absolute(different)) * 27 / (
                    (2 * np.absolute(lambda2) + np.absolute(different)) ** 3)
            response[(lambda2 < lambda3 / 2)] = 1
            response[(lambda2 >= 0)] = 0

            response[np.where(np.isinf(response))[0]] = 0
            if j == 0:
                vesselness = response
            else:
                vesselness = np.maximum(vesselness, response)
        vesselness[(vesselness < 1e-2)] = 0
        return vesselness


def edge(img, position):
    img_dt = np.zeros((len(img), len(img[0]), len(img[0][0])))
    img_dt[:] = img[:]
    origin = img_dt[0][0][0]
    img_dt[img_dt != -origin] = 1
    img_dt[img_dt == -origin] = 0

    tmp = np.ones((len(img_dt), len(img_dt[0]), len(img_dt[0][0])))
    if position == "x":
        for i in range(len(img_dt)):
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            dst = cv.erode(img_dt[i], kernel)
            tmp[i] = dst
        img_dt[tmp == 1] = 0
    elif position == "y":
        for i in range(len(img_dt[0])):
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            dst = cv.erode(img_dt[:, i, :], kernel)
            tmp[:, i, :] = dst
        img_dt[tmp == 1] = 0
    elif position == "z":
        for i in range(len(img_dt[0][0])):
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            dst = cv.erode(img_dt[:, :, i], kernel)
            tmp[:, :, i] = dst
        img_dt[tmp == 1] = 0
    return img_dt


def frangi(img, sigma, spacing, tau, position):
    img_dt = np.zeros((len(img), len(img[0]), len(img[0][0])))
    img_dt[:] = img[:]
    result_dt = np.zeros((len(img_dt), len(img_dt[0]), len(img_dt[0][0])))
    if position == "x":
        for i in range(len(img_dt)):
            image = img_dt[i]
            output = vesselness2d(image, sigma, spacing, tau)
            output = output.vesselness2d()
            result_dt[i] = output
    elif position == "y":
        for i in range(len(img_dt[0])):
            image = img_dt[:, i, :]
            output = vesselness2d(image, sigma, spacing, tau)
            output = output.vesselness2d()
            result_dt[:, i, :] = output
    elif position == "z":
        for i in range(len(img_dt[0][0])):
            image = img_dt[:, :, i]
            output = vesselness2d(image, sigma, spacing, tau)
            output = output.vesselness2d()
            result_dt[:, :, i] = output
    return result_dt


def Hessian3D(image, sigma, tau):
    img_dt = sitk.GetArrayFromImage(image)
    stand = img_dt[0][0][0]
    img_dt[img_dt == stand] = 0
    img_dt = 1000 - img_dt
    img_dt[img_dt >= 800] = 0

    edge_x = edge(img_dt, "x")
    edge_y = edge(img_dt, "y")
    edge_z = edge(img_dt, "z")
    edge_x[edge_y == 1] = 1
    edge_x[edge_z == 1] = 1

    space = image.GetSpacing()
    spacing_x = [space[0], space[1]]
    spacing_y = [space[0], space[2]]
    spacing_z = [space[1], space[2]]

    hessian_x = frangi(img_dt, sigma, spacing_x, tau, "x")
    # return

    hessian_y = frangi(img_dt, sigma, spacing_y, tau, "y")
    hessian_z = frangi(img_dt, sigma, spacing_z, tau, "z")

    result_dt = hessian_x + hessian_y + hessian_z
    result_dt[-1] = np.zeros((len(result_dt[0]), len(result_dt[0][0])))
    result_dt[edge == 1] = 0
    result_dt *= 400
    result_dt[result_dt > 200] = 200
    result_dt[img_dt == -200] = -200
    result_dt = result_dt.astype(int)

    result = sitk.GetImageFromArray(result_dt)
    result.SetSpacing(image.GetSpacing())
    result.SetOrigin(image.GetOrigin())
    result.SetDirection(image.GetDirection())
    return result


def vessleSegment(niipath, name, savepath):
    sigma_minimum = 0.5
    sigma_maximum = 3
    number_of_sigma_steps = 10
    input_image = itk.imread(niipath)
    # 1.采用itk的多尺度hessian矩阵进行血管增强
    ImageType = type(input_image)
    Dimension = input_image.GetImageDimension()
    HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
    HessianImageType = itk.Image[HessianPixelType, Dimension]
    objectness_filter = itk.HessianToObjectnessMeasureImageFilter[HessianImageType, ImageType].New()  # 增强过滤器
    objectness_filter.SetBrightObject(True)
    objectness_filter.SetScaleObjectnessMeasure(True)
    objectness_filter.SetAlpha(5)  # 最小特征值与最大特征值比值 较小的值导致对物体维度的敏感性提高
    objectness_filter.SetBeta(8)  ##最大特征值与较大特征值的比值
    objectness_filter.SetGamma(5)  # Hessian矩阵的Frobenius范数

    multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[ImageType, HessianImageType, ImageType].New()
    multi_scale_filter.SetInput(input_image)
    multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
    multi_scale_filter.SetSigmaStepMethodToLogarithmic()
    multi_scale_filter.SetSigmaMinimum(sigma_minimum)
    multi_scale_filter.SetSigmaMaximum(sigma_maximum)
    multi_scale_filter.SetNumberOfSigmaSteps(number_of_sigma_steps)
    itk.imwrite(multi_scale_filter.GetOutput(), savepath + name)


def vesselenhance(niipath, name, savepath):
    sigma = 0.5
    alpha1 = 3
    alpha2 = 5
    input_image = itk.imread(niipath)
    ImageType = type(input_image)
    hessianFilter = itk.HessianRecursiveGaussianImageFilter[ImageType].New()
    hessianFilter.SetInput(input_image)
    hessianFilter.SetSigma(sigma)
    vesselnessFilter = itk.Hessian3DToVesselnessMeasureImageFilter[itk.F].New()
    vesselnessFilter.SetInput(hessianFilter.GetOutput())
    vesselnessFilter.SetAlpha1(alpha1)
    vesselnessFilter.SetAlpha1(alpha2)
    itk.imwrite(vesselnessFilter.GetOutput(), savepath + name)

    # itk.imwrite(vesselnessFilter.GetOutput(),"./test-brain/1_4_v.nii.gz")


# 这里的main函数根据自己的需要改
# 这里我的直接对整个文件夹中的全部.nii文件进行处理
if __name__ == "__main__":
    # sigma = [0.5, 1, 1.5, 2, 2.5]
    # tau = 2
    path = r"E:\MRA_SUM\IXI\results\UARAI\\"
    result_path = r"E:\MRA_SUM\IXI\results\UARAI\\"
    path_list = os.listdir(path)
    for i in path_list:
        # image_i_path = os.path.join(path, i)

        image_i_path = r'E:\MRA_SUM\IXI\results\UARAI\IXI234-IOP-0870-MRA_img.nii.gz'
        # img = sitk.ReadImage(image_i_path)
        # name = image_i_path.split('data\\')[-1]
        # result = Hessian3D(img,sigma,tau)
        name='test70_1.nii.gz'
        result = vesselenhance(image_i_path, name, result_path)  # vesselenhance
        # result=vesselenhance(image_i_path,name,result_path)#vesselenhance

        # sitk.WriteImage(result,os.path.join(result_path,i))
        # print(i + " is OK!")
