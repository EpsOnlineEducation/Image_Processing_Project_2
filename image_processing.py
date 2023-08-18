#image_processing.py
import cv2
import numpy as np
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap, QImage
from skimage.util import random_noise
from sklearn.cluster import KMeans

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
     
class ImageProcessing(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("image.ui",self)
        self.setWindowTitle("Image Processing Project")
        self.pbImage.clicked.connect(self.open_file)
        self.cboNoise.currentIndexChanged.connect(self.do_noise)
        self.hsMean.valueChanged.connect(self.set_hsMean)
        self.hsVar.valueChanged.connect(self.set_hsVar)
        self.hsAmount.valueChanged.connect(self.set_hsAmount)
        self.leMean.textEdited.connect(self.do_noise)
        self.leVar.textEdited.connect(self.do_noise)
        self.leAmount.textEdited.connect(self.do_noise)
        
        self.sbMinVal.valueChanged.connect(self.set_minval)
        self.sbMaxVal.valueChanged.connect(self.set_maxval)
        self.sbKernel.valueChanged.connect(self.set_kernel)
        self.leMinVal.textEdited.connect(self.do_edge)
        self.leMaxVal.textEdited.connect(self.do_edge)
        self.leKernel.textEdited.connect(self.do_edge)
        self.cboEdge.currentIndexChanged.connect(self.do_edge)

        self.rbMultiple.toggled.connect(self.rbstate)
        self.rbKMeans.toggled.connect(self.rbstate)
        self.sbCluster.valueChanged.connect(self.set_cluster)

        self.cboDenoising.currentIndexChanged.connect(self.do_denoising)
        self.sbH.valueChanged.connect(self.do_denoising)
        self.sbPatch.valueChanged.connect(self.do_denoising)
        self.sbSearch.valueChanged.connect(self.do_denoising)
        self.hsWeight.valueChanged.connect(self.set_hsWeight)
        self.hsSigma.valueChanged.connect(self.set_hsSigma)
        
        
    def set_hsWeight(self, value):
        self.leWeight.setText(str(round((value/10),2)))
        self.do_denoising()

    def set_hsSigma(self, value):
        self.leSigma.setText(str(round((value/100),2)))
        self.do_denoising()
        
    def choose_denoising(self,img): 
        strCB = self.cboDenoising.currentText()
        h = self.sbH.value()
        patch = self.sbPatch.value()
        size = self.sbSearch.value()
        weightVal = float(self.leWeight.text())
        sigmaVal = float(self.leSigma.text())
        noisy = self.choose_noise(img)
        
        if strCB == 'Non-Local Means Denoising': 
            self.hsWeight.setEnabled(False)
            self.leWeight.setEnabled(False)
            self.hsSigma.setEnabled(False)
            self.leSigma.setEnabled(False)
            self.sbH.setEnabled(True)
            self.sbPatch.setEnabled(True)
            self.sbSearch.setEnabled(True)
            denoised_img = cv2.fastNlMeansDenoisingColored(noisy,None,h,h,patch,size)
            return denoised_img
        
        if strCB == 'Total Variation Filter': 
            self.hsWeight.setEnabled(True)
            self.leWeight.setEnabled(True)
            self.hsSigma.setEnabled(False)
            self.leSigma.setEnabled(False)
            self.sbH.setEnabled(False)
            self.sbPatch.setEnabled(False)
            self.sbSearch.setEnabled(False)
            denoised_img = denoise_tv_chambolle(noisy, weight=weightVal, multichannel=True)
            cv2.normalize(denoised_img, denoised_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            denoised_img = denoised_img.astype(np.uint8)
            return denoised_img

        if strCB == 'Bilateral Filter': 
            self.hsWeight.setEnabled(False)
            self.leWeight.setEnabled(False)
            self.hsSigma.setEnabled(True)
            self.leSigma.setEnabled(True)
            self.sbH.setEnabled(False)
            self.sbPatch.setEnabled(False)
            self.sbSearch.setEnabled(False)
            denoised_img = denoise_bilateral(noisy, sigma_color=sigmaVal, sigma_spatial=5, multichannel=True, mode='constant')
            cv2.normalize(denoised_img, denoised_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            denoised_img = denoised_img.astype(np.uint8)
            return denoised_img

        if strCB == 'Wavelet Denoising Filter': 
            self.hsWeight.setEnabled(False)
            self.leWeight.setEnabled(False)
            self.hsSigma.setEnabled(False)
            self.leSigma.setEnabled(False)
            self.sbH.setEnabled(False)
            self.sbPatch.setEnabled(False)
            self.sbSearch.setEnabled(False)
            denoised_img = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True, method='BayesShrink', rescale_sigma=True)
            cv2.normalize(denoised_img, denoised_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            denoised_img = denoised_img.astype(np.uint8)
            return denoised_img

    def do_denoising(self):
        denoised = self.choose_denoising(img)        
        height, width, channel = denoised.shape
        bytesPerLine = channel * width  
        cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB, denoised)
        qImg = QImage(denoised, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.display_image(pixmap, denoised, self.labelFilter, self.widgetHistIm, 'Histogram of Denoised Image')
                
    def rbstate(self):
        nCluster = int(self.leCluster.text())
        noisy = self.choose_noise(img)
        if self.rbMultiple.isChecked() == True:
            output = self.thresh_seg(noisy)
            self.seg_output(output)

        if self.rbKMeans.isChecked() == True:
            output = self.kmeans_seg(noisy,nCluster)
            self.seg_output(output)

    def set_cluster(self):
        self.leCluster.setText(str(self.sbCluster.value())) 
        self.rbKMeans.setChecked(True)
        self.rbstate()
        
    def seg_output(self,output):       
        height, width, channel = output.shape
        bytesPerLine = channel * width  
        cv2.cvtColor(output, cv2.COLOR_BGR2RGB, output)
        qImg = QImage(output, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.display_image(pixmap, output, self.labelFilter, self.widgetHistIm, 'Histogram of Segmented Image')
            
    def thresh_seg(self,img):
        img_seg = np.zeros(img.shape, np.float32)
        if len(img.shape) == 2:
           img_seg = self.multiple_thresh(img)
        else:
            img_seg[:, :, 0] = self.multiple_thresh(img[:, :, 0])
            img_seg[:, :, 1] = self.multiple_thresh(img[:, :, 1])
            img_seg[:, :, 2] = self.multiple_thresh(img[:, :, 2])
            cv2.normalize(img_seg, img_seg, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            img_seg = img_seg.astype(np.uint8)
        return img_seg
    
    def multiple_thresh(self, img):
        img_thresh= img.reshape(img.shape[0]*img.shape[1])
        
        img_thresh[img_thresh < 64] = 0
        img_thresh[(img_thresh < 128) & (img_thresh >= 64)] = 64
        img_thresh[(img_thresh < 192) & (img_thresh >= 128)] = 128
        img_thresh[img_thresh > 192] = 255

        img_mul = img_thresh.reshape(img.shape[0],img.shape[1])
        img_mul = img_mul.astype(np.uint8)        
        return img_mul

    def kmeans_seg(self,img, nCluster):
        img_seg = np.zeros(img.shape, np.float32)
        img_n = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
        kmeans = KMeans(n_clusters=nCluster, random_state=0).fit(img_n)
        pic2show = kmeans.cluster_centers_[kmeans.labels_]
        img_seg = pic2show.reshape(img.shape[0], img.shape[1], img.shape[2])

        cv2.normalize(img_seg, img_seg, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        img_seg = img_seg.astype(np.uint8)
        return img_seg
        
    def open_file(self):
        global img
        self.fname = QFileDialog.getOpenFileName(self, 'Open file','d:\\',"Image Files (*.jpg *.gif *.bmp *.png)")
        pixmap = QPixmap(self.fname[0])
        img = cv2.imread(self.fname[0], cv2.IMREAD_COLOR)
        self.display_image(pixmap, img, self.labelImage, self.widgetHistIm, 'Histogram of Original Image')
        self.setState(True)
        self.hsAmount.setEnabled(False)
        self.leAmount.setEnabled(False)
        
    def setState(self, state):
        self.gbNoise.setEnabled(state)
        self.gbEdge.setEnabled(state)
        self.gbSegmentation.setEnabled(state)
        self.gbDenoising.setEnabled(state) 
        
    def display_image(self, pixmap, img, label, qwidget1, title):       
        label.setPixmap(pixmap)
        label.setScaledContents(True);                
        self.display_histogram(img, qwidget1, title)

    def display_histogram(self, img, qwidget1, title):
        qwidget1.canvas.axes1.clear()
        
        channel = len(img.shape)
        
        if channel == 2: #grayscale image
            histr = cv2.calcHist([img],[0],None,[256],[0,256])
            qwidget1.canvas.axes1.plot(histr,color = 'yellow',linewidth=3.0)
            qwidget1.canvas.axes1.set_ylabel('Frequency',color='white')
            qwidget1.canvas.axes1.set_xlabel('Intensity', color='white')
            qwidget1.canvas.axes1.tick_params(axis='x', colors='white')
            qwidget1.canvas.axes1.tick_params(axis='y', colors='white')
            qwidget1.canvas.axes1.set_title(title,color='white')
            qwidget1.canvas.axes1.set_facecolor('xkcd:black')
            qwidget1.canvas.axes1.set_xticks(np.arange(0,256,15))
            qwidget1.canvas.axes1.grid()
            qwidget1.canvas.draw() 
        
        else : #color image
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,256])
                qwidget1.canvas.axes1.plot(histr,color = col,linewidth=3.0)
                qwidget1.canvas.axes1.set_ylabel('Frequency',color='white')
                qwidget1.canvas.axes1.set_xlabel('Intensity',color='white')
                qwidget1.canvas.axes1.tick_params(axis='x', colors='white')
                qwidget1.canvas.axes1.tick_params(axis='y', colors='white')
                qwidget1.canvas.axes1.set_title(title,color='white')
                qwidget1.canvas.axes1.set_facecolor('xkcd:black')
                qwidget1.canvas.axes1.set_xticks(np.arange(0,256,15))
                qwidget1.canvas.axes1.grid()
            qwidget1.canvas.draw()      

    def gaussian_noise(self,img, mean, sigma):
        # Generate Gaussian noise
        gauss = np.random.normal(mean,sigma,img.size)*np.max(img)
        gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')

        # Add the Gaussian noise to the image
        img_gauss = cv2.add(img,gauss)
        return img_gauss

    def speckle_noise(self,img, mean, sigma):
        # Generate spekcle noise
        noise_img = random_noise(img, mode='speckle', mean=mean, var=sigma) 
        
        imgspeckle = np.array(noise_img*np.max(img) , dtype = 'uint8')
        return imgspeckle

    def salt_pepper_noise(self, img, val):
        # Add salt-and-pepper noise to the image.
        noise_img = random_noise(img, mode='s&p',amount=val)   
        
        # The above function returns a floating-point image
        # on the range [0, 1], thus we changed it to 'uint8'
        # and from [0,255]
        imgsnp = np.array(noise_img*np.max(img), dtype = 'uint8')
        return imgsnp

    def poisson_noise(self, img, mean, amount):
        pois = random_noise(img, mode='poisson')
        imgpois = np.array(pois*np.max(img), dtype = 'uint8')
        return imgpois

    def choose_noise(self,img):        
        strCB = self.cboNoise.currentText()
        mean = float(self.leMean.text())
        var = float(self.leVar.text())
        amount = float(self.leAmount.text())
        sigma = var
        
        if strCB == 'Gaussian':  
            self.hsAmount.setEnabled(False)
            self.leAmount.setEnabled(False)
            self.hsMean.setEnabled(True)
            self.leMean.setEnabled(True)
            self.hsVar.setEnabled(True)
            self.leVar.setEnabled(True)
            noisy_image = self.gaussian_noise(img, mean, sigma)
            return noisy_image
        if strCB == 'Speckle': 
            self.hsAmount.setEnabled(False)
            self.leAmount.setEnabled(False)
            self.hsMean.setEnabled(True)
            self.leMean.setEnabled(True)
            self.hsVar.setEnabled(True)
            self.leVar.setEnabled(True)
            noisy_image = self.speckle_noise(img, mean, sigma)
            return noisy_image
        if strCB == 'Poisson':  
            self.hsMean.setEnabled(False)
            self.leMean.setEnabled(False)
            self.hsVar.setEnabled(False)
            self.leVar.setEnabled(False)
            self.hsAmount.setEnabled(True)
            self.leAmount.setEnabled(True)
            noisy_image = self.poisson_noise(img, mean,sigma)
            return noisy_image
        if strCB == 'Salt & Pepper':  
            self.hsMean.setEnabled(False)
            self.leMean.setEnabled(False)
            self.hsVar.setEnabled(False)
            self.leVar.setEnabled(False)
            self.hsAmount.setEnabled(True)
            self.leAmount.setEnabled(True)
            noisy_image = self.salt_pepper_noise(img, amount)
            return noisy_image

    def do_noise(self):
        noisy = self.choose_noise(img)        
        height, width, channel = noisy.shape
        bytesPerLine = channel * width  
        cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB, noisy)
        qImg = QImage(noisy, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.display_image(pixmap, noisy, self.labelFilter, self.widgetHistFilter, 'Histogram of Noisy Image')

    def set_hsMean(self, value):
        self.leMean.setText(str(round((value/100),2)))
        self.do_noise()
        
    def set_hsVar(self, value):
        self.leVar.setText(str(round((value/500),3)))
        self.do_noise()
        
    def set_hsAmount(self, value):
        self.leAmount.setText(str(round((value/20),2)))
        self.do_noise()

    def canny_edge(self, img, minVal, maxVal):      
        edge_im = np.zeros(img.shape, np.float32)
        edge_im = cv2.GaussianBlur(img,(3,3),0)   
        
        if len(img.shape) == 2:
           edge_im = cv2.Canny(edge_im, minVal, maxVal)
        else:#Color image
            edge_im[:, :, 0] = cv2.Canny(img[:, :, 0],minVal, maxVal)
            edge_im[:, :, 1] = cv2.Canny(img[:, :, 1],minVal, maxVal)
            edge_im[:, :, 2] = cv2.Canny(img[:, :, 2],minVal, maxVal)
            cv2.normalize(edge_im, edge_im, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            edge_im = edge_im.astype(np.uint8)
        return edge_im

    def sobelxy_edge(self, img, d1, d2, kernel):      
        edge_im = np.zeros(img.shape, np.float32)
        edge_im = cv2.GaussianBlur(edge_im,(3,3),0)   
        
        if len(img.shape) == 2:
           edge_im = cv2.Sobel(img,cv2.CV_64F, d1, d2, ksize=kernel)
        else:
            edge_im[:, :, 0] = cv2.Sobel(img[:, :, 0],cv2.CV_64F, d1, d2, ksize=kernel)
            edge_im[:, :, 1] = cv2.Sobel(img[:, :, 1],cv2.CV_64F, d1, d2, ksize=kernel)
            edge_im[:, :, 2] = cv2.Sobel(img[:, :, 2],cv2.CV_64F, d1, d2, ksize=kernel)
            cv2.normalize(edge_im, edge_im, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            edge_im = edge_im.astype(np.uint8)
        return edge_im

    def laplacian_edge(self, img, kernel):      
        edge_im = np.zeros(img.shape, np.float32)
        edge_im = cv2.GaussianBlur(edge_im,(3,3),0)
            
        if len(img.shape) == 2:
           edge_im = cv2.Laplacian(img, cv2.CV_64F, ksize=kernel)
        else:
            edge_im[:, :, 0] = cv2.Laplacian(img[:, :, 0], cv2.CV_64F, ksize=kernel)
            edge_im[:, :, 1] = cv2.Laplacian(img[:, :, 1], cv2.CV_64F, ksize=kernel)
            edge_im[:, :, 2] = cv2.Laplacian(img[:, :, 2], cv2.CV_64F, ksize=kernel)
            cv2.normalize(edge_im, edge_im, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            edge_im = edge_im.astype(np.uint8)
        return edge_im

    def choose_edge(self,img):        
        strCB = self.cboEdge.currentText()
        minVal = float(self.leMinVal.text())
        maxVal = float(self.leMaxVal.text())
        kernel = int(self.leKernel.text())
        noisy = self.choose_noise(img)
        
        if strCB == 'Canny':  
            self.leKernel.setEnabled(False)
            self.sbKernel.setEnabled(False)
            self.leMinVal.setEnabled(True)
            self.sbMinVal.setEnabled(True)
            self.leMaxVal.setEnabled(True)
            self.sbMaxVal.setEnabled(True)
            edge_im = self.canny_edge(noisy, minVal, maxVal)
            return edge_im
        if strCB == 'Sobel X':  
            self.leKernel.setEnabled(True)
            self.sbKernel.setEnabled(True)
            self.leMinVal.setEnabled(False)
            self.sbMinVal.setEnabled(False)
            self.leMaxVal.setEnabled(False)
            self.sbMaxVal.setEnabled(False)
            edge_im = self.sobelxy_edge(noisy, 1, 0, kernel)
            return edge_im
        if strCB == 'Sobel Y': 
            self.leKernel.setEnabled(True)
            self.sbKernel.setEnabled(True)
            self.leMinVal.setEnabled(False)
            self.sbMinVal.setEnabled(False)
            self.leMaxVal.setEnabled(False)
            self.sbMaxVal.setEnabled(False)
            edge_im = self.sobelxy_edge(noisy, 0, 1, kernel)
            return edge_im
        if strCB == 'Laplacian':  
            self.leKernel.setEnabled(True)
            self.sbKernel.setEnabled(True)
            self.leMinVal.setEnabled(False)
            self.sbMinVal.setEnabled(False)
            self.leMaxVal.setEnabled(False)
            self.sbMaxVal.setEnabled(False)
            edge_im = self.laplacian_edge(noisy, kernel)
            return edge_im   

        if strCB == 'Erode':  
            self.leKernel.setEnabled(True)
            self.sbKernel.setEnabled(True)
            self.leMinVal.setEnabled(False)
            self.sbMinVal.setEnabled(False)
            self.leMaxVal.setEnabled(False)
            self.sbMaxVal.setEnabled(False)
            edge_im = self.erode_edge(noisy, kernel)
            return edge_im

        if strCB == 'Dilate':  
            self.leKernel.setEnabled(True)
            self.sbKernel.setEnabled(True)
            self.leMinVal.setEnabled(False)
            self.sbMinVal.setEnabled(False)
            self.leMaxVal.setEnabled(False)
            self.sbMaxVal.setEnabled(False)
            edge_im = self.dilate_edge(noisy, kernel)
            return edge_im

        if strCB == 'Opening':  
            self.leKernel.setEnabled(True)
            self.sbKernel.setEnabled(True)
            self.leMinVal.setEnabled(False)
            self.sbMinVal.setEnabled(False)
            self.leMaxVal.setEnabled(False)
            self.sbMaxVal.setEnabled(False)
            edge_im = self.morph_op(noisy, kernel, cv2.MORPH_OPEN)
            return edge_im

        if strCB == 'Closing':  
            self.leKernel.setEnabled(True)
            self.sbKernel.setEnabled(True)
            self.leMinVal.setEnabled(False)
            self.sbMinVal.setEnabled(False)
            self.leMaxVal.setEnabled(False)
            self.sbMaxVal.setEnabled(False)
            edge_im = self.morph_op(noisy, kernel, cv2.MORPH_CLOSE)
            return edge_im

        if strCB == 'Morphological Gradient':  
            self.leKernel.setEnabled(True)
            self.sbKernel.setEnabled(True)
            self.leMinVal.setEnabled(False)
            self.sbMinVal.setEnabled(False)
            self.leMaxVal.setEnabled(False)
            self.sbMaxVal.setEnabled(False)
            edge_im = self.morph_op(noisy, kernel, cv2.MORPH_GRADIENT)
            return edge_im

        if strCB == 'Top Hat':  
            self.leKernel.setEnabled(True)
            self.sbKernel.setEnabled(True)
            self.leMinVal.setEnabled(False)
            self.sbMinVal.setEnabled(False)
            self.leMaxVal.setEnabled(False)
            self.sbMaxVal.setEnabled(False)
            edge_im = self.morph_op(noisy, kernel, cv2.MORPH_TOPHAT)
            return edge_im
        
        if strCB == 'Black Hat':  
            self.leKernel.setEnabled(True)
            self.sbKernel.setEnabled(True)
            self.leMinVal.setEnabled(False)
            self.sbMinVal.setEnabled(False)
            self.leMaxVal.setEnabled(False)
            self.sbMaxVal.setEnabled(False)
            edge_im = self.morph_op(noisy, kernel, cv2.MORPH_BLACKHAT)
            return edge_im
        
    def set_minval(self):
        self.leMinVal.setText(str(self.sbMinVal.value()))  
        self.do_edge()
        
    def set_maxval(self):
        self.leMaxVal.setText(str(self.sbMaxVal.value())) 
        self.do_edge()

    def set_kernel(self):
        self.leKernel.setText(str(self.sbKernel.value())) 
        self.do_edge()            

    def do_edge(self):
        edges = self.choose_edge(img)        
        height, width, channel = edges.shape
        bytesPerLine = 3 * width  
        cv2.cvtColor(edges, cv2.COLOR_BGR2RGB, edges)
        qImg = QImage(edges, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.display_image(pixmap, edges, self.labelFilter, self.widgetHistIm, 'Histogram of Edge Detection')

    def erode_edge(self, img, kernel):      
        edge_im = np.zeros(img.shape, np.float32)
        edge_im = cv2.GaussianBlur(edge_im,(3,3),0)
        structure_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel,kernel))
        #structure_kernel = np.ones((kernel,kernel),np.uint8)
        
        if len(img.shape) == 2:
           edge_im = cv2.erode(img,structure_kernel, iterations = 1)
        else:
            edge_im[:, :, 0] = cv2.erode(img[:, :, 0],structure_kernel, iterations = 1)
            edge_im[:, :, 1] = cv2.erode(img[:, :, 1],structure_kernel, iterations = 1)
            edge_im[:, :, 2] = cv2.erode(img[:, :, 2],structure_kernel, iterations = 1)
            cv2.normalize(edge_im, edge_im, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            edge_im = edge_im.astype(np.uint8)
        return edge_im

    def dilate_edge(self, img, kernel):      
        edge_im = np.zeros(img.shape, np.float32)
        edge_im = cv2.GaussianBlur(edge_im,(3,3),0)
        structure_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel,kernel))
        
        if len(img.shape) == 2:
           edge_im = cv2.dilate(img,structure_kernel, iterations = 1)
        else:
            edge_im[:, :, 0] = cv2.dilate(img[:, :, 0],structure_kernel, iterations = 1)
            edge_im[:, :, 1] = cv2.dilate(img[:, :, 1],structure_kernel, iterations = 1)
            edge_im[:, :, 2] = cv2.dilate(img[:, :, 2],structure_kernel, iterations = 1)
            cv2.normalize(edge_im, edge_im, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            edge_im = edge_im.astype(np.uint8)
        return edge_im

    def morph_op(self, img, kernel, operation_val):      
        edge_im = np.zeros(img.shape, np.float32)
        edge_im = cv2.GaussianBlur(edge_im,(3,3),0)
        structure_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel,kernel))
        
        if len(img.shape) == 2:
           edge_im = cv2.morphologyEx(img,operation_val,structure_kernel)
        else:
            edge_im[:, :, 0] = cv2.morphologyEx(img[:, :, 0],operation_val,structure_kernel)
            edge_im[:, :, 1] = cv2.morphologyEx(img[:, :, 1],operation_val,structure_kernel)
            edge_im[:, :, 2] = cv2.morphologyEx(img[:, :, 2],operation_val,structure_kernel)
            cv2.normalize(edge_im, edge_im, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            edge_im = edge_im.astype(np.uint8)
        return edge_im
        
if __name__=="__main__":
    import sys
    app = QApplication(sys.argv)    
    form = ImageProcessing()
    form.show()
    sys.exit(app.exec_())