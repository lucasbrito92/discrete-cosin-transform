from __future__ import division
from math import sqrt, cos, pi, pow
from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct
import os
import copy
import time
from datetime import timedelta

path="/Users/lucas/Documents/Mestrado/Processamento de Imagens/DCT/baboon_faces"
os.chdir(path)

# class for hold row, col and value for code matrix
class node:

    def __init__(self, row, col, value):
        self.row = row
        self.col = col
        self.value = value

class discreteCosinTransform(object):

    def __init__(self, filename, factor, block_size):
        self.filename = filename
        self.slice_factor = factor
        self.dim = block_size

        self.image = self.load_image(self.filename)
        self.image_size = self.image.shape[0]

    # load image from path
    def load_image(self, filename):
        img = Image.open(filename)
        img.load()
        
        data = np.asarray(img, dtype="int32")
        return data

    # code matrix in zig zag reading mode
    def zig_zag_code(self, image):
        
        n = image.shape[0]
        size = n-1
        dct_coef_vector = []

        _node = node(0, 0, 0)

        # inserting 0,0 matrix node
        dct_coef_vector.append(node(0, 0, image[0][0]))
        j = 0

        # ERRO: esta copiando i+1 na posicao i

        # k is the row counter
        for k in range(1, 2*size):
            if k < n:
                i = 0
                balance = 1
                
                if k % 2 == 0:
                    terminate_counter = k

                    _node.value = image[k][i]
                    _node.row = k
                    _node.col = i
                    dct_coef_vector.append(copy.deepcopy(_node))

                    while terminate_counter != 0:
                        _node.value = image[k-balance][i+balance]
                        _node.row = k-balance
                        _node.col = i+balance
                        dct_coef_vector.append(copy.deepcopy(_node))
                        terminate_counter -= 1
                        balance += 1
                else:
                    terminate_counter = 0

                    _node.value = image[i][k]
                    _node.row = i
                    _node.col = k
                    dct_coef_vector.append(copy.deepcopy(_node))

                    while terminate_counter != k:
                        _node.value = image[i+balance][k-balance]
                        _node.row = i+balance
                        _node.col = k-balance
                        dct_coef_vector.append(copy.deepcopy(_node))
                        terminate_counter += 1
                        balance += 1
            else:
                j += 1
                balance = 1

                if j % 2 == 0:
                    terminate_counter = size

                    _node.value = image[size][j]
                    _node.row = size
                    _node.col = j
                    dct_coef_vector.append(copy.deepcopy(_node))

                    while terminate_counter != j:
                        _node.value = image[size-balance][j+balance]
                        _node.row = size-balance
                        _node.col = j+balance
                        dct_coef_vector.append(copy.deepcopy(_node))
                        terminate_counter -= 1
                        balance += 1

                else:
                    terminate_counter = j

                    _node.value = image[j][size]
                    _node.row = j
                    _node.col = size
                    dct_coef_vector.append(copy.deepcopy(_node))

                    while terminate_counter != size:
                        _node.value = image[j+balance][n-balance]
                        _node.row = j+balance
                        _node.col = size-balance
                        dct_coef_vector.append(copy.deepcopy(_node))
                        terminate_counter += 1
                        balance += 1
        
        dct_coef_vector.append(node(size, size, image[size][size]))

        return dct_coef_vector

    # uncode matrix in diagonal reading mode
    def zig_zag_uncode(self, flat_image):
        
        # block dimension
        n = len(flat_image)
        uncoded_matrix = np.empty(shape=(self.dim, self.dim))

        for i in range(n):
            uncoded_matrix[flat_image[i].row][flat_image[i].col] = flat_image[i].value

        return uncoded_matrix

    # adapt matrix pixels (-128)
    def pixel_adapt(self, img):

        new = np.zeros_like(img)
        n = new.shape[0]

        for i in range(n):
            for j in range(n):
                new[i][j] = img[i][j] - 128

        return new

    # adapt matrix pixels (+128)
    def pixel_readapt(self, img):

        new = np.zeros_like(img)
        n = new.shape[0]

        for i in range(n):
            for j in range(n):
                new[i][j] = img[i][j] + 128

        return new

    # generate a T matrix for computations
    def generate_Tmatrix(self, image_size):
        
        n = image_size
        t = np.empty(shape=(n, n))

        for i in range(n):
            for j in range(n):
                if i == 0:
                    t[i][j] = 1/sqrt(n)
                else:
                    a = 2*n
                    b = sqrt(2/n)
                    c = 2*j+1
                    d = c * i
                    e = d * pi
                    t[i][j] = b*cos(e/a)

        return t

    # locates all possible (and excludent) dim sized blocks for computate dct matrix and quatization the results
    def run_dct_over_blocks(self, image):

        _image = np.zeros_like(image)
        n = self.image_size
        new_image = []

        for i in range(0, n+self.dim, self.dim):
            for j in range(0, n+self.dim, self.dim):
                for k in range(0, n+self.dim, self.dim):
                    for l in range(0, n+self.dim, self.dim):
                        if ((i != j) and (k != l) and((j-i) == self.dim)
                                and ((l-k) == self.dim)):
                            new_image = image[i:j, k:l].copy()
                            _image[i:j, k:l] = self.discrete_cosin_transform_over_matrix(new_image).copy()

        return _image

    # locates all possible (and excludent) dim sized blocks for computate idct matrix
    def inverse_run_dct_over_blocks(self, image):

        _image = np.zeros_like(image)
        n = self.image_size
        new_image = []

        for i in range(0, n+self.dim, self.dim):
            for j in range(0, n+self.dim, self.dim):
                for k in range(0, n+self.dim, self.dim):
                    for l in range(0, n+self.dim, self.dim):
                        if ((i != j) and (k != l) and((j-i) == self.dim)
                                and ((l-k) == self.dim)):
                            new_image = image[i:j, k:l].copy()
                            _image[i:j, k:l] = self.inverse_discrete_cosin_transform_over_matrix(new_image).copy()

        return _image

    # locates all possible (and excludent) dim sized blocks and do zig zag code
    def run_zigzag_over_blocks(self, image):

        hifi = np.zeros_like(image)
        lowfi = np.zeros_like(image)
        n = self.image_size
        new_image = []

        for i in range(0, n+self.dim, self.dim):
            for j in range(0, n+self.dim, self.dim):
                for k in range(0, n+self.dim, self.dim):
                    for l in range(0, n+self.dim, self.dim):
                        if ((i != j) and (k != l) and((j-i) == self.dim)
                                and ((l-k) == self.dim)):
                            new_image = image[i:j, k:l].copy()
                            _hifi, _lowfi = self.slice_flats(self.zig_zag_code(new_image))
                            hifi[i:j, k:l] = self.zig_zag_uncode(_hifi)
                            lowfi[i:j, k:l] = self.zig_zag_uncode(_lowfi)


        return hifi, lowfi

    # calculate DCT matrix based on Matrix Computations
    def discrete_cosin_transform_over_matrix(self, image):

        T = self.generate_Tmatrix(image.shape[0])
        _M = self.pixel_adapt(image)
        C = np.matmul(T, _M)
        
        dct_matrix = np.matmul(C, np.transpose(T))
        return dct_matrix

    # calculate IDCT matrix based on Matrix Computations
    def inverse_discrete_cosin_transform_over_matrix(self, image):

        # precalculate cosins
        T = self.generate_Tmatrix(image.shape[0])
        C = np.matmul(np.transpose(image), T)
        M = np.matmul(np.transpose(C), T)
        output_matrix = self.pixel_readapt(M)

        return output_matrix

    # returns low frequencies (lowfi) and high frequencies (hifi) sliced flat coded images
    def slice_flats(self, flat_image):
        
        lowfi, hifi = [], []
        m = len(flat_image)-1

        for i in range(m):
            if i < self.slice_factor:
                lowfi.append(copy.deepcopy(flat_image[i]))
                hifi.append(copy.deepcopy(flat_image[i]))
                hifi[i].value = 0
            else:
                hifi.append(copy.deepcopy(flat_image[i]))
                lowfi.append(copy.deepcopy(flat_image[i]))
                lowfi[i].value = 0

        return hifi,lowfi

def main():

    dct = discreteCosinTransform('baboon_128x128px.pgm', 25, 8)
    dct_image = dct.run_dct_over_blocks(dct.image)

    hifi, lowfi = dct.run_zigzag_over_blocks(dct_image)
    dct_hifi = dct.inverse_run_dct_over_blocks(hifi)
    dct_lowfi = dct.inverse_run_dct_over_blocks(lowfi)

    hifreq = Image.fromarray(dct_hifi)
    hifreq.show()

    lowfreq = Image.fromarray(dct_lowfi)
    lowfreq.show()

if __name__ == "__main__":
    main()