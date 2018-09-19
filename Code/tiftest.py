from PIL import Image
import numpy
im = Image.open('data_files/SRTM_data_Norway_1.tif')

imarray = numpy.array(im)
print('dataset-shape')
print(imarray.shape)
print(imarray)
