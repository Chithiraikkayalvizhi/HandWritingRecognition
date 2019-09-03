import convertInkmlToImg as convertor
import matplotlib.pylab as plt
import scipy.ndimage as ndimage
import inkmlParser as parser
import os

rootdir = 'C:/github/Kayal_Capstone/HandWritingRecognition/Data/Raw/2016/symbols/train/good'
#filename = 'iso1.inkml'
range_files = 100
image_list = []
goundtruth_list = []

for filename in os.listdir(rootdir)[:range_files]:
    traces, truth, *rest = parser.parse_inkml(rootdir + '/' + filename)
    selected_tr = convertor.get_traces_data(traces)
    im = convertor.convert_to_imgs(selected_tr, 50)
    im = ndimage.gaussian_filter(im, sigma=(.5, .5), order=0)
    image_list.append(im)
    goundtruth_list.append(truth)

fig = plt.figure()

box_size = 10
for i in range(box_size * box_size):
    ax = fig.add_subplot(box_size,box_size,i+1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(goundtruth_list[i])
    plt.imshow(image_list[i], interpolation='nearest',)

plt.subplots_adjust(hspace=0.63)
plt.show()

print(traces)