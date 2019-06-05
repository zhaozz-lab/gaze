from scipy.io import loadmat
from matplotlib import pyplot as plt
file_dir='/home/leo/Desktop/Dataset/MPIIGaze/Data/Normalized/p00/day01.mat'
f=loadmat(file_dir)
left=f['data'][0]['left'][0]['image'][0][0]
right=f['data'][0]['right'][0]['image'][0][0]
i=160
l_img=left[i]
r_img=right[i]
f,axs=plt.subplots(nrows=1,ncols=2,sharex=True)
axs[0].imshow(r_img)
axs[0].set_title('right eye')
axs[1].imshow(l_img)
axs[1].set_title('left eye')
plt.show()