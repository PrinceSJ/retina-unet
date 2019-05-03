import numpy as np
import random
import configparser

from help_functions import load_hdf5
from help_functions import visualize
from help_functions import group_images

from skimage.util.shape import view_as_blocks
from skimage.util.shape import view_as_windows
from skimage.util import pad

from pre_processing import my_PreProc


#To select the same images
# random.seed(10)

#Load the original data and return the extracted patches for training/testing
def get_data_training(imgs,
                      groundTruths,
                      patch_size,
                      stride_size,
                      inside_FOV):
    train_img = imgs/255.
    train_gt = groundTruths/255.

    data_consistency_check(train_img, train_gt)

    #check gt are within 0-1
    assert(np.min(train_gt)==0 and np.max(train_gt)==1)

    print("\ntrain images/gt shape:")
    print(train_img.shape)
    print("train images range (min-max): " + str(np.min(train_img)) + ' - ' + str(np.max(train_img)))
    print("train gt are within 0-1\n")

    #extract the TRAINING patches from the full images
    # patches_imgs_train, patches_gt_train = extract_random(train_img, train_gt, patch_height, patch_width, N_subimgs * train_img.shape[0], inside_FOV)

    # ordered_overlap with stride of 5 is almost the same as 9500 random selected
    patches_imgs_train = extract_ordered_overlap(train_img, patch_size, stride_size)
    patches_gt_train = extract_ordered_overlap(train_gt, patch_size, stride_size)
    data_consistency_check(patches_imgs_train, patches_gt_train)

    print("\ntrain PATCHES images/gt shape:")
    print(patches_imgs_train.shape)
    print("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_gt_train


#Load the original data and return the extracted patches for training/testing
def get_data_testing(test_imgs_original, test_groundTruths, Imgs_to_test, patch_size):

    test_imgs = test_imgs_original/255.
    test_groundTruths = test_groundTruths/255.

    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[:Imgs_to_test]
    test_groundTruths = test_groundTruths[:Imgs_to_test]

    data_consistency_check(test_imgs, test_groundTruths)

    #check masks are within 0-1
    assert(np.max(test_groundTruths)==1  and np.min(test_groundTruths)==0)

    print("\ntest images/masks shape:")
    print(test_imgs.shape)
    print("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered(test_imgs, (patch_height, patch_width))
    patches_masks_test = extract_ordered(test_groundTruths, (patch_height, patch_width))
    data_consistency_check(patches_imgs_test, patches_masks_test)

    print("\ntest PATCHES images/masks shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, patches_masks_test




# Load the original data and return the extracted patches for testing
# return the ground truth in its original shape
def get_data_testing_overlap(test_imgs_original, test_groundTruths, Imgs_to_test, patch_size, stride_size):
    test_imgs = test_imgs_original/255.
    test_groundTruths = test_groundTruths/255.
    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[:Imgs_to_test]
    test_groundTruths = test_groundTruths[:Imgs_to_test]

    #check masks are within 0-1
    assert(np.max(test_groundTruths)==1  and np.min(test_groundTruths)==0)

    print("\ntest images shape:")
    print(test_imgs.shape)
    print("\ntest mask shape:")
    print(test_groundTruths.shape)
    print("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs,          (patch_height, patch_width), (stride_height, stride_width))
    patches_masks_test = extract_ordered_overlap(test_groundTruths, (patch_height, patch_width), (stride_height, stride_width))

    print("\ntest PATCHES images shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " + str(np.min(patches_imgs_test)) + ' - ' + str(np.max(patches_imgs_test)))

    return patches_imgs_test, patches_masks_test


#data consinstency check
def data_consistency_check(imgs,masks):
    assert(len(imgs.shape)==len(masks.shape))
    assert(imgs.shape[0]==masks.shape[0])
    assert(imgs.shape[2]==masks.shape[2])
    assert(imgs.shape[3]==masks.shape[3])
    assert(masks.shape[1]==1)
    assert(imgs.shape[1]==1 or imgs.shape[1]==3)


#extract patches randomly in the full training images
#  -- Inside OR in full image
def extract_random(full_imgs, full_masks, patch_h,patch_w, N_patches, inside=True):
    if (N_patches % full_imgs.shape[0] != 0):
        print("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[1]==1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])

    # init arrays
    patches       = np.empty((N_patches, full_imgs.shape[1],  patch_h, patch_w))
    patches_masks = np.empty((N_patches, full_masks.shape[1], patch_h, patch_w))
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image

    # (0,0) in the center of the image
    patch_per_img = int(N_patches/full_imgs.shape[0])  #N_patches equally divided in the full images
    print("patches per full image: " +str(patch_per_img))

    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for k in range(patch_per_img):
            x_center = random.randint(0 + int(patch_w/2), img_w - int(patch_w/2))
            y_center = random.randint(0 + int(patch_h/2), img_h - int(patch_h/2))
            #check whether the patch is fully contained in the FOV
            if inside and not is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h):
                continue
            bounds_y = [y_center - int(patch_h/2), y_center + int(patch_h/2)]
            bounds_x = [x_center - int(patch_w/2), x_center + int(patch_w/2)]

            patch      = full_imgs[i, :, bounds_y[0]: bounds_y[1], bounds_x[0]: bounds_x[1]]
            patch_mask = full_masks[i,:, bounds_y[0]: bounds_y[1], bounds_x[0]: bounds_x[1]]

            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask
            iter_tot += 1   #total
    return patches, patches_masks


#check if the patch is fully contained in the FOV
def is_patch_inside_FOV(x,y,img_w,img_h,patch_h):
    x_ = x - int(img_w/2) # origin (0,0) shifted to image center
    y_ = y - int(img_h/2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0) #radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_*x_)+(y_*y_))
    return radius < R_inside


#Divide all the full_imgs in pacthes
def extract_ordered(full_imgs, patch_size):
    return _extract_patches(full_imgs, patch_size, (0, 0), False)

def extract_ordered_overlap(full_imgs, patch_size, stride_size):
    return _extract_patches(full_imgs, patch_size, stride_size, True)

# patch_size and stride_size => h x w
def _extract_patches(full_imgs, patch_size, stride_size, overlap):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3

    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image

    # prepare padding
    img_shape = np.array(full_imgs[0,0].shape)
    patch_size = np.array(patch_size)
    if overlap:
        residual = (img_shape - patch_size) / stride_size
        N_patches = np.ceil((img_shape - patch_size) // stride_size + 1).astype(np.int)
        pad_by = np.array(stride_size) - residual
    else:
        residual = img_shape % patch_size
        N_patches = np.ceil(img_shape / patch_size).astype(np.int)
        pad_by = patch_size - residual

    needs_padding = pad_by[0] > 0 or pad_by[1] > 0
    # can be padded only on one side because border is 0 anyway
    if needs_padding:
        pad_by = ((0, 0), (0, pad_by[0]), (0, pad_by[1])) # only pad img dont add another channel
    patches_per_img = N_patches[0] * N_patches[1]
    print("number of patches per image: " + str(patches_per_img))
    N_patches_tot = patches_per_img * full_imgs.shape[0]
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_size[0], patch_size[1]))

    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        img = full_imgs[i]
        # pad if needed
        if needs_padding:
            img = pad(img, pad_by, 'constant', constant_values=0)
        
        if not overlap:
            patches_of_img = view_as_blocks(img, patch_size).reshape(patches_per_img, 1, patch_size[0], patch_size[1])
        else:
            patches_of_img = view_as_windows(img[0], patch_size, stride_size)
        patches[i * patches_per_img : (i + 1) * patches_per_img] = patches_of_img.reshape(N_patches_tot, 1, patch_size[0], patch_size[1])
        iter_tot += patches_per_img
    assert (iter_tot == N_patches_tot)
    return patches  #array with all the full_imgs divided in patches

def get_padding_values(img_shape, divider):
    

    return N_patches, pad_by

def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape)==4)  #4D arrays
    assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: " +str(N_patches_h))
    print("N_patches_w: " +str(N_patches_w))
    print("N_patches_img: " +str(N_patches_img))

    assert (preds.shape[0] % N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print("According to the dimension inserted, there are " + str(N_full_imgs) + " full images (of " + str(img_h) + "x" + str(img_w) + " each)")
    full_prob = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))  #itialize to zero mega array with sum of Probabilities
    full_sum  = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:, h * stride_h: h * stride_h + patch_h, w * stride_w: w * stride_w + patch_w] += preds[k]
                full_sum[i, :, h * stride_h: h * stride_h + patch_h, w * stride_w: w * stride_w + patch_w] += 1
                k += 1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    print(final_avg.shape)
    assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0
    return final_avg


#Recompone the full images with the patches
def recompone(data, N_h, N_w):
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    assert(len(data.shape)==4)
    N_pacth_per_img = N_w * N_h
    assert(data.shape[0] % N_pacth_per_img == 0)
    N_full_imgs = data.shape[0]/N_pacth_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    N_pacth_per_img = N_w * N_h
    #define and start full recompone
    full_recomp = np.empty((N_full_imgs,data.shape[1], N_h * patch_h, N_w * patch_w))
    k = 0  #iter full img
    s = 0  #iter single patch
    while (s<data.shape[0]):
        #recompone one:
        single_recon = np.empty((data.shape[1], N_h * patch_h, N_w * patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:, h * patch_h: h * (patch_h + 1), w * patch_w: w * (patch_w + 1)] = data[s]
                s += 1
        full_recomp[k] = single_recon
        k += 1
    assert (k == N_full_imgs)
    return full_recomp


#return only the pixels contained in the FOV, for both images and masks
def pred_only_FOV(data_imgs,data_masks,original_imgs_border_masks):
    assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)  #4D arrays
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[2]==data_masks.shape[2])
    assert (data_imgs.shape[3]==data_masks.shape[3])
    assert (data_imgs.shape[1]==1 and data_masks.shape[1]==1)  #check the channel is 1
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,x,y,original_imgs_border_masks)==True:
                    new_pred_imgs.append(data_imgs[i,:,y,x])
                    new_pred_masks.append(data_masks[i,:,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks

#function to set to black everything outside the FOV, in a full image
def kill_border(data):
    assert (len(data.shape)==4)  #4D arrays
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if is_inside_FOV(x, y, width, height)==False:
                    data[i,:,y,x]=0.0


def is_inside_FOV(x,y,img_w,img_h):
    x_ = x - int(img_w/2) # origin (0,0) shifted to image center
    y_ = y - int(img_h/2)  # origin (0,0) shifted to image center
    R_inside = 270
    radius = np.sqrt((x_*x_)+(y_*y_))
    if radius < R_inside:
        return True
    else:
        return False
