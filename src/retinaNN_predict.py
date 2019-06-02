###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
import numpy as np
import configparser
from matplotlib import pyplot as plt

#Keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
import tensorflow as tf
import tensorflow.keras.backend as K

import sys
sys.path.insert(0, './lib/')
from help_functions import *
from loader import load_testset
from extract_patches import recompone
from extract_patches import recompone_overlap
from nn_utils import *
from unet import *
from resnet import *

session = K.get_session()

#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('configuration.txt')

path_data = config.get('data paths', 'path_local')
test_data_path = config.get('data paths', 'test_data_path')
test_data_stats = config.get('data paths', 'test_data_stats')

stats_config = configparser.RawConfigParser()
stats_config.read(test_data_stats)
full_img_height = int(stats_config.get('statistics', 'new_image_height'))
full_img_width = int(stats_config.get('statistics', 'new_image_width'))

# dimension of the patches
patch_size = (int(config.get('data attributes', 'patch_height')), int(config.get('data attributes', 'patch_width')))

#the stride in case output with average
stride_size = (int(config.get('testing settings', 'stride_height')), int(config.get('testing settings', 'stride_width')))
assert (stride_size[0] < patch_size[0] and stride_size[1] < patch_size[1])

#model name
name_experiment = config.get('experiment', 'name')
arch = config.get('experiment', 'arch')
testset = config.get('experiment', 'testset')
experiment_path = path_data + '/' + name_experiment + '_' + arch
save_path = experiment_path + '/' + testset

u_net = arch == 'unet'

#N full images to be predicted
imgs_to_visualize = int(config.get('testing settings', 'imgs_to_visualize'))
N_subimgs = int(config.get('testing settings', 'N_subimgs'))
patches_per_img = int(stats_config.get('statistics', 'subimages_per_image'))

#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
batch_size = int(config.get('training settings', 'batch_size'))


#================= Load the data =====================================================
dataset = load_testset(test_data_path, batch_size)
iterator = dataset.make_one_shot_iterator()

# n_samples = int(patches_per_img * imgs_to_visualize)
# batches = int(np.ceil(n_samples / batch_size))
# patches_embedding = np.zeros((batches * batch_size, 1, patch_size[0], patch_size[1]))
# patches_embedding_gt = np.zeros((batches * batch_size, 1, patch_size[0], patch_size[1]))

# batch_img, batch_gt = iterator.get_next()
# print('loading visualization data')
# for i in range(batches):
#     if i % 50 == 0:
#         print(str(i) + ' / ' + str(batches))
#     batch_gt_np = tf.reshape(batch_gt[:, 1], (batch_size, 1, patch_size[0], patch_size[1]))
#     batch_img_np, batch_gt_np = session.run([batch_img, batch_gt_np])
#     patches_embedding[i * batch_size: i * batch_size + batch_size] = batch_img_np
#     patches_embedding_gt[i * batch_size: i * batch_size + batch_size] = batch_gt_np

# patches_embedding = patches_embedding[:n_samples]
# patches_embedding_gt = patches_gts_samples[:n_samples]


# orig_imgs = recompone_overlap(
#     patches_embedding,
#     full_img_height,
#     full_img_width,
#     stride_size[0],
#     stride_size[1]
# ) * 255
# gtruth_masks = recompone_overlap(
#     patches_embedding_gt,
#     full_img_height,
#     full_img_width,
#     stride_size[0],
#     stride_size[1]
# ) * 255

#================ Run the prediction of the patches ==================================
best_last = config.get('testing settings', 'best_last')

#Load the saved model
if u_net:
    model = get_unet(1, batch_size, patch_size[0], patch_size[1], False)  #the U-net model
else:
    model = UResNet34(input_shape=(1, patch_size[0], patch_size[1]))

thresholds = np.linspace(0, 1, 200).tolist()
model.compile(
    optimizer = 'sgd',
    loss = weighted_cross_entropy(9),
    metrics = [
        BinaryAccuracy(),
        TruePositives(thresholds = thresholds),
        FalsePositives(thresholds = thresholds),
        TrueNegatives(thresholds = thresholds),
        FalseNegatives(thresholds = thresholds) # confusion
    ]
)
model.load_weights(experiment_path + '/' + name_experiment + '_' + best_last + '_weights.h5')

print("start prediction")
#Calculate the predictions
samples_to_predict = np.ceil(patches_per_img * imgs_to_visualize / batch_size) * batch_size
predictions = model.predict(
    dataset.take(samples_to_predict),
    batch_size = batch_size,
    steps = int(samples_to_predict / batch_size)
)

predictions = predictions[:patches_per_img * imgs_to_visualize]

print("predicted images size :")
print(predictions.shape)

#===== Convert the prediction arrays in corresponding images

pred_patches = pred_to_imgs(predictions, patch_size[0], patch_size[1], "original")
print(np.max(pred_patches))
print(np.min(pred_patches))

# #========== Elaborate and visualize the predicted images ====================
pred_imgs = recompone_overlap(
    pred_patches,
    full_img_height,
    full_img_width,
    stride_size[0],
    stride_size[1]
) * 255

assert(np.max(pred_imgs) <= 255)
assert(np.min(pred_imgs) >= 0)

# print("Orig imgs shape: " +str(orig_imgs.shape))
print("pred imgs shape: " +str(pred_imgs.shape))
# print("Gtruth imgs shape: " +str(gtruth_masks.shape))
# visualize(group_images(orig_imgs, N_visual), save_path + "_all_originals")#.show()
visualize(group_images(pred_imgs, N_visual), save_path + "_all_predictions")#.show()

# visualize(group_images(gtruth_masks,N_visual), save_path + "_all_groundTruths")#.show()
#visualize results comparing mask and prediction:
# assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
# N_predicted = orig_imgs.shape[0]
# group = N_visual
# assert (N_predicted%group==0)
# for i in range(int(N_predicted/group)):
#     fr = i * group
#     to = i * group + group
#     orig_stripe =  group_images(orig_imgs[fr: to], group)
#     masks_stripe = group_images(gtruth_masks[fr: to], group)
#     pred_stripe =  group_images(pred_imgs[fr: to], group)
#     total_img = np.concatenate((orig_stripe, masks_stripe, pred_stripe), axis=0)
#     visualize(total_img, save_path + "_Original_GroundTruth_Prediction" + str(i))#.show()

#========================== Evaluate the results ===================================
print("\n\n========  Evaluate the results =======================")

# sensitivities,
# specificities,
eval_values = model.evaluate(
    dataset,
    batch_size = batch_size,
    steps = int(N_subimgs / batch_size),
    verbose = 1
)

print(eval_values)
loss, true_positives, false_positives, true_negatives, false_negatives = eval_values

# Area under the ROC curve
tpr = true_positives / (true_positives + false_negatives)
fpr = false_positives / (false_positives + true_negatives)
roc_curve = plt.figure()
roc_curve.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % auc )
roc_curve.title('ROC curve')
roc_curve.xlabel("FPR (False Positive Rate)")
roc_curve.ylabel("TPR (True Positive Rate)")
roc_curve.legend(loc = "lower right")
roc_curve.savefig(save_path + "_ROC.png")

# Precision-recall curve
# print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
precision = true_positives / (true_positives + false_positives)
prec_rec_curve = plt.figure()
prec_rec_curve.plot(tpr, precision, '-', label = 'Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
prec_rec_curve.title('Precision - Recall curve')
prec_rec_curve.xlabel("Recall")
prec_rec_curve.ylabel("Precision")
prec_rec_curve.legend(loc = "lower right")
prec_rec_curve.savefig(save_path + "_Precision_recall.png")

# Confusion matrix
confusion = np.array([[true_positives[99], false_positives[99]], [true_negatives[99], false_negatives[99]]])
print(confusion)

if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print("Global Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print("Precision: " +str(precision))

#Save the results
with open(path_experiment+'performances.txt', 'w') as file:
    file.write(
        "Confusion matrix:"
        + str(confusion)
        + "\nACCURACY: " + str(accuracy)
        + "\nSENSITIVITY: " + str(sensitivity)
        + "\nSPECIFICITY: " + str(specificity)
        + "\nPRECISION: " + str(precision)
    )
    file.close()
