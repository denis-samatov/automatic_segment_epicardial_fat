import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

from typing import List, Tuple
from algorithm.BasicOperationsForImageProcessing import convert_image_to_rgb, segmentation, add_images
from algorithm.ConvertToHUAndSortedData import load_scan, get_pixels_hu
from algorithm.CountVolume import volume
from algorithm.ThresholdSegmentationByHU import threshold_segmentation
from algorithm.LoadingDICOMData import create_directory_dict
# from algorithm.CountRadiomicsFeatures import calculate_radiomics_features


def rgb_to_binary(filename):
    rgb = cv.imread(filename)
    bw_img = rgb[:,:,0] > 0
    return bw_img

def segment_epicardial_fat(DICOM_DATASET: List[str], count_radiomics_features: bool = False, count_volume: bool = True, OUTPUT_FOLDER: str = 'result_folder/'):

    # print(DICOM_DATASET)
    # patient_dir = DICOM_DATASET.split(os.path.sep)[-1]
    result_dicom_pacient_paths = create_directory_dict(DICOM_DATASET)
    patient_dir = str(list(result_dicom_pacient_paths.keys())[0])
    # print(len(*result_dicom_pacient_paths.values()))
    'Select the patient dataset'
    patient = load_scan(*result_dicom_pacient_paths.values())


    'Thickness and pixel spacing'
    thickness = patient[0].SliceThickness
    px_spacing = patient[0].PixelSpacing[0]
    patient_id = patient[0].PatientID
    number_of_slices = len(patient)


    'Get image in HU scale'
    patients_hu = get_pixels_hu(patient)

    'Constants'
    WIDTH = patients_hu[0].shape[0]
    HEIGHT = patients_hu[0].shape[1]
    MID_HEIGHT = int(HEIGHT // 2)

    ### CONSTANTS
    MAX_FAT = -30
    MIN_FAT = -190


    ### 1. ROI SELECTION
    'Get contours redifined by user'
    new_masks = np.stack([rgb_to_binary(f'{OUTPUT_FOLDER}/{patient_dir}/contours/{patient_id}_{i}.png') for i in range(0, number_of_slices)])

    i = patients_hu[0]
    m = new_masks[0]
    nh= segmentation(i, m)
    plt.imshow(nh)

    new_heart = np.stack([segmentation(i, m) for i, m in zip(patients_hu, new_masks)])

    ### 4. EAT SEGMENTATION
    'Fat thresholding'
    fat_masks = np.stack([threshold_segmentation(r, MIN_FAT, MAX_FAT) for r in new_heart])

    'To show the final result of segmentation'
    np.stack([convert_image_to_rgb(mask, f'{patient_dir}/fat/{patient_id}_{i}', OUTPUT_FOLDER) for mask, i in zip(fat_masks, range(0, number_of_slices))])
    'Read the slice images already save in automatic method'
    images_png = np.stack([cv.imread(f'{OUTPUT_FOLDER}/{patient_dir}/slices/{patient_id}_{i}.png') for i in range(0, number_of_slices)])

    'Add slice image with segmentation mask'
    combined_images = np.stack([add_images(m, img) for m, img in zip(fat_masks, images_png)])

    'Save image with segmentation enhanced'
    np.stack([convert_image_to_rgb(img, f'{patient_dir}/combined/{patient_id}_{i}', OUTPUT_FOLDER) for img, i in zip(combined_images, range(0, number_of_slices))])

    ### 6. VOLUME CALCULATION
    if count_volume:
        fat_volume = volume(fat_masks, thickness, px_spacing)
    else:
        fat_volume = None

    ### 5. CALCULATION RADIOMICS FEATURES
    # if count_radiomics_features:
    #     radiomics_features = calculate_radiomics_features(patients_hu, fat_masks)
    # else:
    #     radiomics_features = None

    return patient_id, patient_dir, number_of_slices, fat_volume
