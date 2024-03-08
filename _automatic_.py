# common packages
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

from skimage.morphology import convex_hull_image
from typing import List, Tuple
from algorithm.BasicOperationsForImageProcessing import smoothing_and_denoising, otsu_mask, convert_image_to_rgb, connect_components, segmentation, opening, add_images
from algorithm.ConvertToHUAndSortedData import load_scan, get_pixels_hu
from algorithm.DiscardingUnnecessarySections import do_touch_margin, count_discard_slices
from algorithm.CountVolume import volume
from algorithm.SearchForHeartPositionPatterns import find_template_spine, find_template_sternal, cut_image_from_bottom, cut_image_from_top
from algorithm.ProcessingCardiacROIContours import draw_contours, correct_contours
from algorithm.ThresholdSegmentationByHU import threshold_segmentation
from algorithm.LoadingDICOMData import create_directory_dict
# from algorithm.CountRadiomicsFeatures import calculate_radiomics_features

index = -1

def segment_epicardial_fat(DICOM_DATASET: List[str], count_radiomics_features: bool = False, count_volume: bool = True, OUTPUT_FOLDER: str = 'result_folder/'):
  """
  Сегментация эпикардиального жира на основе DICOM-данных.

  Параметры:
  - DICOM_DATASET (List[str]): Список путей к DICOM-файлам.
  - OUTPUT_FOLDER (str): Папка для сохранения результатов. По умолчанию 'result_folder/'.

  Возвращает кортеж из идентификатора пациента, числа срезов, объема жира и радиометрических признаков.
  """
  'Select the patient dataset'

  if type(DICOM_DATASET) == str: 
    result_dicom_pacient_paths = create_directory_dict(DICOM_DATASET)

    patient_dir = str(list(result_dicom_pacient_paths.keys())[0])
    patient = load_scan(*result_dicom_pacient_paths.values())
    print('Passed: LOAD DATA')
  else:
    patient_dir = DICOM_DATASET[0].split(os.path.sep)[-1]
    patient = load_scan(DICOM_DATASET)
    print('Passed: LOAD DATA')
    
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

  ### PRE-PROCESSING
  median_images = smoothing_and_denoising(patients_hu)

  ### 1. ROI SELECTION
  print('Passed: ROI SELECTION')
  'Matching template to detect sternal and spine'
  template_sternal = cv.imread('./HARTA/resources/template_sternum.png')
  template_spine = cv.imread('./HARTA/resources/template_spine.png')

  'Convert imagens to rgb channels'
  images_png = np.stack([convert_image_to_rgb(dicom, f'{patient_id}_{i}', f'{OUTPUT_FOLDER}/{patient_dir}/slices') for i, dicom in enumerate(median_images)])

  #  НУЖНО СОЗДАТЬ СВОЮ СТАТИСТИКУ ДЛЯ НАКОЖДЕНИЯ РАМОК СЕРДЦА
  top_points = find_template_sternal(images_png, template_sternal, method='cv.TM_CCOEFF_NORMED')[1]
  bottom_points = find_template_spine(images_png, template_spine, method='cv.TM_CCOEFF_NORMED')[0]

  'Remove the lungs'
  remove_lung_mask = np.stack([otsu_mask(patient) for patient in median_images])

  'Remove torax'
  remove_torax_mask = np.stack([cut_image_from_top(slice, point[1], WIDTH, y_reject=0) for slice, point in
                                zip(remove_lung_mask, top_points)])

  'Remove spine'
  remove_spine_mask = np.stack(
      [cut_image_from_bottom(slice, MID_HEIGHT + point[1], WIDTH, y_reject=MID_HEIGHT) for slice, point in
        zip(remove_torax_mask, bottom_points)])


  ### 2. HEART ROI
  print('Passed: HEART ROI')
  'Find contours with ellipse morphology'
  contours = np.stack([draw_contours(img.astype(np.uint8)) for img in remove_spine_mask])

  'Selection of the bigger component regarding the component with the heart'
  bigger_component = np.stack([connect_components(mask) for mask in contours])

  'Apply masks of ROI to segmentate the heart'
  masks = np.stack([contour > 0 for contour in bigger_component])
  heart = np.stack([segmentation(patient_hu, mask) for patient_hu, mask in zip(patients_hu, masks)])


  ### 3. PERICARDIUM DELIMITATION
  print('Passed: PERICARDIUM DELIMITATION')
  # -44 a 18 HU
  'Pericardium thresholding'
  pericardium_mask = np.stack([threshold_segmentation(slice, -44, -18) for slice in heart])
  pericardium_opening = np.stack([opening(mask, kernel_size=(1,1)) for mask in pericardium_mask])
  # pericardium_erosion = np.stack([erosion(mask) for mask in pericardium_mask])
  

  'Get pericardium contour'
  pericardium_contour = np.stack([convex_hull_image(mask) for mask in pericardium_opening])
  pericardium_opening = np.stack([opening(mask) for mask in pericardium_contour])

  'Redefine ROI of heart'
  new_contours = np.stack([contour > 0 for contour in pericardium_opening])

  'Discard slices that touch left or right margins'
  processed_slices = np.stack([do_touch_margin(slice) for slice in new_contours])
  correced_slices = correct_contours(processed_slices)
  discard_slices = count_discard_slices(correced_slices)

  'Convert imagens to rgb channels'
  np.stack([convert_image_to_rgb(slice, f'{patient_id}_{i}', f'{OUTPUT_FOLDER}/{patient_dir}/contours') for slice, i in zip(correced_slices, range(0, number_of_slices))])

  'Discard slices that touch left or right margins'
  new_masks = np.stack([contour > 0 for contour in correced_slices])
  new_heart = np.stack([segmentation(patient_hu, mask) for patient_hu, mask in zip(patients_hu, new_masks)])

  ### 4. FAT SEGMENTATION
  print('Passed: FAT SEGMENTATION')
  'Fat thresholding'
  fat_masks = np.stack([threshold_segmentation(slice, MIN_FAT, MAX_FAT) for slice in new_heart])

  'To show the final result of segmentation'
  np.stack([convert_image_to_rgb(mask, f'{patient_id}_{i}', f'{OUTPUT_FOLDER}/{patient_dir}/fat') for mask, i in zip(fat_masks, range(0, number_of_slices))])

  'Add slice image with segmentation mask'
  combined_images = np.stack([add_images(mask, img) for mask, img in zip(fat_masks, images_png)])

  'Save image with segmentation enhanced'
  np.stack([convert_image_to_rgb(img, f'{patient_id}_{i}', f'{OUTPUT_FOLDER}/{patient_dir}/combined') for img, i in zip(combined_images, range(0, number_of_slices))])

  ### 5. CALCULATION OF FAT VOLUME
  print('Passed: CALCULATION OF FAT VOLUME')
  if count_volume:
    fat_volume = volume(fat_masks, thickness, px_spacing)
  else:
    fat_volume = None

  ## 6. CALCULATION RADIOMICS FEATURES
  if count_radiomics_features:
    radiomics_features = calculate_radiomics_features(patients_hu, fat_masks)
  else:
    radiomics_features = None


  # plt.figure(figsize=(20, 16))
  
  # plt.subplot(1, 11, 1)
  # plt.title('1')
  # plt.imshow(images_png[index], cmap='gray')
  # plt.grid(None)
  # plt.axis(False)

  # plt.subplot(1, 11, 2)
  # plt.title('2')
  # plt.imshow(remove_lung_mask[index], cmap='gray')
  # plt.grid(None)
  # plt.axis(False)

  # plt.subplot(1, 11, 3)
  # plt.title('3')
  # plt.imshow(remove_torax_mask[index], cmap='gray')
  # plt.grid(None)
  # plt.axis(False)

  # plt.subplot(1, 11, 4)
  # plt.title('4')
  # plt.imshow(remove_spine_mask[index], cmap='gray')
  # plt.grid(None)
  # plt.axis(False)

  # plt.subplot(1, 11, 5)
  # plt.title('5')
  # plt.imshow(bigger_component[index], cmap='gray')
  # plt.grid(None)
  # plt.axis(False)

  # plt.subplot(1, 11, 6)
  # plt.title('6')
  # plt.imshow(heart[index], cmap='gray')
  # plt.grid(None)
  # plt.axis(False)

  # plt.subplot(1, 11, 7)
  # plt.title('7')
  # plt.imshow(pericardium_mask[index], cmap='gray')
  # plt.grid(None)
  # plt.axis(False)

  # plt.subplot(1, 11, 8)
  # plt.title('8')
  # plt.imshow(pericardium_contour[index], cmap='gray')
  # plt.grid(None)
  # plt.axis(False)

  # plt.subplot(1, 11, 9)
  # plt.title('9')
  # plt.imshow(pericardium_opening[index], cmap='gray')
  # plt.grid(None)
  # plt.axis(False)

  # plt.subplot(1, 11, 10)
  # plt.title('10')
  # plt.imshow(new_contours[index], cmap='gray')
  # plt.grid(None)
  # plt.axis(False)

  # plt.subplot(1, 11, 11)
  # plt.title('11')
  # plt.imshow(fat_masks[index], cmap='gray')
  # plt.grid(None)
  # plt.axis(False)

  # plt.show()

  return patient_id, patient_dir, number_of_slices, fat_volume