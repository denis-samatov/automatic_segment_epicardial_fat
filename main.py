from algorithm.UnzipData import upzip_data


from algorithm.LoadingCorrectData import load_correct_data
from algorithm.LoadingDICOMData import create_directory_dict
from _automatic_ import segment_epicardial_fat

import os 
 

def main(zip_file_path: str):
    ### Step 0: Data download    
    extract_folder_path = r'E:\Studies\Bachelor\'s degree work\SOTA\Data\dicom_fat'
    unzip_data_folder = r'E:\Studies\Bachelor\'s degree work\SOTA\Data\unzip_data'
    # upzip_data(zip_file_path, unzip_data_folder, extract_folder_path)

    ### Step 1: Visualizing and Analyzing DICOMs in Python
    directory = r"E:\Studies\Bachelor's degree work\SOTA\Data\unzip_data\Dicom-Originals\Dicom _ Treino"
    result_dicom_pacient_paths = create_directory_dict(directory)

    
    patient_id, number_of_slices, thickness, px_spacing, fat_volume, radiomics_features = segment_epicardial_fat(result_dicom_pacient_paths['ACel'], count_radiomics_features=True, count_volume=False)













if "__main__" == __name__:

    # zip_file_path = str(input('Введите путь к файлу: '))
    zip_file_path = r'E:\Studies\Bachelor\'s degree work\SOTA\Data\Облако Mail.ru.zip'
    main(zip_file_path)