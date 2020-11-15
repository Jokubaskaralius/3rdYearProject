import dicom2nifti
import os
import re
from shutil import copyfile
from utils import getDataPath

# dicom2nifti.convert_directory(
#     "/home/jokubas/DevWork/3rdYearProject/data/IGT_glioma/00008/5",
#     "/home/jokubas/DevWork/3rdYearProject/data/IGT_glioma/00008/5/",
#     reorient=True,
#     compression=True)


class Dataset_dicom2nifti:
    def __init__(self, dataset, output_dir):
        self.dataset_name = dataset
        self.output_dir_name = output_dir
        self.dataset_dir = os.path.join(getDataPath(), dataset)

    def subfolder_list(self, dir_name):
        return [f.path for f in os.scandir(dir_name) if f.is_dir()]

    def conv_dataset_dicom2nifti(self):
        for folder1 in self.subfolder_list(self.dataset_dir):
            for folder1_2 in self.subfolder_list(folder1):
                for folder2 in self.subfolder_list(folder1_2):
                    input_dir = folder2
                    output_dir = re.sub(self.dataset_name,
                                        self.output_dir_name, folder2)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    print(folder2)
                    dicom2nifti.convert_directory(input_dir,
                                                  output_dir,
                                                  compression=True,
                                                  reorient=True)


#gd = Dataset_dicom2nifti("REMBRANDT", "nifti_REMBRANDT")
#gd = Dataset_dicom2nifti("IGT_glioma", "nifti_IGT_glioma")
#gd = Dataset_dicom2nifti("TCGA-LGG", "nifti_TCGA_LGG")
#gd.conv_dataset_dicom2nifti()
