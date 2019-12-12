import os
import shutil


def copy_file(source_path,target_path,filter):

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    num = 0

    class_dir_list = sorted(os.listdir(source_path))

    for class_dir in class_dir_list:
        class_dir_path = os.path.join(source_path,class_dir)
        class_image_list = sorted(os.listdir(class_dir_path))
        print(class_dir)
        for class_image in class_image_list:
            if filter in class_image:
                fp = os.path.join(class_dir_path,class_image)
                if num % 100 == 0:
                    print(class_image)
                newfp = os.path.join(target_path,'%04d'%num+".tiff")
                shutil.copy(fp,newfp)
                num += 1

    print("移动完成")

if __name__ == '__main__':

    nirscene_path = "/mnt/4T/zlk/datasets/mulitspectral/nirscene1"
    nir_target_path = "/mnt/4T/zlk/datasets/mulitspectral/nirscene_total/nir_image"
    rgb_target_path = "/mnt/4T/zlk/datasets/mulitspectral/nirscene_total/rgb_image"

    copy_file(nirscene_path,nir_target_path,'nir')
    copy_file(nirscene_path,rgb_target_path,'rgb')


