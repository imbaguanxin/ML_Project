import os
import re
import shutil


def data_cleaning():
    source_path = os.path.join('data_set', 'self-collected')  # 'data_set/self-collected (copy)'
    destination_path = os.path.join('data_set', 'modeling_data')
    source_pics_folders = os.listdir(source_path)

    doujin_re = '.*doujin.*'
    official_re = '.*official.*'
    doujin = [x for x in source_pics_folders if re.match(doujin_re, x)]
    official = [x for x in source_pics_folders if re.match(official_re, x)]

    shutil.rmtree(destination_path)
    os.makedirs(destination_path)

    for file in official:
        pic_folder_path = os.path.join(source_path, file)
        modified_file_name = file[6:-9]
        destination_folder = os.path.join(destination_path, modified_file_name)
        pics = os.listdir(pic_folder_path)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        print("COPYING PICS IN {} TO {}".format(pic_folder_path, destination_folder))
        for pic in pics:
            # if re.match(official_re, pic) is None:
            shutil.copy(os.path.join(pic_folder_path, pic), destination_folder)
            os.rename(os.path.join(destination_folder, pic), os.path.join(destination_folder, "official_" + pic))
        print("{} COPY WORK FINISHED".format(file))

    for file in doujin:
        pic_folder_path = os.path.join(source_path, file)
        modified_file_name = file[6:-7]
        destination_folder = os.path.join(destination_path, modified_file_name)
        print("COPYING PICS IN {} TO {}".format(pic_folder_path, destination_folder))
        pics = os.listdir(pic_folder_path)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        for pic in pics:
            # if re.match(doujin_re, pic) is None:
            shutil.copy(os.path.join(pic_folder_path, pic), destination_folder)
            os.rename(os.path.join(destination_folder, pic), os.path.join(destination_folder, "doujin_" + pic))
        print("{} COPY WORK FINISHED".format(file))


if __name__ == '__main__':
    data_cleaning()
