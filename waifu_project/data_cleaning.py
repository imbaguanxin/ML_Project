import os
import re
import shutil


def data_cleaning(moeimouto=True, self_collected=True, destination="modeling_data", force_write=False):
    """
    Copy and rename images stored in data_set/moeimouto-filtered and data_set/self-collected into data_set/{destination}.

    Parameters
    ----------
    moeimouto: bool
        whether to include images in data_set/moeimouto-filtered
    self_collected: bool
        whether to include images in data_set/self-collected
    destination: str
        destination folder
    force_write: bool
        whether to force write in the destination folder if it has already exist
    """
    destination_path = os.path.join('data_set', destination)
    # clean the destination path
    if os.path.exists(destination_path):
        if not force_write:
            print("[STATUS] Destination folder {} already exist, do nothing".format(destination_path))
            return
        shutil.rmtree(destination_path)
    os.makedirs(destination_path)
    if moeimouto:
        moeimouto_clean_up(destination=destination)
    if self_collected:
        self_collected_clean_up(destination=destination)


def moeimouto_clean_up(destination="modeling_data"):
    """
    Copy images stored in data_set/moeimouto-filtered into data_set/{destination}.

    Parameters
    ----------
    destination: str
        destination folder
    """
    source_path = os.path.join('data_set', 'moeimouto-faces-filtered')
    destination_path = os.path.join('data_set', destination)
    source_pics_folders = os.listdir(source_path)
    for folder in source_pics_folders:
        pic_folder_path = os.path.join(source_path, folder)
        modified_folder_name = folder[4:]
        destination_folder = os.path.join(destination_path, modified_folder_name)
        print("[STATUS] copying pics in {} to {}".format(pic_folder_path, destination_folder))
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        pics = os.listdir(pic_folder_path)
        for pic in pics:
            if not re.match(".*csv", pic):
                # handle the statistic file stored in moeimouto data
                shutil.copy(os.path.join(pic_folder_path, pic), destination_folder)
                os.rename(os.path.join(destination_folder, pic), os.path.join(destination_folder, pic))
        print("[STATUS] {} copy work finished".format(folder))


def self_collected_clean_up(destination="modeling_data"):
    """
    Copy images stored in data_set/self-collected into data_set/{destination}.

    Parameters
    ----------
    destination: str
        destination folder
    """
    # set source path and destination path
    source_path = os.path.join('data_set', 'self-collected')
    destination_path = os.path.join('data_set', destination)
    source_pics_folders = os.listdir(source_path)

    # find out doujin pictures and official pictures
    doujin_re = '.*doujin.*'
    official_re = '.*official.*'
    doujin = [x for x in source_pics_folders if re.match(doujin_re, x)]
    official = [x for x in source_pics_folders if re.match(official_re, x)]

    # copy official pics to destination folder
    for file in official:
        # find the picture folder
        pic_folder_path = os.path.join(source_path, file)
        # extract character's name from folder's name
        modified_folder_name = file[6:-9]
        destination_folder = os.path.join(destination_path, modified_folder_name)
        # list all the pictures under the folder
        pics = os.listdir(pic_folder_path)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        print("[STATUS] copying pics in {} to {}".format(pic_folder_path, destination_folder))
        # copy
        for pic in pics:
            shutil.copy(os.path.join(pic_folder_path, pic), destination_folder)
            os.rename(os.path.join(destination_folder, pic), os.path.join(destination_folder, "official_" + pic))
        print("[STATUS] {} copy work finished".format(file))

    # copy doujin pics to destination folder
    for file in doujin:
        # find the picture folder
        pic_folder_path = os.path.join(source_path, file)
        # extract character's name from folder's name
        modified_folder_name = file[6:-7]
        destination_folder = os.path.join(destination_path, modified_folder_name)
        print("[STATUS] copying pics in {} to {}".format(pic_folder_path, destination_folder))
        # list all the pictures under the folder
        pics = os.listdir(pic_folder_path)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        # copy
        for pic in pics:
            shutil.copy(os.path.join(pic_folder_path, pic), destination_folder)
            os.rename(os.path.join(destination_folder, pic), os.path.join(destination_folder, "doujin_" + pic))
        print("[STATUS] {} copy work finished".format(file))


if __name__ == '__main__':
    data_cleaning()
