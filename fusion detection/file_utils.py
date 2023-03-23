import os
def create_folders(folder_name):
    for mode in ["train", "val", "test", "test-challenge"]:
        if not os.path.exists(os.path.join(".", folder_name, mode)):
            os.makedirs(os.path.join(".", folder_name, mode))
            os.makedirs(os.path.join(".", folder_name, mode, "images"))
            os.makedirs(os.path.join(".", folder_name, mode, "annotations"))
