from PIL import Image
import numpy as np 
import glob
import cv2
import datetime
import os

folder_tracebility = 'TRACEBILITY_IMAGES'

variant_df = {
    "00" : "TEST",
    "21" : "VELOZ",
    "20" : "VELOZ",
    "90" : "YARIS CROSS",
    "40" : "CALYA",
    "41" : "CALYA",
    "05" : "YARIS"
}

def saveImageCapture(*args, variant, suffix_no, camera_idx):

    timestamp = datetime.datetime.now()
    date = timestamp.strftime("%y-%m-%d")

    folder_vehicle_trace = f'{folder_tracebility}/{date}/{camera_idx}/{variant}'
    folder_vehicle_check_trace = os.path.exists(folder_vehicle_trace)

    if folder_vehicle_check_trace == False:
        os.makedirs(folder_vehicle_trace)


    frame_bgr2rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in args]

    folder_vehicle_glob = glob.glob(f'{folder_vehicle_trace}/*jpeg')

    seq_vehicle  = len(folder_vehicle_glob)

    images = [Image.fromarray(np.array(x)) for x in frame_bgr2rgb]

    for im in images:
        im.save(r'{}/{}/{}/{}/{}_{}_{}.jpeg'.format(folder_tracebility, date, camera_idx, variant, variant, suffix_no, seq_vehicle))
