import argparse
import fnmatch
import functools
import json
import multiprocessing
import os
import logging
import subprocess
import zipfile
import wave
from tqdm import tqdm


logging.basicConfig(level = logging.INFO)


parser = argparse.ArgumentParser(description="AIHub Korean Number Data Pre-processing")
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--data_sets", type=str, default='FINANCE')
parser.add_argument("--training_set", type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument("--num_workers", default=8, type=int)



args = parser.parse_args()

# The parameters are prerequisite information. More specifically,
# channels, bit_depth, sampling_rate must be known to use this function.
# REF: https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=aimldl&logNo=221559323232
def pcm2wav( pcm_file, wav_file, channels=1, bit_depth=16, sampling_rate=16000 ):

    # Check if the options are valid.
    if bit_depth % 8 != 0:
        raise ValueError("bit_depth "+str(bit_depth)+" must be a multiple of 8.")
        
    # Read the .pcm file as a binary file and store the data to pcm_data
    with open( pcm_file, 'rb') as opened_pcm_file:
        pcm_data = opened_pcm_file.read();
        
        obj2write = wave.open( wav_file, 'wb')
        obj2write.setnchannels( channels )
        obj2write.setsampwidth( bit_depth // 8 )
        obj2write.setframerate( sampling_rate )
        obj2write.writeframes( pcm_data )
        obj2write.close()


def __get_audio_path(label_path: str):
    """
    Generate audio file path from label file path

    Args:
        label_path (str): Input label file path to find corresponding audio file path
    Returns:
        _type_: Audio file path
    """
    
    
    # replace label folder with audio folder for both training and label
    pcm_audio_file = label_path.replace("라벨링데이터/TL_", "원천데이터/TS_").replace("라벨링데이터/VL_", "원천데이터/VS_")[:-5] + ".pcm"
    
    # there is a small issue in VS_8.단위 -> 01. | Folder not follow the naming rule
    if("/VS_8.단위/0" in pcm_audio_file):
        pcm_audio_file =  pcm_audio_file.replace("/VS_8.단위/0", "/VS_8.단위/VS_")
        
    return pcm_audio_file


def __extract_file(file_path: str, dst_folder: str):
    """ 
    Convert pcm files to wav, get duration, create json manifest

    Args:   
        file_path (str): source .zip file
        dst_folder (str): destination folder
    """
    
    try:
        # with zipfile.ZipFile(file_path, 'r') as zip_ref:
        #     zip_ref.extractall(dst_folder)
        # Can't work with Korean , wrong output font
        
        
        cmd = ["unzip", file_path, "-d", dst_folder]
        subprocess.run(cmd)
                                     
    except:
        logging.info("Not extracting. Maybe files already exist!")

def __process_transcript(file_path: str, dst_folder: str):
    """
    Converts flac files to wav from a given transcript, capturing the metadata.
    Args:
        file_path: path to a source transcript  with flac sources
        dst_folder: path where wav files will be stored
    Returns:
        a list of metadata entries for processed files.
    """
    
    
    records = []
    # root = os.path.dirname(file_path)
    with open(file_path, encoding="utf-8") as fin:
        
        
        data = json.load(fin)
        transcript_text = data['script']['scriptTN']
        recorded_time = data['audio']['recordedTime']

        # Get .PCM file path from LABEL
        pcm_audio_file = __get_audio_path(label_path=file_path)
        
        
        wav_audio_file = pcm_audio_file.replace("Training", "Training_Processed").replace("Validation", "Validation_Processed")[:-4] + ".wav"
        
        
        if not os.path.exists(wav_audio_file):
            
            wav_dir = os.path.dirname(wav_audio_file)
            os.makedirs(wav_dir, exist_ok=True)
            
            pcm2wav(pcm_file=pcm_audio_file, wav_file=wav_audio_file)
            # Transformer().build(pcm_audio_file, wav_audio_file)  => Not work with .pcm
            
        # check duration   
        # duration = subprocess.check_output("soxi -D {0}".format(wav_audio_file), shell=True)   #=> duration is identical with json label 

        record = {}
        record["audio_filepath"] = os.path.abspath(wav_audio_file)
        record["duration"] = float(recorded_time)
        record["text"] = transcript_text
        records.append(record)
    
    return records

def __process_data(audio_data_folder: str, label_data_folder:str, dst_folder:str, manifest_ouput: str, num_workers: int):
    """Convert pcm to wav and build manifest json file

    Args:
        data_folder (str): data source 
        dst_folder (str): data destination folder
        manifest_ouput (str): where to save manifest file
        num_worker (int): number of parallel workers which will use to process files
    """
    
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        
    
    trans_files = []
    audio_files = []
    records = []
    
    for root_dir, dirnames, filenames in tqdm(os.walk(label_data_folder)):
        for filename in fnmatch.filter(filenames, "*.json"):
        
            audio_file_path = __get_audio_path(label_path=os.path.join(root_dir, filename))
            
            trans_files.append(os.path.join(root_dir, filename))
            audio_files.append(audio_file_path)
            
            if not os.path.isfile(audio_file_path):
                logging.error("ERROR: Audio file not found {0}".format(audio_file_path))
    
    with multiprocessing.Pool(num_workers) as p:
        processing_func = functools.partial(__process_transcript, dst_folder=dst_folder)
        results = p.imap(processing_func, trans_files)
        for result in tqdm(results, total=len(trans_files)):
            records.extend(result)

    with open(manifest_ouput, "w",  encoding='utf-8') as fout:
        for m in records:
            fout.write(json.dumps(m, ensure_ascii=False) + "\n")
            
            
    return records


def main():
    data_root = args.data_root
    num_workers = args.num_workers
    data_sets = args.data_sets
    training_set = args.training_set
    
    if data_sets == "ALL":
        data_sets = "1.개인고유번호,2.고유어기수,3.고유어서수,4.교통정보,5.금융-은행,6.나이-생년월일,7.날짜-시간,8.단위,9.사업자번호,10.사이즈,11.사투리,12.스포츠,13.신분증번호,14.외래영어,15.자동생성번호,16.주문정보,17.주소-구역,18.통계-수치,19.통신번호,20.통화-금액,21.헬스케어"
    if data_sets == "FINANCE":
        data_sets = "5.금융-은행,6.나이-생년월일,7.날짜-시간,8.단위,18.통계-수치,20.통화-금액"

    
    all_records = []
    for data_set in data_sets.split(','):
        logging.info("Working on: {0}".format("TS_" + data_set + "(음성)"))
        
        if training_set:
            audio_path = os.path.join(data_root, "원천데이터", f'TS_{data_set}(음성).zip')
            label_path = os.path.join(data_root, "라벨링데이터", f'TL_{data_set}.zip')
            
            dst_audio_path = os.path.join(data_root, "원천데이터", f'TS_{data_set}')
            dst_label_path = os.path.join(data_root, "라벨링데이터", f'TL_{data_set}')
            
        else:
            audio_path = os.path.join(data_root, "원천데이터", f'VS_{data_set}(음성).zip')
            label_path = os.path.join(data_root, "라벨링데이터", f'VL_{data_set}.zip')
            
            dst_audio_path = os.path.join(data_root, "원천데이터", f'VS_{data_set}')
            dst_label_path = os.path.join(data_root, "라벨링데이터", f'VL_{data_set}')
            
        
        # Extract data if not available
        if os.path.isdir(dst_audio_path):
            logging.info("Audio file {0} already unzipped!".format(dst_audio_path))
        else:
            logging.info("Extracting audio {0}".format(audio_path))
            __extract_file(audio_path, dst_audio_path)
            
        if os.path.isdir(dst_label_path):
            logging.info("Label file {0} already unzipped!".format(dst_label_path))
        else:
            logging.info("Extracting label {0}".format(label_path))
            __extract_file(label_path, dst_label_path)
            
            
        # Process and convert .pcm to .wav / generate manifest json file
        dst_folder_path = data_root + "_Processed"
        curr_records = __process_data(dst_audio_path, dst_label_path, dst_folder_path, os.path.join(dst_folder_path, f'{data_set}_manifest.json'), num_workers)
        all_records.extend(curr_records)
        
        # Save all files to manifest_all.json
        all_record_manifest_path = os.path.join(dst_folder_path, 'manifest_all.json')
        with open(all_record_manifest_path, "w",  encoding='utf-8') as fout:
            for m in all_records:
                fout.write(json.dumps(m, ensure_ascii=False) + "\n")
            
    

if __name__ == '__main__':
    main()
    
