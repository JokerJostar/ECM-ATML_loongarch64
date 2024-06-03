import argparse
import serial
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os
import struct
import pandas as pd

def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float64)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1
    return datamat


def float_to_hex(f):
    # return hex(struct.unpack('<I', struct.pack('<f', f))[0])
    return struct.pack('>f', float(f)).hex()


def hex_to_float(h):
    i = int(h, 16)
    return struct.unpack('<f', struct.pack('<I', i))[0]


def double_to_hex(f):
    return hex(struct.unpack('<Q', struct.pack('<d', f))[0])


def hex_to_double(h):
    i = int(h, 16)
    return struct.unpack('<d', struct.pack('<Q', i))[0]


def print_hex(bytes):
    l = [hex(int(i)) for i in bytes]
    print(" ".join(l))

def ACC(mylist):
  tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
  total = sum(mylist)
  acc = (tp + tn) / total
  return acc

def PPV(mylist):
  tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
  # for the case: there is no VA segs for the patient, then ppv should be 1
  if tp + fn == 0:
    ppv = 1
  # for the case: there is some VA segs, but the predictions are wrong
  elif tp + fp == 0 and tp + fn != 0:
    ppv = 0
  else:
    ppv = tp / (tp + fp)
  return ppv


def NPV(mylist):
  tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
  # for the case: there is no non-VA segs for the patient, then npv should be 1
  if tn + fp == 0:
    npv = 1
  # for the case: there is some VA segs, but the predictions are wrong
  elif tn + fn == 0 and tn + fp != 0:
    npv = 0
  else:
    npv = tn / (tn + fn)
  return npv

def Sensitivity(mylist):
  tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
  # for the case: there is no VA segs for the patient, then sen should be 1
  if tp + fn == 0:
    sensitivity = 1
  else:
    sensitivity = tp / (tp + fn)
  return sensitivity


def Specificity(mylist):
  tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
  # for the case: there is no non-VA segs for the patient, then spe should be 1
  if tn + fp == 0:
    specificity = 1
  else:
    specificity = tn / (tn + fp)
  return specificity


def BAC(mylist):
  sensitivity = Sensitivity(mylist)
  specificity = Specificity(mylist)
  b_acc = (sensitivity + specificity) / 2
  return b_acc


def F1(mylist):
  precision = PPV(mylist)
  recall = Sensitivity(mylist)
  if precision + recall == 0:
    f1 = 0
  else:
    f1 = 2 * (precision * recall) / (precision + recall)
  return f1


def FB(mylist, beta=2):
  precision = PPV(mylist)
  recall = Sensitivity(mylist)
  if precision + recall == 0:
    f1 = 0
  else:
    f1 = (1 + beta ** 2) * (precision * recall) / ((beta ** 2) * precision + recall)
  return f1

def main():

    test_indice_path = args.path_indices + 'final_test_indice.csv'
    test_indices = pd.read_csv(test_indice_path)  # Adjust delimiter if necessary
    subjects = test_indices['Filename'].apply(lambda x: x.split('-')[0]).unique().tolist()
    # List to store metrics for each participant
    subject_metrics = []
    subjects_above_threshold = 0
    timeList = []

    if not os.path.exists('./log/'):
        os.makedirs('./log/')
    dataList = {}

    for subject_id in subjects:
      dataList[subject_id] = []

    for file in test_indices['Filename']:
      subject_id = file.split('-')[0]
      dataList[subject_id].append(file)

    port = args.com  # set port number
    ser = serial.Serial(port=port, baudrate=args.baudrate, timeout=10)  # open the serial , timeout=1
    ser.set_buffer_size(rx_size=12800, tx_size=12800)
    print(ser)

    if not os.path.exists(args.path_records):
        os.makedirs(args.path_records)

    for subject_id in subjects:
    # for subject_id in ('S6','S7'):
      print(subject_id)
      segs_TP = 0
      segs_TN = 0
      segs_FP = 0
      segs_FN = 0
      for idx in tqdm(range(len(dataList[subject_id]))):
      # for idx in tqdm(range(100)):
        testX = txt_to_numpy(args.path_data + dataList[subject_id][idx], 1250).reshape(1, 1, 1250, 1)
        testZ = np.arange(1250 + 1, dtype=np.int64)
        s = 1
        for i in range(0, testX.shape[0]):
          for j in range(0, testX.shape[1]):
            for k in range(0, testX.shape[2]):
              for l in range(0, testX.shape[3]):
                testZ[s] = int(float_to_hex(testX[i][j][k][0]), base=16)
                s += 1
                # print(k,":",testX[i][j][k][0])
        testW = np.asanyarray(testZ, dtype="uint32")
        ser.flushOutput()
        datalen = 1250
        testW[0] = (datalen << 16) | 0x55aa
        result = ser.write(testW)
        # ser.in_waiting()
        while ser.in_waiting < 5:
          pass
          time.sleep(0.01)
        recv = ser.read(8)
        ser.reset_input_buffer()
        # the format of recv is ['<result>','<dutation>']
        result = recv[3]
        tm_cost = recv[4] | (recv[5] << 8) | (recv[6] << 16) | (recv[7] << 24)
        s_tm_cost = str(hex(tm_cost))
        f_tm_cost = hex_to_float(s_tm_cost)
        timeList.append(f_tm_cost)
        if 'AFIB' in dataList[subject_id][idx]:
          segs_FN += 1 if result == 0 else 0
          segs_TP += 1 if result == 1 else 0
        else:
          segs_TN += 1 if result == 0 else 0
          segs_FP += 1 if result == 1 else 0
      f1 = round(F1([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
      fb = round(FB([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
      se = round(Sensitivity([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
      sp = round(Specificity([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
      bac = round(BAC([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
      acc = round(ACC([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
      ppv = round(PPV([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
      npv = round(NPV([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
      subject_metrics.append([f1, fb, se, sp, bac, acc, ppv, npv])

      if fb > 0.9:
        subjects_above_threshold += 1

    subject_metrics_array = np.array(subject_metrics)
    average_metrics = np.mean(subject_metrics_array, axis=0)

    avg_f1, avg_fb, avg_se, avg_sp, avg_bac, avg_acc, avg_ppv, avg_npv = average_metrics
    total_time = sum(timeList)
    avg_time = np.mean(timeList)
    # Print average metric values
    print(f"Final F-1: {avg_f1:.5f}")
    print(f"Final F-B: {avg_fb:.5f}")
    print(f"Final SEN: {avg_se:.5f}")
    print(f"Final SPE: {avg_sp:.5f}")
    print(f"Final BAC: {avg_bac:.5f}")
    print(f"Final ACC: {avg_acc:.5f}")
    print(f"Final PPV: {avg_ppv:.5f}")
    print(f"Final NPV: {avg_npv:.5f}")
    print(f"Total Time: {total_time}")
    print(f"Average Time: {avg_time}")

    proportion_above_threshold = subjects_above_threshold / len(subjects)
    print("G Score:", proportion_above_threshold)

    with open(args.path_records + 'seg_stat_'+time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))+'.txt', 'w') as f:
        f.write(f"Final F-1: {avg_f1:.5f}\n")
        f.write(f"Final F-B: {avg_fb:.5f}\n")
        f.write(f"Final SEN: {avg_se:.5f}\n")
        f.write(f"Final SPE: {avg_sp:.5f}\n")
        f.write(f"Final BAC: {avg_bac:.5f}\n")
        f.write(f"Final ACC: {avg_acc:.5f}\n")
        f.write(f"Final PPV: {avg_ppv:.5f}\n")
        f.write(f"Final NPV: {avg_npv:.5f}\n\n")
        f.write(f"Total Time: {total_time}\n")
        f.write(f"Average Time: {avg_time}\n")
        f.write(f"G Score: {proportion_above_threshold}\n")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--com', type=str, default='com1')
    argparser.add_argument('--baudrate', type=int, default=115200)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./data/testing_dataset/')
    argparser.add_argument('--path_net', type=str, default='./saved_models/')
    argparser.add_argument('--path_records', type=str, default='./records/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    args = argparser.parse_args()
    main()