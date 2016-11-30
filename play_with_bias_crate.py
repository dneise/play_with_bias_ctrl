# IPython log file

import numpy as np
import serial
import struct
import time
import pandas as pd
from itertools import zip_longest
from tqdm import tqdm, trange
from collections import namedtuple

ser = serial.Serial('/dev/serial/by-id/usb-FTDI_UM245R_FTE03K4V-if00-port0')
ser.timeout = 1
commands = {'reset': 0, 'read': 1, 'set': 3}

Answer = namedtuple('Answer', 'current, counter, over_current, errors, board')


def grouper(iterable, num, fillvalue=None):
    args = [iter(iterable)] * num
    return zip_longest(*args, fillvalue=fillvalue)


def make_send_bytes(cmd, board, channel, voltage):
    word = 0
    word |= (voltage & 0xfff) << 0
    word |= (channel & 0x1f) << 12
    word |= (board & 0xf) << (12+5)
    word |= (cmd & 0x7) << (12+5+4)
    return struct.pack('!I', word)[1:]


def make_answer(string):
    if not len(string) == 4:
        string = b'\x00' + string
    word = struct.unpack('!I', string)[0]
    return Answer(
        current=(word >> 9) & 0x7ff, # its an 11-bit ADC really.
        counter=(word >> 20) & 0x7,
        over_current=(word >> 23) & 0x1,
        errors=(word >> 4) & 0xf,
        board=word & 0xf
        )


def set_voltage_read_N_times(board, channel, voltage, num=1):
    return set_voltage_M_times_read_N_times(board, channel, voltage, M=1, N=num-1)


def set_voltage_M_times_read_N_times(board, channel, voltage, M=1, N=0):
    cmds = (
        M * make_send_bytes(commands['set'], board, channel, voltage)
        + N * make_send_bytes(commands['read'], board, channel, 0)
        )
    start_time = time.time()
    ser.write(cmds)
    answer = b''
    while not len(answer) == len(cmds):
        answer += ser.read(len(cmds) - len(answer))
    stop_time = time.time()

    df = pd.DataFrame([
        make_answer(answer[i*3:(i+1)*3]) for i in range(len(answer)//3)
    ])
    df['Time'] = pd.to_datetime(
        np.linspace(start_time, stop_time, M+N) * 1e9,
        unit='ns'
        )
    df['dac'] = voltage
    df['channel'] = channel
    df['board_'] = board

    # change dtypes, so df is nice and small
    df['current'] = df['current'].astype(np.uint16)
    df['counter'] = df['counter'].astype(np.uint8)
    df['over_current'] = df['over_current'].astype(np.bool)
    df['errors'] = df['errors'].astype(np.uint8)
    df['board'] = df['board'].astype(np.int8)
    df['dac'] = df['dac'].astype(np.uint16)
    df['channel'] = df['channel'].astype(np.uint8)
    df['board_'] = df['board_'].astype(np.uint8)

    return check_and_delete_cols(df)


def check_and_delete_cols(df):
    if not (df.board_ == df.board).all():
        print("no all boards equal - should never happen")
    else:
        df.drop('board_', axis=1, inplace=True)

    if df.over_current.any():
        print("over_current bit was set")
    else:
        df.drop('over_current', axis=1, inplace=True)

    if df.errors.any():
        print("some error bits were set")
    else:
        df.drop('errors', axis=1, inplace=True)

    counter_start = df.counter.iloc[0]
    comparison_counter = np.arange(counter_start, counter_start + len(df)) % 8
    if not (df.counter == comparison_counter).all():
        print("the 3-bit wrap counter does not always count up")
    else:
        df.drop('counter', axis=1, inplace=True)

    return df


def ramp_up_down_experiment(
        channel=0,
        board=0,
        low_dac=0,
        high_dac=200,
        dac_step=10,
        N_settings=1,
        N_readings=200,
        delay=0
        ):

    dfs = []
    for dac in range(low_dac, high_dac+dac_step, dac_step):
        df = set_voltage_M_times_read_N_times(
            board=board,
            channel=channel,
            voltage=dac,
            M=N_settings,
            N=N_readings)
        dfs.append(df)
        time.sleep(delay)

    for dac in range(high_dac, low_dac-dac_step, -dac_step):
        df = set_voltage_M_times_read_N_times(
            board=board,
            channel=channel,
            voltage=dac,
            M=N_settings,
            N=N_readings)
        dfs.append(df)
        time.sleep(delay)
    dfs = pd.concat(dfs)
    return dfs


def ramp_up_down_whole_camera_experiment(full_data=False):
    dfs = []
    for board in trange(10):
        for channel in range(32):
            d = ramp_up_down_experiment(
                channel=channel,
                board=board,
                dac_step=25,
                high_dac=200,
                )
            if not full_data:
                d = d[d.index >= 150]
                d = d[d.dac >= 100]
            dfs.append(d)
    return pd.concat(dfs)


def set_whole_camera(dac):
    boards = np.arange(10, dtype=np.uint8).repeat(32)
    channels = np.arange(32, dtype=np.uint8).tile(10)
    cmds = b''
    for i in range(len(boards)):
        cmds += make_send_bytes(
            commands['set'],
            board=boards[i],
            channel=channels[i],
            voltage=dac
            )

    start_time = time.time()
    ser.write(cmds)
    answer = b''
    while not len(answer) == len(cmds):
        answer += ser.read(len(cmds) - len(answer))
    stop_time = time.time()
    df = pd.DataFrame([
        make_answer(answer[i*3:(i+1)*3]) for i in range(len(answer)//3)
    ])
    df['Time'] = pd.to_datetime(
        np.linspace(start_time, stop_time, len(answer)//3) * 1e9,
        unit='ns'
        )
    df['dac'] = dac
    df['channel'] = channels
    df['board_'] = boards

    # change dtypes, so df is nice and small
    df['current'] = df['current'].astype(np.uint16)
    df['counter'] = df['counter'].astype(np.uint8)
    df['over_current'] = df['over_current'].astype(np.bool)
    df['errors'] = df['errors'].astype(np.uint8)
    df['board'] = df['board'].astype(np.int8)
    df['dac'] = df['dac'].astype(np.uint16)

    return check_and_delete_cols(df)


"""
if __name__ == '__main__':

    d = ramp_up_down_whole_camera_experiment(full_data=True)
    d.to_hdf('ramp_up_down_whole_camera_experiment__full_data.h5', 'all')
    """