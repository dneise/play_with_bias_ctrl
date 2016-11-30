# IPython log file

import numpy as np
import serial
import struct
import time
import pandas as pd
from itertools import zip_longest

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
    up_dacs = list(range(low_dac, high_dac+dac_step, dac_step))
    dacs = up_dacs + up_dacs[::-1]

    dfs = []
    for dac in dacs:
        df = set_voltage_M_times_read_N_times(
            board=board,
            channel=channel,
            voltage=dac,
            M=N_settings,
            N=N_readings)
        df.append(dfs)
        time.sleep(delay)
    return dfs
