import numpy as np
import matplotlib.pyplot as plt
import struct
import mne

# Adapted from fieldtrip to python by DLP
# Original fieldtrip function:
# reads Micromed .TRC file into matlab, version Mariska, edited by Romain
# input: filename
# output: datamatrix


def read_raw_micromed_trc(trc_fname, begsample=None, endsample=None):
    """Reader for Micromed EEG file.

    Parameters
    ----------
    trc_fname : path-like
        Path to the EEG file.
    begsample : int
        first sample to read.
    endsample : int
        last sample to read.

    Returns
    -------
    raw : instance of RawBrainVision
        A Raw object containing BrainVision data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawBrainVision.
    """
    info = {}
    elec_info = []

    # Opening File
    with open(trc_fname, 'rb') as fid:
        # Reading patient & recording info
        fid.seek(64, 0)
        last_name = fid.read(22).decode('utf-8')
        first_name = fid.read(20).decode('utf-8')

        fid.seek(128, 0)
        day = int.from_bytes(fid.read(1), byteorder='little')
        month = str(int.from_bytes(fid.read(1), byteorder='little'))
        year = int.from_bytes(fid.read(1), byteorder='little')

        info['last_name'] = last_name
        info['first_name'] = first_name
        info['day'] = day
        info['month'] = month
        info['year'] = str(year + 1900)

        # Reading Header Info
        fid.seek(175, 0)
        hdr_code = struct.unpack('B', fid.read(1))[0]
        if hdr_code != 4:
            print("Invalid header code for Micromed file")

        fid.seek(138, 0)
        data_offset = struct.unpack('I', fid.read(4))[0]
        n_chann = struct.unpack('H', fid.read(2))[0]
        multiplex = struct.unpack('H', fid.read(2))[0]
        fs = struct.unpack('H', fid.read(2))[0]
        n_bytes = struct.unpack('H', fid.read(2))[0]

        """
        info['Data_Start_Offset'] = struct.unpack('I', fid.read(4))[0]
        n_chann = struct.unpack('H', fid.read(2))[0]
        info['Multiplexer'] = struct.unpack('H', fid.read(2))[0]
        info['Rate_Min'] = struct.unpack('H', fid.read(2))[0]
        info['Bytes'] = struct.unpack('H', fid.read(2))[0]
           
        fid.seek(184, 0)
        info['Code_Area'] = struct.unpack('I', fid.read(4))[0]
        info['Code_Area_Length'] = struct.unpack('I', fid.read(4))[0]

        fid.seek(192 + 8, 0)
        info['Electrode_Area'] = struct.unpack('I', fid.read(4))[0]
        info['Electrode_Area_Length'] = struct.unpack('I', fid.read(4))[0]

        fid.seek(400 + 8, 0)
        info['Trigger_Area'] = struct.unpack('I', fid.read(4))[0]
        info['Trigger_Area_Length'] = struct.unpack('I', fid.read(4))[0] 
        """

        # Retrieving electrode info
        fid.seek(184, 0)
        order_offset = struct.unpack('I', fid.read(4))[0]
        fid.seek(order_offset, 0)
        order = struct.unpack(f'{n_chann}H', fid.read(2 * n_chann))
        fid.seek(200, 0)
        elec_offset = struct.unpack('I', fid.read(4))[0]

        for i in range(n_chann):
            fid.seek(elec_offset + 128 * order[i], 0)
            if not struct.unpack('B', fid.read(1))[0]:
                continue

            elec = {}
            elec['bip'] = struct.unpack('B', fid.read(1))[0]

            elec_name_bytes = fid.read(6)
            elec_name_str = elec_name_bytes.decode("utf-8")
            elec_name_str = elec_name_str.replace("\x00", "")
            elec_name_str = elec_name_str.replace(' ', '')
            elec_name_str = elec_name_str.strip()
            elec['Name'] = elec_name_str

            ref_bytes = fid.read(6)
            ref_str = ref_bytes.decode("utf-8")
            ref_str = ref_str.replace("\x00", "")
            ref_str = ref_str.replace(' ', '')
            ref_str = ref_str.strip()
            elec['Ref'] = ref_str

            elec['LogicMin'] = struct.unpack('l', fid.read(4))[0]
            elec['LogicMax'] = struct.unpack('l', fid.read(4))[0]
            elec['LogicGnd'] = struct.unpack('l', fid.read(4))[0]
            elec['PhysMin'] = struct.unpack('l', fid.read(4))[0]
            elec['PhysMax'] = struct.unpack('l', fid.read(4))[0]

            unit = struct.unpack('H', fid.read(2))[0]
            units = {
                -1: 'nV',
                0: 'uV',
                1: 'mV',
                2: 'V',
                100: '%',
                101: 'bpm',
                102: 'Adim.'
            }
            elec['Unit'] = units[unit]

            fid.seek(elec_offset + 128 * order[i] + 44, 0)
            elec['FsCoeff'] = struct.unpack('H', fid.read(2))[0]
            fid.seek(elec_offset + 128 * order[i] + 90, 0)
            elec['XPos'] = struct.unpack('f', fid.read(4))[0]
            elec['YPos'] = struct.unpack('f', fid.read(4))[0]
            elec['ZPos'] = struct.unpack('f', fid.read(4))[0]

            fid.seek(elec_offset + 128 * order[i] + 102, 0)
            elec['Type'] = struct.unpack('H', fid.read(2))[0]

            elec_info.append(elec)

        info['elec'] = elec_info

        # Read Trace Data
        # Determine the number of samples
        fid.seek(data_offset, 0)
        datbeg = fid.tell()
        fid.seek(0, 2)
        datend = fid.tell()
        n_samples_tot = (datend - datbeg) // (n_bytes * n_chann)
        if (datend - datbeg) % (n_bytes * n_chann) != 0:
            print('Rounding off the number of samples')
            n_samples_tot = (datend - datbeg) // (n_bytes * n_chann)

        # Determine the range of data to read
        if begsample is None:
            begsample = 0
        if endsample is None or endsample >= n_samples_tot:
            endsample = n_samples_tot-1

        fid.seek(data_offset, 0)
        fid.seek(n_chann*n_bytes*(begsample), 1)

        n_channs = n_chann
        n_samples_read = endsample - begsample + 1
        data = np.zeros((n_channs, n_samples_read), dtype=float)
        for si in range(n_samples_read):
            if n_bytes == 1:
                pack_bytes = fid.read(n_channs)
                pack_struct = struct.unpack(f'{n_channs}B', pack_bytes)
                data[:, si] = pack_struct
            elif n_bytes == 2:
                pack_bytes = fid.read(n_bytes*n_channs)
                pack_struct = struct.unpack(f'{n_channs}H', pack_bytes)
                data[:, si] = pack_struct
            elif n_bytes == 4:
                pack_bytes = fid.read(n_bytes*n_channs)
                pack_struct = struct.unpack(f'{n_channs}I', pack_bytes)
                data[:, si] = pack_struct

        # Perform data conversion to microvolts
        for chi in range(n_channs):
            term_a = elec_info[chi]['LogicGnd']
            term_b = (elec_info[chi]['LogicMax'] -
                      elec_info[chi]['LogicMin'] + 1)
            term_c = (elec_info[chi]['PhysMax'] - elec_info[chi]['PhysMin'])
            data[chi, :] = (data[chi, :]-term_a)/term_b * term_c

        ch_names = [info['elec'][chi]['Name']
                    for chi in range(n_chann)]
        ch_types = ["eeg"] * n_chann

        mne_info = mne.create_info(
            ch_names=ch_names, ch_types=ch_types, sfreq=fs)

        plot_ok = False
        if plot_ok:
            time = np.arange(n_samples_read)/fs
            for chi in range(n_channs):
                plt.plot(time, data[chi, :], '-k', linewidth=1)
                plt.title(ch_names[chi])
                plt.show(block=False)
                plt.close()

        micromed_raw = mne.io.RawArray(data, mne_info)
        # micromed_raw.plot(show_scrollbars=False, show_scalebars=False)
        return micromed_raw
