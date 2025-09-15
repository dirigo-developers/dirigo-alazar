from functools import cached_property
from typing import Optional

import numpy as np
from numba import njit, prange, uint8, int8, int16, uint16, int64

from atsbindings import Ats, System, Buffer
from atsbindings import Board as AlazarBoard

from dirigo import units
from dirigo.hw_interfaces import digitizer  
from dirigo.sw_interfaces.acquisition import AcquisitionProduct


"""
Alazar Tech digitizer implementation for Dirigo.

This module provides a concrete implementation of the `digitizer.Digitizer` API
for Alazar Tech digitizers, using the `atsbindings` library to interface with
the hardware.

Classes:
    AlazarChannel: Configures individual input channels.
    AlazarSampleClock: Configures the sample clock for acquisition.
    AlazarTrigger: Manages trigger settings and operations.
    AlazarAcquire: Handles acquisition logic and data transfer.
    AlazarAuxiliaryIO: Configures auxiliary input/output operations.
    AlazarDigitizer: Combines the above components into a digitizer interface.
"""


class AlazarChannel(digitizer.Channel):
    """
    Configures the parameters for individual input channels on an Alazar Tech 
    digitizer.

    Properties:
        index (int): The index of the channel (0-based).
        coupling (str): Signal coupling mode (e.g., "AC", "DC").
        impedance (str): Input impedance setting (e.g., 50 Ohm, 1 MOhm).
        range (str): Voltage range for the channel.
        enabled (bool): Indicates whether the channel is active for acquisition.
    """
    _coupling_mapping = { # Dirigo enums -> ATS enums
        digitizer.ChannelCoupling.AC:       Ats.Couplings.AC_COUPLING,
        digitizer.ChannelCoupling.DC:       Ats.Couplings.DC_COUPLING,
        digitizer.ChannelCoupling.GROUND:   Ats.Couplings.GND_COUPLING,
    }
    _impedance_mapping = { # Dirigo units -> ATS enums
        units.Resistance("1 Mohm"):     Ats.Impedances.IMPEDANCE_1M_OHM,
        units.Resistance("50 ohm"):     Ats.Impedances.IMPEDANCE_50_OHM,
        units.Resistance("75 ohm"):     Ats.Impedances.IMPEDANCE_75_OHM,
        units.Resistance("300 ohm"):    Ats.Impedances.IMPEDANCE_300_OHM,
        units.Resistance("100 ohm"):    Ats.Impedances.IMPEDANCE_100_OHM,
    }

    def __init__(self, board: AlazarBoard, channel_index: int):
        self._board = board
        self._index = channel_index

        # Set parameters to None to indicate they have not been initialized 
        # (though they are set to something on the digitizer)
        self._coupling: digitizer.ChannelCoupling | None = None
        self._impedance: units.Resistance | None = None
        self._range: units.VoltageRange | None = None
    
    @property
    def index(self) -> int:
        return self._index

    @property
    def coupling(self) -> digitizer.ChannelCoupling:
        if self._coupling is None:
            raise RuntimeError("Coupling not initialized")
        return self._coupling
    
    @coupling.setter
    def coupling(self, coupling: digitizer.ChannelCoupling):
        if coupling not in self.coupling_options:
            raise ValueError(f"Invalid input coupling {coupling}"
                             f"Valid options are: {self.coupling_options}")
        self._coupling = coupling
        self._set_input_control()

    @property
    def coupling_options(self) -> set[digitizer.ChannelCoupling]:
        ats_couplings = self._board.bsi.input_couplings
        rvs_map = {v: k for k, v in self._coupling_mapping.items()}
        return {rvs_map[c] for c in ats_couplings}

    @property
    def impedance(self) -> units.Resistance:
        if self._impedance is None:
            raise RuntimeError("Impedance not initialized")
        return self._impedance
    
    @impedance.setter
    def impedance(self, impedance: units.Resistance):
        if impedance not in self.impedance_options:
            raise ValueError(f"Invalid input impedance {impedance}. "
                             f"Valid options are: {self.impedance_options}")
        self._impedance = impedance
        self._set_input_control()

    @property
    def impedance_options(self) -> set[units.Resistance]:
        ats_impedances = self._board.bsi.input_impedances
        rvs_map = {v: k for k, v in self._impedance_mapping.items()}
        return {rvs_map[i] for i in ats_impedances}
    
    @property
    def range(self) -> units.VoltageRange:
        if self._range is None:
            raise RuntimeError("Range is not initialized")
        return self._range
    
    @range.setter
    def range(self, rng: units.VoltageRange):
        # supported Alazar input ranges are always bipolar
        if abs(rng.max) != abs(rng.min):
            raise ValueError("Voltage range must be bipolar: e.g. +/-1V")
        
        if rng not in self.range_options:
            raise ValueError(f"Invalid input impedance {rng}. "
                             f"Valid options are: {self.range_options}")
        self._range = rng
        self._set_input_control()
    
    @property
    def range_options(self) -> set[units.VoltageRange]:
        if self._impedance is None:
            raise RuntimeError("Impedance must be initialized before accessing range options")
        ats_ranges = self._board.bsi.input_ranges(self._impedance_mapping[self._impedance])
        return {units.VoltageRange(-r.to_volts, r.to_volts) for r in ats_ranges}

    def _set_input_control(self):
        """Helper method to apply input control settings to the digitizer.

        Ensures that all required properties are set before applying settings.
        """
        if self._coupling is None or self._range is None or self._impedance is None:
            return
        else:
            self._board.input_control_ex(
                channel=Ats.Channels.from_int(self.index),
                coupling=self._coupling_mapping[self._coupling],
                input_range=Ats.InputRanges.from_volts(self._range.max),
                impedance=self._impedance_mapping[self._impedance],
            )


class AlazarSampleClock(digitizer.SampleClock):
    """
    Configures the sample clock for an Alazar Tech digitizer.

    This class handles the configuration of the digitizer's sample clock, 
    including the source, rate, and edge settings.

    Properties:
        source (str): The source of the sample clock (e.g., "Internal", "External").
        rate (dirigo.Frequency): The sample clock rate in hertz.
        edge (str): The clock edge to use for sampling (e.g., "Rising", "Falling").

    Note:
        The clock source determines which rates and ranges are valid. Internal 
        clocks use predefined rates, while external clocks can accept user-defined
        frequencies within specific limits.
    """
    
    def __init__(self, board:AlazarBoard):
        self._board = board

        # Set parameters to None to signify that they have not been initialized
        self._source: digitizer.SampleClockSource | None = None
        self._rate: units.SampleRate | None = None
        
        # Default clock edge, set to rising
        self._edge: digitizer.SampleClockEdge = digitizer.SampleClockEdge.RISING
        
    @property
    def source(self) -> digitizer.SampleClockSource:
        if self._source is None:
            raise RuntimeError("Source not initialized")
        return self._source
    
    @source.setter
    def source(self, source: digitizer.SampleClockSource):
        if not isinstance(source, digitizer.SampleClockSource):
            raise ValueError("Sample clock source must be set with a SampleClockSource enumeration.")
        if source not in self.source_options:
            raise ValueError(f"{source} (sample clock source) is not available")
        self._source = source

    @property
    def source_options(self) -> set[digitizer.SampleClockSource]:
        options = []
        for ats_source in [str(s).lower() for s in self._board.bsi.supported_clocks]:
            if "internal" in ats_source:
                options.append(digitizer.SampleClockSource.INTERNAL)
            elif "external" in ats_source:
                options.append(digitizer.SampleClockSource.EXTERNAL)
        return set(options)

    @property
    def rate(self) -> units.SampleRate:
        """
        Depending on the clock source, either the internal sample clock rate, or
        the user-specified external clock rate.
        """
        if self._source is None:
            raise RuntimeError("`source` must be initialized before accessing rate")
        if self._rate is None:
            raise RuntimeError("Sample clock rate is not initialized")
        return self._rate
    
    @rate.setter
    def rate(self, rate: units.SampleRate):
        if self._source is None:
            raise ValueError("`source` must be set before attempting to set `rate`")

        if self._source == digitizer.SampleClockSource.INTERNAL:
            # Check if proposed rate matches an available internal clock rate
            clock_rate_enum = Ats.SampleRates.from_hertz(int(rate))
            if clock_rate_enum not in self._board.bsi.sample_rates:
                valid_options = ', '.join([str(s) for s in self._board.bsi.sample_rates])
                raise ValueError(f"Invalid sample clock rate: {clock_rate_enum}. "
                                 f"Valid options are: {valid_options}")
            self._rate = rate
            self._set_capture_clock()

        elif self._source == digitizer.SampleClockSource.EXTERNAL:
            # check that the proposed external clock rate is achievable
            valid_range = self.rate_options
            if valid_range.min < rate < valid_range.max:
                self._rate = rate
                return

            raise ValueError(f"Proposed externally clocked sample rate: {rate} " 
                             f"is not within any external clock range: "
                             f"min: {valid_range.min}, max: {valid_range.max}")
            
        else:
            raise RuntimeError(f"Invalid sample clock source: {self._source}")

    @property
    def rate_options(self) -> set[units.SampleRate] | units.SampleRateRange:
        if self._source is None:
            raise RuntimeError("`source` must be set before attempting to set `rate`")
        
        if self._source == digitizer.SampleClockSource.INTERNAL:
            return {units.SampleRate(option.to_hertz) for option in self._board.bsi.sample_rates}
        
        if self._source == digitizer.SampleClockSource.EXTERNAL:
            min_rate, max_rate = float('inf'), -float('inf')
            for ats_clock in self._board.bsi.supported_clocks:
                if ats_clock == Ats.ClockSources.INTERNAL_CLOCK: continue

                valid_range = self._board.bsi.external_clock_frequency_ranges(ats_clock)
                min_rate = min(min_rate, valid_range.min)
                max_rate = max(max_rate, valid_range.max)
                            
            return units.SampleRateRange(min=min_rate, max=max_rate)
        
        else:
            raise RuntimeError(f"Unsupported source: {self._source}")
    
    @property
    def edge(self) -> digitizer.SampleClockEdge:
        # has a default so no need to check for None
        return self._edge
    
    @edge.setter
    def edge(self, edge: digitizer.SampleClockEdge):
        if edge not in self.edge_options:
            raise ValueError(f"Proposed clock edge: {edge} not an available "
                             f"option {self.edge_options}")
        self._edge = edge
        self._set_capture_clock()

    @property
    def edge_options(self) -> set[digitizer.SampleClockEdge]:
        # ALl non-DES boards support rising/falling edge sampling
        options = [digitizer.SampleClockEdge.RISING, 
                   digitizer.SampleClockEdge.FALLING]
        return {str(s) for s in options}

    def _set_capture_clock(self):
        """
        Helper to set capture clock if all required parameters have been set:
        source, rate, and edge (has a default).
        """
        if (self._source is not None) and (self._rate is not None):
            if self._source == digitizer.SampleClockSource.INTERNAL:
                ats_source = Ats.ClockSources.INTERNAL_CLOCK
                ats_rate = Ats.SampleRates.from_hertz(int(self._rate))

            elif self._source == digitizer.SampleClockSource.EXTERNAL:
                for ats_clock in self._board.bsi.supported_clocks:
                    if ats_clock == Ats.ClockSources.INTERNAL_CLOCK: 
                        continue
                    valid_range = self._board.bsi.external_clock_frequency_ranges(ats_clock)
                    if valid_range.min <= self._rate <= valid_range.max:
                        ats_source = ats_clock
                        break
                ats_rate = Ats.SampleRates.SAMPLE_RATE_USER_DEF
            
            else:
                raise RuntimeError(f"Unsupported source: {self._source}")
            
            if self._edge == digitizer.SampleClockEdge.RISING:
                ats_edge = Ats.ClockEdges.CLOCK_EDGE_RISING
            elif self._edge == digitizer.SampleClockEdge.FALLING:
                ats_edge = Ats.ClockEdges.CLOCK_EDGE_FALLING
            else:
                raise RuntimeError(f"Unsupported edge: {self._edge}")

            self._board.set_capture_clock(ats_source, ats_rate, ats_edge)


class AlazarTrigger(digitizer.Trigger):
    """
    Configures triggering behavior for an Alazar Tech digitizer.

    This class manages trigger settings, including source, slope, level, and 
    external coupling. It supports both internal and external trigger sources.

    Properties:
        source (str): The trigger source (e.g., "Channel A", "External").
        slope (str): The trigger slope (e.g., "Positive", "Negative").
        level (dirigo.Voltage): The trigger level in volts.
        external_coupling (str): Coupling mode for the external trigger source (e.g., "DC").
        external_range (str): Voltage range for the external trigger source.

    Note:
        Trigger source and settings must be compatible with the enabled channels
        or external trigger specifications.
    """
    _trigger_source_mapping = {
        digitizer.TriggerSource.INTERNAL:   Ats.TriggerSources.TRIG_DISABLE, # Not sure about this
        digitizer.TriggerSource.EXTERNAL:   Ats.TriggerSources.TRIG_EXTERNAL,
        digitizer.TriggerSource.CHANNEL_A:  Ats.TriggerSources.TRIG_CHAN_A,
        digitizer.TriggerSource.CHANNEL_B:  Ats.TriggerSources.TRIG_CHAN_B,
        digitizer.TriggerSource.CHANNEL_C:  Ats.TriggerSources.TRIG_CHAN_C,
        digitizer.TriggerSource.CHANNEL_D:  Ats.TriggerSources.TRIG_CHAN_D,
    }
    _trigger_slope_mapping = {
        digitizer.TriggerSlope.RISING:  Ats.TriggerSlopes.TRIGGER_SLOPE_POSITIVE,
        digitizer.TriggerSlope.FALLING: Ats.TriggerSlopes.TRIGGER_SLOPE_NEGATIVE,
    }
    _external_coupling_map = {
        digitizer.ExternalTriggerCoupling.AC:   Ats.Couplings.AC_COUPLING,
        digitizer.ExternalTriggerCoupling.DC:   Ats.Couplings.DC_COUPLING
    }

    def __init__(self, board: AlazarBoard, channels: tuple[AlazarChannel, ...]):
        self._board = board
        self._channels = channels

        # Set parameters to None to signify that they have not been initialized
        self._source: digitizer.TriggerSource | None = None
        self._slope: digitizer.TriggerSlope | None = None
        self._external_coupling: digitizer.ExternalTriggerCoupling | None = None
        self._external_range: units.VoltageRange | digitizer.ExternalTriggerRange | None = None
        self._level: units.Voltage = units.Voltage("0 V") # level is an 8-bit value in ATSApi

    @property
    def source(self) -> digitizer.TriggerSource:
        if self._source is None:
            raise RuntimeError("Source not initialized")
        return self._source
    
    @source.setter
    def source(self, source: digitizer.TriggerSource):
        if source not in self.source_options:
            raise ValueError(f"Invalid trigger source: {source}. "
                             f"Valid options are: {self.source_options}")
        self._source = source
        self._set_trigger_operation()

    @property
    def source_options(self) -> set[digitizer.TriggerSource]:
        ats_trig_sources = self._board.bsi.supported_trigger_sources

        # remove channels that are not currently enabled
        for channel in self._channels:
            if not channel.enabled:
                s = f"channel {chr(channel.index + ord('A'))}"
                ats_trig_sources.remove(Ats.TriggerSources.from_str(s))
        
        rvs_map = {v: k for k, v in self._trigger_source_mapping.items()}
        return {rvs_map[t] for t in ats_trig_sources}

    @property
    def slope(self) -> digitizer.TriggerSlope:
        if self._slope is None:
            raise RuntimeError("Slope not initialized")
        return self._slope
    
    @slope.setter
    def slope(self, slope: digitizer.TriggerSlope):
        if slope not in self.slope_options:
            raise ValueError(f"Invalid trigger slope: {slope}. "
                             f"Valid options: {self.slope_options}")
        self._slope = slope
        self._set_trigger_operation()

    @property
    def slope_options(self) -> set[digitizer.TriggerSlope]:
        return {digitizer.TriggerSlope.RISING, digitizer.TriggerSlope.FALLING}

    @property
    def level(self) -> units.Voltage:
        if self._source is None:
            raise RuntimeError("Source not initialized")
        if self._level is None:
            raise RuntimeError("Trigger level not initialized")
        
        trigger_channel = self._channels[self._source.channel_index]
        if trigger_channel._range is None:
            raise RuntimeError("Trigger channel not initialized")
        trig_source_range = trigger_channel._range.to_volts
        return units.Voltage((self._level - 128) * trig_source_range / 127)
        
    @level.setter
    def level(self, level: units.Voltage):
        if self._source is None:
            raise RuntimeError("Trigger source must be set before trigger level")
        if self._source == digitizer.TriggerSource.INTERNAL:
            raise RuntimeError("Cannot set trigger level. Trigger is disabled")
        if self._source == digitizer.TriggerSource.EXTERNAL:
            if self._external_range is None:
                raise RuntimeError("External range not initialized")
            if self._external_range == digitizer.ExternalTriggerRange.TTL:
                trigger_source_range = units.VoltageRange(min="0 V", max="5 V")
            else:
                trigger_source_range = self._external_range 
        else:
            raise NotImplementedError("Haven't completed trigger level for input channels")
            # trigger_channel = self._channels[self._source.channel_index]
            # if trigger_channel._range is None:
            #     raise RuntimeError("Level range not set")
            # trigger_source_range = trigger_channel._range.to_volts
        
        if not trigger_source_range.within_range(level):
            raise ValueError(f"Trigger level, {level} is outside the current trigger source range")

        #self._level = 
        self._level = level
        self._set_trigger_operation()

    @property
    def level_limits(self) -> units.VoltageRange:
        if self._source is None:
            raise RuntimeError("Source not initialized")
        
        trigger_channel_sources = {
            Ats.TriggerSources.from_str(f"Channel {chr(i + ord('A'))}") 
            for i in range(len(self._channels))
        }
        
        if self._source == Ats.TriggerSources.TRIG_EXTERNAL:
            if self._external_range is None:
                raise RuntimeError("External trigger range not initialized")
            trigger_source_range = self._external_range.to_volts # TODO need to allow for offset range like TTL

        elif self._source in trigger_channel_sources: 
            source_channel = self._channels[self._source.channel_index]
            if source_channel._range is None:
                raise RuntimeError("Tirgger source channel range not initialized")
            trigger_source_range = source_channel._range.to_volts

        return units.VoltageRange(
            min=-abs(trigger_source_range),
            max=abs(trigger_source_range)
        )

    @property
    def external_coupling(self) -> digitizer.ExternalTriggerCoupling:
        if self._external_coupling is None:
            raise RuntimeError("External trigger coupling not initialized")
        return self._external_coupling

    @external_coupling.setter
    def external_coupling(self, external_coupling: digitizer.ExternalTriggerCoupling):
        if external_coupling not in self.external_coupling_options:
            raise ValueError(f"Unsupported external trigger coupling mode: {external_coupling}"
                             f"Supported: {self.external_coupling_options}")
        self._external_coupling = external_coupling
        self._set_external_trigger()

    @property
    def external_coupling_options(self) -> set[digitizer.ExternalTriggerCoupling]:
        # Only support DC external trigger. Only a few old boards support AC.
        return {digitizer.ExternalTriggerCoupling.DC,}

    @property
    def external_range(self) -> units.VoltageRange | digitizer.ExternalTriggerRange:
        if self._external_range is None:
            raise ValueError("External range not initialized")
        return self._external_range
    
    @external_range.setter
    def external_range(self, rng: units.VoltageRange | digitizer.ExternalTriggerRange):
        if rng not in self.external_range_options:
            raise ValueError(f"Invalid external trigger range: {rng}. "
                             f"Valid options are: {self.external_range_options}")
        self._external_range = rng
        self._set_external_trigger()

    @property
    def external_range_options(self) -> set[units.VoltageRange | digitizer.ExternalTriggerRange]:
        ranges = []
        for ats_range in self._board.bsi.external_trigger_ranges:
            if ats_range == Ats.ExternalTriggerRanges.ETR_TTL:
                ranges.append(digitizer.ExternalTriggerRange.TTL)
            elif ats_range == Ats.ExternalTriggerRanges.ETR_1V_50OHM:
                ranges.append(units.VoltageRange("-1 V", "1 V"))
            elif ats_range == Ats.ExternalTriggerRanges.ETR_2V5_50OHM:
                ranges.append(units.VoltageRange("-2.5 V", "2.5 V"))
            elif ats_range == Ats.ExternalTriggerRanges.ETR_5V_50OHM:
                ranges.append(units.VoltageRange("-5 V", "5 V"))
            else:
                raise RuntimeError(f"Encountered unsupported external trigger range: {ats_range}")
        return set(ranges)
        
    def _set_trigger_operation(self):
        """
        Helper to set trigger operation if all required parameters have been set.
        By default, uses trigger engine J and disables engine K
        """
        if (self._source is not None) and (self._slope is not None):
            if self._source == digitizer.TriggerSource.EXTERNAL:
                if self._external_range is None:
                    raise RuntimeError("External range not initialized")
                if self._external_range == digitizer.ExternalTriggerRange.TTL:
                    trigger_source_range = units.VoltageRange(min="0 V", max="5 V")
                else:
                    trigger_source_range = self._external_range 
            else:
                raise NotImplementedError("Haven't completed trigger level for input channels")
            int_lvl = int(128 + 127 * (self._level / trigger_source_range.range))
            
            self._board.set_trigger_operation(
                operation=Ats.TriggerOperations.TRIG_ENGINE_OP_J,
                engine1=Ats.TriggerEngines.TRIG_ENGINE_J,
                source1=self._trigger_source_mapping[self._source],
                slope1=self._trigger_slope_mapping[self._slope],
                level1=int_lvl,
                engine2=Ats.TriggerEngines.TRIG_ENGINE_K,
                source2=Ats.TriggerSources.TRIG_DISABLE,
                slope2=Ats.TriggerSlopes.TRIGGER_SLOPE_POSITIVE,
                level2=0
            )

    def _set_external_trigger(self):
        """
        Helper method to set external trigger parameters.
        """
        if (self._external_coupling is not None) and (self._external_range is not None):
            if isinstance(self._external_range, units.VoltageRange):
                if self._external_range.max == units.Voltage("1 V"):
                    rng = Ats.ExternalTriggerRanges.ETR_1V_50OHM
                elif self._external_range.max == units.Voltage("2.5 V"):
                    rng = Ats.ExternalTriggerRanges.ETR_2V5_50OHM
                elif self._external_range.max == units.Voltage("5 V"):
                    rng = Ats.ExternalTriggerRanges.ETR_5V_50OHM
                else:
                    raise RuntimeError(f"Unexpected external trigger range: {self._external_range}")
            elif self._external_range == digitizer.ExternalTriggerRange.TTL:
                rng = Ats.ExternalTriggerRanges.ETR_TTL
            else:
                raise RuntimeError(f"Unexpected external trigger range: {self._external_range}")

            self._board.set_external_trigger(
                coupling=self._external_coupling_map[self._external_coupling], 
                range=rng
            )


@njit((uint8[:,:,:], int64), parallel=True, fastmath=True, nogil=True, cache=True)
def fix_alazar_inplace_8(buf: np.ndarray, right_shift: int):
    """
    In-place: offset-binary uint8  →  twos-complement uint8 pattern.

    After the call, do   signed = buf.view(np.int8)   (zero-copy).
    """
    nr, ns, nc = buf.shape
    rs = right_shift
    if rs < 0:
        rs = 0
    elif rs > 7:
        rs = 7

    for r in prange(nr):
        for s in range(ns):
            base = buf[r, s]
            for c in range(nc):
                v = base[c] ^ 0x80              # remove offset
                if rs != 0:
                    v = uint8(int8(v) >> rs)    # arithmetic shift then back to unsigned
                base[c] = v                     # write back uint8


@njit((uint16[:,:,:], int64), parallel=True, fastmath=True, nogil=True, cache=True)
def fix_alazar_inplace_16(buf: np.ndarray, right_shift: int):
    """
    In-place: offset-binary uint16  →  twos-complement uint16 pattern.

    After the call, do   signed = buf.view(np.int16)   (zero-copy).
    """
    nr, ns, nc = buf.shape
    rs = right_shift
    if rs < 0:
        rs = 0
    elif rs > 15:
        rs = 15

    for r in prange(nr):
        for s in range(ns):
            base = buf[r, s]          # 1-D uint16 view of the nc channels
            for c in range(nc):
                v = base[c] ^ 0x8000            # remove offset
                if rs != 0:
                    v = uint16(int16(v) >> rs)  # arithmetic shift then back to unsigned
                base[c] = v                     # write back uint16



class AlazarAcquire(digitizer.Acquire):
    """
    Handles acquisition settings and data transfer for an Alazar Tech digitizer.

    This class manages acquisition parameters, including trigger delay, record 
    length, and buffer management. It also facilitates data transfer using the 
    Alazar Tech API's asynchronous DMA (ADMA) mode.

    Properties:
        trigger_delay_samples (int): Delay between the trigger event and acquistion, in sample periods.
        trigger_delay_duration (dirigo.Time): Delay between the trigger event and acquisition start, in time.
        record_length (int): Number of samples per record.
        record_duration (dirigo.Time): Duration of record acquistion.
        records_per_buffer (int): Number of records per buffer.
        buffers_per_acquisition (int): Total number of buffers to acquire.
        buffers_allocated (int): Number of memory buffers allocated for acquisition.

    Note:
        Ensure that all required settings (e.g., record length, buffer counts) 
        are configured before starting the acquisition.
    """

    def __init__(self, board: AlazarBoard, sample_clock: AlazarSampleClock,
                 channels: tuple[AlazarChannel, ...]):
        self._board = board
        self._sample_clock = sample_clock
        self._channels = channels

        # Set some defaults
        self._pre_trigger_samples: int = 0

        self._trigger_delay: Optional[int] = None # in samples
        self._record_length: Optional[int] = None
        self._records_per_buffer: Optional[int] = None
        self._buffers_per_acquisition: Optional[int] = None # uses -1 to code for unlimited

        self._adma_mode: Ats.ADMAModes = Ats.ADMAModes.ADMA_TRADITIONAL_MODE # TODO allow setting NPT mode
        self._timestamps_enabled: bool = False

        self._buffers_allocated: Optional[int] = None
        self._buffers: Optional[list[Buffer]] = None

    @property
    def trigger_delay_samples(self) -> int:
        if self._trigger_delay is None:
            raise RuntimeError("Trigger delay not initialized")
        return self._trigger_delay

    @trigger_delay_samples.setter
    def trigger_delay_samples(self, samples: int):
        if not isinstance(samples, int):
            raise ValueError("`trigger_delay_samples` must be set with an integer")
        if not (0 <= samples < 9_999_999): # not clear whether this includes 9,999,999 or not, assume not
            raise ValueError(f"`trigger_delay_samples` outside settable range, got {samples}")
        if samples % self.trigger_delay_sample_resolution:
            raise ValueError(
                f"Attempted to set `trigger_delay_samples` {samples}, must"
                f"be divisible by {self.trigger_delay_sample_resolution}."
            )
        self._board.set_trigger_delay(samples)
        self._trigger_delay = samples

    @property
    def trigger_delay_duration(self) -> units.Time:
        if self._trigger_delay is None:
            raise RuntimeError("Trigger delay not initialized")
        return units.Time(self._trigger_delay / self._sample_clock.rate)

    @property
    def trigger_delay_sample_resolution(self) -> int:
        # samples per timestamp resolution is also the resolution for trigger delay
        return self._board.bsi.samples_per_timestamp(self.n_channels_enabled)

    # TODO, combine the next three with trigger_delay
    @property
    def pre_trigger_samples(self) -> int:
        return self._pre_trigger_samples
    
    @pre_trigger_samples.setter
    def pre_trigger_samples(self, samples: int):
        if samples < 0:
            raise ValueError(f"Attempted to set pre-trigger samples {samples} "
                             f"must be ≥ 0")
        pretrig_res = self.pre_trigger_resolution
        if samples % pretrig_res != 0:
            raise ValueError(f"Attempted to set pre-trigger samples {samples}, "
                             f"must be multiple of {pretrig_res}")
        self._pre_trigger_samples = samples
        self._set_record_size()

    @property
    def pre_trigger_resolution(self) -> int:
        return self._board.bsi.pretrig_alignment

    @property
    def record_length(self) -> int:
        if self._record_length is None:
            raise RuntimeError("Record length not initialized")
        return self._record_length

    @record_length.setter
    def record_length(self, length: int):
        if length < self.record_length_minimum:
            raise ValueError(f"Attempted to set record length {length}, must "
                             f"be ≥ {self.record_length_minimum}")
        rec_res = self.record_length_resolution
        if length % rec_res != 0:
            raise ValueError(f"Attempted to set record length {length}, must "
                             f"be multiple of {rec_res}")
        self._record_length = length
        self._set_record_size()

    @property
    def record_duration(self) -> units.Time:
        if self._record_length is None:
            raise RuntimeError("Record length not initialized")
        return units.Time(self._record_length / self._sample_clock.rate)

    @property
    def record_length_minimum(self) -> int:
        return self._board.bsi.min_record_size
    
    @property
    def record_length_resolution(self) -> int:
        return self._board.bsi.record_resolution
    
    @property
    def records_per_buffer(self) -> int:
        if self._records_per_buffer is None:
            raise RuntimeError("Records per buffer not initialized")
        return self._records_per_buffer
    
    @records_per_buffer.setter
    def records_per_buffer(self, records: int):
        if records < 1:
            ValueError(f"Attempted to set records per buffer {records}, "
                       f"must be ≥ 1")
        self._records_per_buffer = records

    @property
    def buffers_per_acquisition(self) -> int:
        if self._buffers_per_acquisition is None:
            raise RuntimeError("Buffers per acquisitions not initialized")
        return self._buffers_per_acquisition
    
    @buffers_per_acquisition.setter
    def buffers_per_acquisition(self, buffers: int):
        if buffers == -1:
            # -1 codes for unlimited buffers per acquisition
            pass
        else:
            if buffers < 1:
                raise ValueError(f"Attempted to set buffers per acquisition "
                                f"{buffers}, must be ≥ 1")

        self._buffers_per_acquisition = buffers

    @property
    def buffers_allocated(self) -> int:
        if self._buffers_allocated is None:
            raise RuntimeError("Buffers allocated not initialized")
        return self._buffers_allocated

    @buffers_allocated.setter
    def buffers_allocated(self, buffers: int):
        buffers = int(buffers)
        if buffers < 1:
            raise ValueError("Tried setting buffer allocation below 1")
        self._buffers_allocated = buffers

    @cached_property
    def _bit_depth(self) -> int: 
        _, bit_depth = self._board.get_channel_info()
        return bit_depth

    def start(self):
        # Check whether essential parameters have been set
        if not self.record_length:
            raise RuntimeError("Must set record length before beginning acquisition")
        if not self.records_per_buffer:
            raise RuntimeError("Must set records per buffer before beginning acquisition")
        if not self.buffers_per_acquisition:
            raise RuntimeError("Must set buffers per acquisition before beginning acquisition")
        # check sample clock and channels?
        
        # Prepare board for acquisition
        self._buffers_acquired = 0
        channels_bit_mask = sum([c.enabled * Ats.Channels.from_int(i) 
                                for i,c in enumerate(self._channels)])
        flags = self._adma_mode | Ats.ADMAFlags.ADMA_EXTERNAL_STARTCAPTURE 
        if self.supports_interleaved:
            flags = flags | Ats.ADMAFlags.ADMA_INTERLEAVE_SAMPLES
        if self.timestamps_enabled:
            if self._adma_mode == Ats.ADMAModes.ADMA_TRADITIONAL_MODE:
                flags = flags | Ats.ADMAFlags.ADMA_ENABLE_RECORD_HEADERS
            elif self._adma_mode == Ats.ADMAModes.ADMA_NPT:
                flags = flags | Ats.ADMAFlags.ADMA_ENABLE_RECORD_FOOTERS

        if self.buffers_per_acquisition == float('inf'):
            records_per_acquisition = 0x7FFFFFFF
        else:
            records_per_acquisition = int(
                self.buffers_per_acquisition * self.records_per_buffer
            )

        self._board.before_async_read(
            channels                = channels_bit_mask,
            transfer_offset         = -self.pre_trigger_samples, # note the neg. value for pre-trigger samples
            transfer_length         = self.record_length,
            records_per_buffer      = self.records_per_buffer,
            records_per_acquisition = records_per_acquisition,
            flags                   = flags
        )

        # Allocate buffers
        if self.timestamps_enabled:
            if self._adma_mode == Ats.ADMAModes.ADMA_TRADITIONAL_MODE:
                headers, footers = True, False
            elif self._adma_mode == Ats.ADMAModes.ADMA_NPT:
                headers, footers = False, True
        else:
            headers, footers = False, False
        self._buffers = []
        for _ in range(self.buffers_allocated): 
            buffer = Buffer(
                board=self._board, 
                channels=self.n_channels_enabled,
                records_per_buffer=self.records_per_buffer,
                samples_per_record=self.record_length,
                include_header=headers,
                include_footer=footers,
                interleave_samples=self.supports_interleaved
            )
            self._buffers.append(buffer)
            self._board.post_async_buffer(buffer.address, buffer.size)

        self._sec_per_tic = \
            self._board.bsi.samples_per_timestamp(self.n_channels_enabled) \
            / self._sample_clock.rate 
        
        self._board.start_capture()

    @property
    def buffers_acquired(self) -> int:
        return self._buffers_acquired

    def get_next_completed_buffer(self, acq_buf: AcquisitionProduct): 
        """Retrieve the next available buffer"""
        if self._buffers is None:
            raise RuntimeError("Buffers not initialized")
        # Determine the index of the buffer, retrieve reference
        buffer_index = self._buffers_acquired % self.buffers_allocated
        buffer = self._buffers[buffer_index]

        # Wait for the buffer to complete and copy data when ready--want this to be long
        self._board.wait_async_buffer_complete(buffer.address)

        buffer.get_data(acq_buf.data)

        # ATS API returns offset unsigned 8 or 16 bit data, fully scaled to 8 or 
        # 16 bits regardless of the digitizer bit depth. Fix this before passing.
        if acq_buf.data.dtype == np.uint8:
            fix_alazar_inplace_8(acq_buf.data, 8 - self._bit_depth)
            acq_buf.data.dtype = np.int8 # type: ignore
        else:
            fix_alazar_inplace_16(acq_buf.data, 16 - self._bit_depth)
            acq_buf.data.dtype = np.int16 # type: ignore

        # Retrieve timestamps
        acq_buf.timestamps = self._sec_per_tic * np.array(buffer.get_timestamps())
        self._buffers_acquired += 1

        # Repost buffer
        self._board.post_async_buffer(buffer.address, buffer.size)

    def stop(self):
        self._board.abort_async_read()

    @property
    def adma_mode(self) -> str: # TODO, should we use Alazar-specific enumeration here?
        """Alazar-specific: returns current ADMA mode"""
        return str(self._adma_mode)

    @adma_mode.setter
    def adma_mode(self, new_adma_mode: str):
        """Set ADMA mode (default without using setter is 'Traditional')"""
        self._adma_mode = Ats.ADMAModes.from_str(str(new_adma_mode))

    @property
    def timestamps_enabled(self) -> bool:
        """
        Enables hardware timestamps. 

        In Traditional ADMA mode, enables headers. In NPT ADMA mode, enables
        footers.
        """
        return self._timestamps_enabled
    
    @timestamps_enabled.setter
    def timestamps_enabled(self, enable: bool):
        supported_modes = {Ats.ADMAModes.ADMA_TRADITIONAL_MODE, Ats.ADMAModes.ADMA_NPT}
        if enable and self._adma_mode not in supported_modes:
            raise ValueError("Timestamps are only available in Traditional or NPT AMDA mode.")
        self._timestamps_enabled = enable

    def _set_record_size(self):
        """Helper"""
        if self._pre_trigger_samples and self._record_length:
            self._board.set_record_size(
                pre_trigger_samples=self._pre_trigger_samples, 
                post_trigger_samples=self._record_length)
            
    @cached_property
    def supports_interleaved(self) -> bool:
        """Returns whether interleaved acquisition is supported."""
        # Reference:
        # https://docs.alazartech.com/ats-sdk-user-guide/latest/reference/AlazarBeforeAsyncRead.html#c.ALAZAR_ADMA_FLAGS.ADMA_INTERLEAVE_SAMPLES
        unsupported_boards = {
            Ats.BoardType.ATS310, # all 3-digit boards are PCI (not PCIe)
            Ats.BoardType.ATS315,
            Ats.BoardType.ATS330,
            Ats.BoardType.ATS335,
            Ats.BoardType.ATS460,
            Ats.BoardType.ATS660,
            Ats.BoardType.ATS665,
            Ats.BoardType.ATS850,
            Ats.BoardType.ATS855,
            Ats.BoardType.ATS860,
            Ats.BoardType.ATS9462 # and ATS9462 (see reference)
        }
        if self._board.get_board_kind() in unsupported_boards:
            return False
        else:
            return True


class AlazarAuxiliaryIO(digitizer.AuxiliaryIO):
    """
    Configures auxiliary input/output (IO) operations for an Alazar Tech digitizer.

    This class allows control over auxiliary IO modes, such as triggering,
    pacer clock output, or auxiliary signal input monitoring.
    """

    def __init__(self, board: AlazarBoard):
        self._board = board
        self._mode: Optional[Ats.AuxIOModes] = None

    def configure_mode(self, mode: digitizer.AuxiliaryIOEnums, **kwargs):
        """
        Usage:
        for mode AuxiliaryIOEnums.OutTrigger, no keyword arg required,
        for mode AuxiliaryIOEnums.OutPacer, provide 'divider' keyword arg (int),
        for mode AuxiliaryIOEnums.OutDigital, provide 'state' keyword arg (bool),
        for mode AuxiliaryIOEnums.InTriggerEnable, provide 'slope' keyword arg (digitizer.TriggerSlope),
        for mode AuxiliaryIOEnums.InDigital, no keyword arg required.
        """
        if mode == digitizer.AuxiliaryIOEnums.OutTrigger:
            self._board.configure_aux_io(
                mode        = Ats.AuxIOModes.AUX_OUT_TRIGGER, 
                parameter   = 0 # have to provide, but not used
            )

        elif mode == digitizer.AuxiliaryIOEnums.OutPacer:
            divider = int(kwargs["divider"])
            self._board.configure_aux_io(
                mode        = Ats.AuxIOModes.AUX_OUT_PACER, 
                parameter   = divider
            )

        elif mode == digitizer.AuxiliaryIOEnums.OutDigital:
            state = bool(kwargs.get('state'))
            self._board.configure_aux_io(
                mode        = Ats.AuxIOModes.AUX_OUT_SERIAL_DATA, 
                parameter   = state
            )

        elif mode == digitizer.AuxiliaryIOEnums.InTriggerEnable:
            slope = kwargs["slope"]
            if slope == digitizer.TriggerSlope.RISING:
                ats_slope = Ats.TriggerSlopes.TRIGGER_SLOPE_POSITIVE
            elif slope == digitizer.TriggerSlope.FALLING:
                ats_slope = Ats.TriggerSlopes.TRIGGER_SLOPE_NEGATIVE
            else:
                raise ValueError(f"Unspported trigger slope: {slope}")
            self._board.configure_aux_io(
                mode        = Ats.AuxIOModes.AUX_IN_TRIGGER_ENABLE, 
                parameter   = ats_slope
            )

        elif mode == digitizer.AuxiliaryIOEnums.InDigital:
            self._board.configure_aux_io(
                mode        = Ats.AuxIOModes.AUX_IN_AUXILIARY, 
                parameter   = 0
            )

        else:
            raise ValueError(f"Unsupported auxiliary IO mode: {mode}")
        
    # TODO provide supported modes?
        
    def read_input(self) -> bool:
        if self._mode == Ats.AuxIOModes.AUX_IN_AUXILIARY:
            return self._board.get_parameter(
                Ats.Channels.CHANNEL_ALL, 
                Ats.Parameters.GET_AUX_INPUT_LEVEL
            ) is True
        else:
            raise RuntimeError("Auxiliary IO not configured as input.")
        
    def write_output(self, state: bool):
        self.configure_mode(Ats.AuxIOModes.AUX_OUT_SERIAL_DATA, state=state)


class AlazarDigitizer(digitizer.Digitizer):
    """
    Combines all components into a complete digitizer interface for Alazar Tech hardware.

    This class provides a unified interface to configure and operate an Alazar 
    Tech digitizer, integrating channel, sample clock, trigger, acquisition, 
    and auxiliary IO settings.

    Args:
        system_id (int): The system ID of the digitizer (default: 1).
        board_id (int): The board ID of the digitizer (default: 1).

    Attributes:
        channels (list[AlazarChannel]): List of configured input channels.
        sample_clock (AlazarSampleClock): Sample clock configuration.
        trigger (AlazarTrigger): Trigger configuration.
        acquire (AlazarAcquire): Acquisition settings and logic.
        aux_io (AlazarAuxiliaryIO): Auxiliary input/output configuration.

    Note:
        Ensure the digitizer hardware is correctly connected and initialized
        before creating an instance of this class.
    """

    def __init__(self, system_id: int = 1, board_id: int = 1, **kwargs):
        self.input_mode = digitizer.InputMode.ANALOG    # Alazar cards are analog-only
        self.streaming_mode = digitizer.StreamingMode.TRIGGERED # Only triggered modes supported at this time

        # Check system
        nsystems = System.num_of_systems()
        if nsystems < 1:
            raise RuntimeError("No board systems found. At least one is required.")
        nboards = System.boards_in_system_by_system_id(system_id)
        if nboards < 1: # not sure this is actually possible 
            raise RuntimeError("No boards found. At least one is required.")
        
        self.driver_version = System.get_driver_version()
        self.dll_version = System.get_sdk_version() 

        self._board = AlazarBoard(system_id, board_id)

        chan_list = []
        for i in range(self._board.bsi.channels):
            chan_list.append(AlazarChannel(self._board, i))

        self.channels: tuple[AlazarChannel, ...] = tuple(chan_list)

        self.sample_clock: AlazarSampleClock = AlazarSampleClock(self._board)

        self.trigger: AlazarTrigger = AlazarTrigger(self._board, self.channels)

        self.acquire: AlazarAcquire = AlazarAcquire(self._board, self.sample_clock, self.channels)
        
        self.aux_io: AlazarAuxiliaryIO = AlazarAuxiliaryIO(self._board)

    @property
    def bit_depth(self) -> int: 
        return self.acquire._bit_depth

    @cached_property
    def data_range(self) -> units.IntRange:
        return units.IntRange(
            min=-2**(self.bit_depth-1),
            max=2**(self.bit_depth-1) - 1 
        )

