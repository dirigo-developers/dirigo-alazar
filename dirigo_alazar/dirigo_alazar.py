from functools import cached_property

import numpy as np
from atsbindings import Ats, System, Buffer
from atsbindings import Board as AlazarBoard

from dirigo import units
from dirigo.hw_interfaces import digitizer


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
    AlazarAuxillaryIO: Configures auxiliary input/output operations.
    AlazarDigitizer: Combines the above components into a digitizer interface.
"""


class AlazarChannel(digitizer.Channel):
    """
    Configures the parameters for individual input channels on an Alazar Tech 
    digitizer.

    Properties:
        index (int): The index of the channel (0-based).
        coupling (str): Signal coupling mode (e.g., "AC", "DC").
        impedance (str): Input impedance setting (e.g., "50 Ohm", "1 MOhm").
        range (str): Voltage range for the channel.
        enabled (bool): Indicates whether the channel is active for acquisition.
    """
    def __init__(self, board: AlazarBoard, channel_index: int):
        self._board = board
        self._index = channel_index
        self._enabled = False # Disabled as default

        # Set parameters to None to indicate they have not been initialized 
        # (though they are set to something on the digitizer)
        self._coupling: Ats.Couplings = None
        self._impedance: Ats.Impedances = None
        self._range: Ats.InputRanges = None
    
    @property
    def index(self) -> int:
        return self._index

    @property
    def coupling(self) -> str:
        if self._coupling:
            return str(self._coupling)
    
    @coupling.setter
    def coupling(self, coupling: str):
        coupling_enum = Ats.Couplings.from_str(coupling)
        if coupling_enum not in self._board.bsi.input_couplings:
            valid = self._board.bsi.input_couplings
            raise ValueError(f"Invalid input coupling {coupling_enum}"
                             f"Valid options are: {valid}")
        self._coupling = coupling_enum
        self._set_input_control()

    @property
    def coupling_options(self) -> set[str]:
        options = self._board.bsi.input_couplings
        return {str(s) for s in options}

    @property
    def impedance(self) -> str:
        if self._impedance:
            return str(self._impedance)
    
    @impedance.setter
    def impedance(self, impedance: str):
        impedance_enum = Ats.Impedances.from_str(impedance)
        if impedance_enum not in self._board.bsi.input_impedances:
            valid_options = ', '.join([str(s) for s in self._board.bsi.input_impedances])
            raise ValueError(f"Invalid input impedance {impedance_enum}. "
                             f"Valid options are: {valid_options}")
        self._impedance = impedance_enum
        self._set_input_control()

    @property
    def impedance_options(self) -> set[str]:
        options = self._board.bsi.input_impedances
        return {str(s) for s in options}
    
    @property
    def range(self) -> str:
        if self._range:
            return str(self._range)
    
    @range.setter
    def range(self, rng: str):
        range_enum = Ats.InputRanges.from_str(rng)
        current_ranges = self._board.bsi.input_ranges(self._impedance)
        if range_enum not in current_ranges:
            valid_options = ', '.join([str(s) for s in current_ranges])
            raise ValueError(f"Invalid input impedance {range_enum}. "
                             f"Valid options are: {valid_options}")
        self._range = range_enum
        self._set_input_control()
    
    @property
    def range_options(self) -> set[str]:
        if self._impedance:
            options = self._board.bsi.input_ranges(self._impedance)
            return {str(s) for s in options}

    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, enabled: bool):
        self._enabled = enabled

    def _set_input_control(self):
        """Helper method to apply input control settings to the digitizer.

        Ensures that all required properties are set before applying settings.
        """
        if self.coupling and self.impedance and self._range:
            self._board.input_control_ex(
                channel=Ats.Channels.from_int(self.index),
                coupling=self._coupling,
                input_range=self._range,
                impedance=self._impedance,
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
        self._source: Ats.ClockSources = None
        self._rate: Ats.SampleRates = None
        self._external_rate: units.Frequency = None # Used only with external clock source, otherwise ignored
        
        # Default clock edge, set to rising
        self._edge: Ats.ClockEdges = Ats.ClockEdges.CLOCK_EDGE_RISING
        
    @property
    def source(self) -> str:
        if self._source:
            return str(self._source)
    
    @source.setter
    def source(self, source: str):
        previous_source_enum = self._source
        source_enum = Ats.ClockSources.from_str(source)

        # Check if same, if so return immediately
        if source_enum == previous_source_enum:
            return 
        
        # Check whether new source is supported, if so store in private attr
        if source_enum not in self._board.bsi.supported_clocks:
            valid_options = ', '.join([str(s) for s in self._board.bsi.supported_clocks])
            raise ValueError(f"Invalid sample clock source: {source_enum}. "
                             f"Valid options are: {valid_options}")
        self._source = source_enum

        # Reset rate
        if source_enum == Ats.ClockSources.INTERNAL_CLOCK:
            self._rate = None
        elif "external" in str(self._source).lower():
            self._rate = Ats.SampleRates.SAMPLE_RATE_USER_DEF
            self._external_rate = None
        # TODO, other sources?

        #self._set_capture_clock() -- why commented?

    @property
    def source_options(self) -> set[str]:
        return {str(s) for s in self._board.bsi.supported_clocks}

    @property
    def rate(self) -> units.SampleRate:
        """
        Depending on the clock source, either the internal sample clock rate, or
        the user-specified external clock rate.
        """
        if self._source == Ats.ClockSources.INTERNAL_CLOCK:
            if self._rate:
                # convert atswrapper enum into a dirigo.Frequency object
                return units.SampleRate(self._rate.to_hertz) 
        elif "external" in str(self._source).lower():
            if self._external_rate:
                return str(self._external_rate) 
    
    @rate.setter
    def rate(self, rate: units.SampleRate):
        if self.source is None:
            raise ValueError("`source` must be set before attempting to set `rate`")

        if self._source == Ats.ClockSources.INTERNAL_CLOCK:
            clock_rate_enum = Ats.SampleRates.from_str(rate)
            if clock_rate_enum not in self._board.bsi.sample_rates:
                valid_options = ', '.join([str(s) for s in self._board.bsi.sample_rates])
                raise ValueError(f"Invalid sample clock rate: {clock_rate_enum}. "
                                f"Valid options are: {valid_options}")
            self._rate = clock_rate_enum
            self._set_capture_clock()

        elif "external" in str(self._source).lower():
            # check that the proposed external clock is valid
            proposed_rate = float(rate)
            valid_range = self._board.bsi.external_clock_frequency_ranges(self._source)
            if valid_range.min < proposed_rate < valid_range.max:
                self._external_rate = proposed_rate
            else:
                raise ValueError(f"Tried setting external clock frequency outside "
                                 f"acceptable range for source: {self._source} "
                                 f"Requested: {proposed_rate}, "
                                 f"Min: {valid_range.min}"
                                 f"Max: {valid_range.max}")

    @property
    def rate_options(self) -> set[units.SampleRate] | units.FrequencyRange:
        if self._source == Ats.ClockSources.INTERNAL_CLOCK:
            return {units.SampleRate(option.to_hertz) for option in self._board.bsi.sample_rates}
        
        elif "external" in str(self._source).lower():
            return units.FrequencyRange(
                **self._board.bsi.external_clock_frequency_ranges(self._source)
            ) # TODO, haven't tested this conversion
    
    @property
    def edge(self) -> str:
        if self._edge:
            return str(self._edge)
    
    @edge.setter
    def edge(self, edge: str):
        clock_edge_enum = Ats.ClockEdges.from_str(edge)
        self._edge = clock_edge_enum
        self._set_capture_clock()

    @property
    def edge_options(self) -> set[str]:
        options = [Ats.ClockEdges.CLOCK_EDGE_RISING, # ALl non-DES boards support rising/falling edge sampling
                   Ats.ClockEdges.CLOCK_EDGE_FALLING]
        return {str(s) for s in options}

    def _set_capture_clock(self):
        """
        Helper to set capture clock if all required parameters have been set:
        source, rate, and edge
        """
        if self._source and self._rate and self._edge:
            self._board.set_capture_clock(self._source, self._rate, self._edge)


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

    def __init__(self, board: AlazarBoard, channels: list[AlazarChannel]):
        self._board = board
        self._channels = channels

        # Set parameters to None to signify that they have not been initialized
        self._source: Ats.TriggerSources = None
        self._slope: Ats.TriggerSlopes = None
        self._external_coupling: Ats.Couplings = None
        self._external_range: Ats.ExternalTriggerRanges = None
        self._level: int = None # level is an 8-bit value in ATSApi

    @property
    def source(self) -> str:
        if self._source and str(self._source) in self.source_options:
            # If _source (an enum) exists and if it is currently a valid source option
            return str(self._source)
    
    @source.setter
    def source(self, source: str):
        source_enum = Ats.TriggerSources.from_str(source)
        trig_srcs = self._board.bsi.supported_trigger_sources
        if source_enum not in trig_srcs:
            valid_options = ', '.join([str(s) for s in trig_srcs])
            raise ValueError(f"Invalid trigger source: {source_enum}. "
                             f"Valid options are: {valid_options}")
        self._source = source_enum
        self._set_trigger_operation()

    @property
    def source_options(self) -> set[str]:
        all_options = self._board.bsi.supported_trigger_sources

        # remove channels that are not currently enabled
        for channel in self._channels:
            if not channel.enabled:
                s = f"channel {chr(channel.index + ord('A'))}"
                all_options.remove(Ats.TriggerSources.from_str(s))
        
        # may want to remove 'Disable' option, which would require SW trigger
        return {str(s) for s in all_options}

    @property
    def slope(self) -> str:
        if self._slope:
            return str(self._slope)
    
    @slope.setter
    def slope(self, slope: str):
        self._slope = Ats.TriggerSlopes.from_str(slope)
        self._set_trigger_operation()

    @property
    def slope_options(self) -> set[str]:
        options = [Ats.TriggerSlopes.TRIGGER_SLOPE_POSITIVE,
                   Ats.TriggerSlopes.TRIGGER_SLOPE_NEGATIVE]
        return {str(s) for s in options}

    @property
    def level(self) -> units.Voltage:
        #TODO switch for external trigger?
        if self.source and self._level:
            trig_source_range = self._channels[self._source.channel_index]._range.to_volts
            return units.Voltage((self._level - 128) * trig_source_range / 127)
        
    @level.setter
    def level(self, level: units.Voltage):
        if not self._source:
            raise RuntimeError("Trigger source must be set before trigger level")
        if self._source == Ats.TriggerSources.TRIG_DISABLE:
            raise RuntimeError("Cannot set trigger level. Trigger is disabled")
        if self._source == Ats.TriggerSources.TRIG_EXTERNAL:
            trigger_source_range = self._external_range.to_volts 
        else:
            trigger_source_range = self._channels[self._source.channel_index]._range.to_volts
        if abs(level) > trigger_source_range:
            raise ValueError(f"Trigger level, {level} is outside the current trigger source range")

        self._level = int(128 + 127 * level / trigger_source_range)
        self._set_trigger_operation()

    @property
    def level_limits(self) -> units.VoltageRange:
        if not self.source:
            return None
        
        trigger_channel_sources = {
            Ats.TriggerSources.from_str(f"Channel {chr(i + ord('A'))}") 
            for i in range(len(self._channels))
        }
        
        if self._source == Ats.TriggerSources.TRIG_EXTERNAL:
            trigger_source_range = self._external_range.to_volts # TODO need to allow for offset range like TTL

        elif self._source in trigger_channel_sources: 
            source_channel = self._channels[self._source.channel_index]
            trigger_source_range = source_channel._range.to_volts

        return units.VoltageRange(
            min=-abs(trigger_source_range),
            max=abs(trigger_source_range)
        )

    @property
    def external_coupling(self) -> str:
        return str(self._external_coupling)

    @external_coupling.setter
    def external_coupling(self, external_coupling: str):
        external_coupling_enum = Ats.Couplings.from_str(external_coupling)
        self._external_coupling = external_coupling_enum
        self._set_external_trigger()

    @property
    def external_coupling_options(self) -> set[str]:
        # Only support DC external trigger. Only a few old boards support AC.
        options = [Ats.Couplings.DC_COUPLING]
        return {str(s) for s in options} # leave the comprehension in place in case we revise

    @property
    def external_range(self) -> str:
        return str(self._external_range)
    
    @external_range.setter
    def external_range(self, range: str):
        external_range_enum = Ats.ExternalTriggerRanges.from_str(range)
        supported_ranges = self._board.bsi.external_trigger_ranges
        if external_range_enum not in supported_ranges:
            valid_options = ', '.join([str(s) for s in supported_ranges])
            raise ValueError(f"Invalid trigger source: {external_range_enum}. "
                             f"Valid options are: {valid_options}")
        self._external_range = external_range_enum
        self._set_external_trigger()

    @property
    def external_range_options(self) -> set[str]:
        options = self._board.bsi.external_trigger_ranges
        return [str(s) for s in options]
        
    def _set_trigger_operation(self):
        """
        Helper to set trigger operation if all required parameters have been set.
        By default, uses trigger engine J and disables engine K
        """
        if self._source and self._slope and self._level:
            self._board.set_trigger_operation(
                operation=Ats.TriggerOperations.TRIG_ENGINE_OP_J,
                engine1=Ats.TriggerEngines.TRIG_ENGINE_J,
                source1=self._source,
                slope1=self._slope,
                level1=self._level,
                engine2=Ats.TriggerEngines.TRIG_ENGINE_K,
                source2=Ats.TriggerSources.TRIG_DISABLE,
                slope2=Ats.TriggerSlopes.TRIGGER_SLOPE_POSITIVE,
                level2=0
            )

    def _set_external_trigger(self):
        """
        Helper method to set external trigger parameters.
        """
        if self._external_coupling and self._external_range:
            self._board.set_external_trigger(
                self._external_coupling, 
                self._external_range
            )


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
                 channels: list[AlazarChannel]):
        self._board = board
        self._sample_clock = sample_clock
        self._channels = channels

        # Set some defaults
        self._trigger_delay: int = 0 # in number samples
        self._pre_trigger_samples: int = 0

        self._record_length: int = None
        self._records_per_buffer: int = None
        self._buffers_per_acquisition: int | float = None

        self._adma_mode: Ats.ADMAModes = Ats.ADMAModes.ADMA_TRADITIONAL_MODE # TODO allow setting NPT mode

        self._buffers_allocated: int = None
        self._buffers: list[Buffer] = None

    @property
    def trigger_delay_samples(self) -> int:
        return self._trigger_delay

    @trigger_delay_samples.setter
    def trigger_delay_samples(self, samples: int):
        if not isinstance(samples, int):
            raise ValueError("`trigger_delay_samples` must be set with an integer")
        if not samples % self.trigger_delay_sample_resolution:
            raise ValueError(
                f"Attempted to set `trigger_delay_samples` {samples}, must"
                f"be divisible by {self.trigger_delay_sample_resolution}."
            )
        self._trigger_delay = samples

    @property
    def trigger_delay_duration(self) -> units.Time:
        return units.Time(self._trigger_delay / self._sample_clock.rate)
    
    # @property
    # def trigger_delay_duration(self, delay: units.Time):
    #     if delay < 0:
    #         raise ValueError("Trigger delay must be non-negative.")
        
    #     delay_res = self.trigger_delay_resolution

    #     if delay % delay_res != 0:
    #         raise ValueError(f"Attempted to set trigger delay {delay}, must"
    #                          f"be divisible by {delay_res}")
    #     self._trigger_delay = delay * self._sample_clock.rate # stored privately in number samples
    #     self._board.set_trigger_delay(self._trigger_delay)

    @property
    def trigger_delay_sample_resolution(self) -> units.Time:
        # samples per timestamp resolution is also the resolution for trigger delay
        spts = self._board.bsi.samples_per_timestamp(self.n_channels_enabled)
        return units.Time(spts / self._sample_clock.rate)

    # TODO, combine the next three with trigger_delay
    @property
    def pre_trigger_samples(self):
        return self._pre_trigger_samples
    
    @pre_trigger_samples.setter
    def pre_trigger_samples(self, samples:int):
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
    def pre_trigger_resolution(self):
        return self._board.bsi.pretrig_alignment

    @property
    def record_length(self) -> int:
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
        return units.Time(self._record_length / self._sample_clock.rate)

    @property
    def record_length_minimum(self) -> int:
        return self._board.bsi.min_record_size
    
    @property
    def record_length_resolution(self) -> int:
        return self._board.bsi.record_resolution
    
    @property
    def records_per_buffer(self) -> int:
        return self._records_per_buffer
    
    @records_per_buffer.setter
    def records_per_buffer(self, records: int):
        if records < 1:
            ValueError(f"Attempted to set records per buffer {records}, "
                       f"must be ≥ 1")
        self._records_per_buffer = records

    @property
    def buffers_per_acquisition(self) -> int:
        return self._buffers_per_acquisition
    
    @buffers_per_acquisition.setter
    def buffers_per_acquisition(self, buffers: int | float):
        if isinstance(buffers, float) and not buffers == float('inf'):
            raise ValueError("`buffers_per_acquisition` can not be non-infinite float.")
        if buffers < 1:
            raise ValueError(f"Attempted to set buffers per acquisition {buffers}, "
                             f"must be ≥ 1")

        self._buffers_per_acquisition = buffers

    @property
    def buffers_allocated(self) -> int:
        return self._buffers_allocated

    @buffers_allocated.setter
    def buffers_allocated(self, buffers: int):
        buffers = int(buffers)
        if buffers < 1:
            raise ValueError("Tried setting buffer allocation below 1")
        self._buffers_allocated = buffers

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

        if self.buffers_per_acquisition == float('inf'):
            records_per_acquisition = 0x7FFFFFFF
        else:
            records_per_acquisition = int(
                self.buffers_per_acquisition * self.records_per_buffer
            )

        self._board.before_async_read(
            channels=channels_bit_mask,
            transfer_offset=-self.pre_trigger_samples, # note the neg. value for pre-trigger samples
            samples_per_record=self.record_length,
            records_per_buffer=self.records_per_buffer,
            records_per_acquisition=records_per_acquisition,
            flags=flags
        )

        # Allocate buffers
        self._buffers = []
        for _ in range(self.buffers_allocated): 
            buffer = Buffer(
                board=self._board, 
                channels=self.n_channels_enabled,
                records_per_buffer=self.records_per_buffer,
                samples_per_record=self.record_length,
                include_header=True,
                interleave_samples=self.supports_interleaved
            )
            self._buffers.append(buffer)
            self._board.post_async_buffer(buffer.address, buffer.size)
        
        self._board.start_capture()

    @property
    def buffers_acquired(self) -> int:
        return self._buffers_acquired

    def get_next_completed_buffer(self, blocking: bool = True) -> np.ndarray: # TODO a non-blocking version of this
        # Determine the index of the buffer, retrieve reference
        buffer_index = self._buffers_acquired % self.buffers_allocated
        buffer = self._buffers[buffer_index]

        # Wait for the buffer to complete and copy data when ready
        self._board.wait_async_buffer_complete(buffer.address)
        tmp_data = buffer.get_data() # TODO, return headers
        self._buffers_acquired += 1

        # Repost buffer
        self._board.post_async_buffer(buffer.address, buffer.size)

        return tmp_data
        
    def stop(self):
        self._board.abort_async_read()

    @property
    def adma_mode(self) -> str:
        """Alazar-specific: returns current ADMA mode"""
        return str(self._adma_mode)

    @adma_mode.setter
    def adma_mode(self, new_adma_mode: str):
        """Set ADMA mode (default without using setter is 'Traditional')"""
        self._adma_mode = Ats.ADMAModes.from_str(str(new_adma_mode))

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


class AlazarAuxillaryIO(digitizer.AuxillaryIO):
    """
    Configures auxiliary input/output (IO) operations for an Alazar Tech digitizer.

    This class allows control over auxiliary IO modes, such as triggering,
    pacer clock output, or auxiliary signal input monitoring.
    """

    def __init__(self, board: AlazarBoard):
        self._board = board
        self._mode: Ats.AuxIOModes = None

    def configure_mode(self, mode: Ats.AuxIOModes, **kwargs):
        if mode == Ats.AuxIOModes.AUX_OUT_TRIGGER:
            self._board.configure_aux_io(mode, 0)

        elif mode == Ats.AuxIOModes.AUX_OUT_PACER:
            divider = int(kwargs.get('divider'))
            self._board.configure_aux_io(mode, divider)

        elif mode == Ats.AuxIOModes.AUX_OUT_SERIAL_DATA:
            state = bool(kwargs.get('state'))
            self._board.configure_aux_io(mode, state)

        elif mode == Ats.AuxIOModes.AUX_IN_TRIGGER_ENABLE:
            slope:Ats.TriggerSlopes = kwargs.get('slope')
            self._board.configure_aux_io(mode, slope)

        elif mode == Ats.AuxIOModes.AUX_IN_AUXILIARY:
            self._board.configure_aux_io(mode, 0)
            # Note, read requires a call to board.get_parameter()

        else:
            raise ValueError(f"Unsupported auxillary IO mode: {mode}")
        
    # TODO provide supported modes?
        
    def read_input(self) -> bool:
        if self._mode == Ats.AuxIOModes.AUX_IN_AUXILIARY:
            return self._board.get_parameter(
                Ats.Channels.CHANNEL_ALL, 
                Ats.Parameters.GET_AUX_INPUT_LEVEL
            )
        else:
            raise RuntimeError("Auxillary IO not configured as input.")
        
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
        aux_io (AlazarAuxillaryIO): Auxiliary input/output configuration.

    Note:
        Ensure the digitizer hardware is correctly connected and initialized
        before creating an instance of this class.
    """

    def __init__(self, system_id: int = 1, board_id: int = 1, **kwargs):
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

        self.channels = []
        for i in range(self._board.bsi.channels):
            self.channels.append(AlazarChannel(self._board, i))

        self.sample_clock = AlazarSampleClock(self._board)

        self.trigger = AlazarTrigger(self._board, self.channels)

        self.acquire = AlazarAcquire(self._board, self.sample_clock, self.channels)
        
        self.aux_io = AlazarAuxillaryIO(self._board)


