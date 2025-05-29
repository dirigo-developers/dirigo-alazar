import time

from dirigo import units
from dirigo_alazar import AlazarDigitizer


digitizer = AlazarDigitizer()

# Vertical
for i, channel in enumerate(digitizer.channels):
    channel.coupling = "DC"
    channel.impedance = "50 Ω"
    channel.range = units.VoltageRange("±2 V")
    channel.enabled = i < 2

# Horizontal
digitizer.sample_clock.edge = "Rising"
digitizer.sample_clock.source = "Internal Clock"
digitizer.sample_clock.rate = units.SampleRate("10 MS/s")

# Trigger
digitizer.trigger.external_coupling = "DC"
digitizer.trigger.external_range = "TTL"
digitizer.trigger.source = "External"
digitizer.trigger.slope = "Positive"
digitizer.trigger.level = units.Voltage(0)

# Acquire
digitizer.acquire.trigger_delay_samples = 100
digitizer.acquire.timestamps_enabled = True
digitizer.acquire.record_length = 2048
digitizer.acquire.records_per_buffer = 256
digitizer.acquire.buffers_per_acquisition = 16
digitizer.acquire.buffers_allocated = 4

digitizer.acquire.start()
time.sleep(0.02)
digitizer.acquire.stop()

print("success")

