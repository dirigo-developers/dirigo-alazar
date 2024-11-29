import time

from dirigo_alazar import AlazarDigitizer


digitizer = AlazarDigitizer()

# Vertical
for i, channel in enumerate(digitizer.channels):
    channel.coupling = "DC"
    channel.impedance = "50 Ω"
    channel.range = "±2 V"
    channel.enabled = i < 2

# Horizontal
digitizer.sample_clock.edge = "Rising"
digitizer.sample_clock.source = "Internal Clock"
digitizer.sample_clock.rate = "10 MS/s"

# Trigger
digitizer.trigger.external_coupling = "DC"
digitizer.trigger.external_range = "TTL"
digitizer.trigger.source = "External"
digitizer.trigger.slope = "Positive"
digitizer.trigger.level = 0

# Acquire
digitizer.acquire.pre_trigger_samples = 0
digitizer.acquire.record_length = 2048
digitizer.acquire.records_per_buffer = 256
digitizer.acquire.buffers_per_acquisition = 4

digitizer.acquire.start()
time.sleep(0.02)
digitizer.acquire.stop()

print("success")

