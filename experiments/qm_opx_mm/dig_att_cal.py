from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()

na = im['NWA']
att = im['atten']

plt.figure(figsize=(8, 6))
for i in arange(0.0, 31.25, 0.5):
    att.set_attenuator(i)
    time.sleep(2)
    delta_F = 5.0e9
    na.set_center_frequency(6e9)
    na.set_span(delta_F)
    time.sleep(2)
    tr = na.take_one_averaged_trace()
    plt.plot(tr[0], tr[1], label='att = %.2 dB'%i)
plt.show()