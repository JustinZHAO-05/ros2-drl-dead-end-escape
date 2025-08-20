from igncontactserver import IgnContactsWatcher as IgnContactsWatcher
import time


watcher = IgnContactsWatcher("/world/all_training/model/all_walls_and_cylinders/link/single_link/sensor/sensor_contact/contact")
watcher.start()


for _ in range(100):
    if watcher.collided_recently(0.5):
        print("Collision detected!")
    else:
        print("No collision detected.")
    time.sleep(0.5)

watcher.stop()
print("Watcher stopped.")
