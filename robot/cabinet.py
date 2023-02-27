import time
import serial

class CabinetDoorCloser():
    def __init__(self, ):
        self.arduino = serial.Serial(port = '/dev/ttyACM0', baudrate=9600, timeout=0)

    def env_reset(self, unspool_time_ms: int = 4_200):
        self.arduino.write(str.encode(f"{unspool_time_ms}"))
        start = time.time()
        while time.time() - start < unspool_time_ms:
            line = self.arduino.readline()   # read a byte string
            if line:
                string = line.decode()  # convert the byte string to a unicode string
                print(string)

                if "1" in string:
                    break
                elif "2" in string:
                    assert False # something bad is happening
                elif "4" in string:
                    break


if __name__ == "__main__":
    cabinet = CabinetDoorCloser()
    while True:
        res = input("Reset? y/n")
        if res == 'y':
            cabinet.env_reset()
        elif res == 'b':
            break