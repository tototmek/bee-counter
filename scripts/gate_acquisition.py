import serial
import time

serial_port = "/dev/ttyACM0"
baudrate = 115200
output_path = "data/measurement-long.csv"

try:
    ser = serial.Serial(serial_port, baudrate)
    time.sleep(0.2)
    ser.flush()
    ser.read_all()
    with open(output_path, "w") as file:
        file.writelines(["time, left_gate_raw, right_gate_raw\n"])
        while True:
            if ser.in_waiting > 0:
                data = ser.readline().decode("utf-8").strip()
                print(data)
                file.writelines([data + "\n"])
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
except KeyboardInterrupt:
    print("Exiting serial reader.")
finally:
    if "ser" in locals() and ser.is_open:
        ser.close()
