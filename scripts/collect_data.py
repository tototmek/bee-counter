#!/usr/bin/env python3
import serial
import time
import serial.tools.list_ports
import os
from typing import Optional
import sys

# Import Rich library components
from rich.console import Console
from rich.table import Table
from rich.live import Live


def test_device_port(port, baudrate) -> (bool, str):
    try:
        ser = serial.Serial(port, baudrate)
        ser.close()
        return True, "port is ok"
    except serial.SerialException as e:
        return False, e.strerror


def detect_device_port(baudrate) -> Optional[str]:
    """Scans for an available serial device."""
    for serial_port in serial.tools.list_ports.comports():
        try:
            # A simple check to see if the port can be opened
            ser = serial.Serial(serial_port.device, baudrate)
            ser.close()
            return serial_port.device
        except serial.SerialException:
            continue
    return None


def str_or_not(value: int, unit: str) -> str:
    """Helper to format time units only if value is greater than zero."""
    return f"{str(value)}{unit}" if value > 0 else ""


def format_time(time_val: float) -> str:
    """Formats total seconds into a compact string like 1d5h3m12s."""
    seconds = round(time_val)
    days, r = divmod(seconds, 86400)
    hours, r = divmod(r, 3600)
    minutes, seconds = divmod(r, 60)
    return (
        str_or_not(days, "d")
        + str_or_not(hours, "h")
        + str_or_not(minutes, "m")
        + str(seconds)
        + "s"
    )


def update_symlink(output_path, symlink_path):
    abs_output_path = os.path.abspath(output_path)
    if os.path.lexists(symlink_path):  # If a symlink already exists, remove it first
        os.remove(symlink_path)
    os.symlink(abs_output_path, symlink_path)  # Create the new symbolic link


# --- Configuration ---
N = 8
BAUDRATE = 115200
COLUMN_NAMES = ["time"] + [f"delta{i}" for i in range(N)]
RATE_UPDATE_INTERVAL_S = 1.0  # How often to update the Hz calculation

if len(sys.argv) > 1:
    device_port = sys.argv[1]
    port_ok, message = test_device_port(device_port, BAUDRATE)
    if not port_ok:
        print(f"Device connection failed: {message}")
        device_port = None
        exit(1)

else:
    device_port = None

# --- File Setup ---
root_dir_name = os.path.join("data", "experiments")
dir_name = os.path.join(root_dir_name, time.strftime("%Y-%m-%d"))
os.makedirs(dir_name, exist_ok=True)
output_path = os.path.join(dir_name, f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv")

# --- Rich Console ---
console = Console()

# --- Main Logic ---
console.print("Searching for device...")
if device_port is None:
    device_port = detect_device_port(BAUDRATE)
if device_port is None:
    console.print("Bee counter not detected", style="bold red")
    exit(1)

console.print(f"Found bee counter at [green]{device_port}[/green].")
console.print(f"Starting data stream...")

try:
    ser = serial.Serial(device_port, BAUDRATE)
    time.sleep(0.1)
    ser.flushInput()

    with open(output_path, "w", newline="") as file:
        file.write(",".join(COLUMN_NAMES) + "\n")
        start_time = time.time()

        # Variables for rate calculation
        line_count = 0
        last_rate_time = time.time()
        rate_hz = 0.0

        with Live(
            console=console,
            screen=False,
            auto_refresh=False,
            vertical_overflow="visible",
        ) as live:
            while True:
                if ser.in_waiting > 0:
                    # 1. Read and process data
                    data_str = ser.readline().decode("utf-8").strip()
                    data_values = data_str.split(",")
                    elapsed = format_time(time.time() - start_time)
                    line_count += 1

                    # 2. Write data to the CSV file
                    file.write(data_str + "\n")

                    # 3. Calculate the rate periodically
                    current_time = time.time()
                    if (current_time - last_rate_time) >= RATE_UPDATE_INTERVAL_S:
                        rate_hz = line_count / (current_time - last_rate_time)
                        line_count = 0
                        last_rate_time = current_time

                    # 4. Build the Rich table for display
                    table = Table(
                        border_style="blue",
                        title=f"🐝 Receiving data at [bold yellow]{rate_hz:.2f} Hz[/bold yellow],,,,,,,,,"
                    )
                    table.add_column("PC elapsed", style="cyan", justify="right")
                    table.add_column("device time", style="cyan", justify="right")
                    for name in COLUMN_NAMES[1:]:
                        table.add_column(name, style="magenta", justify="right")

                    # 5. Add the new row of data to the table
                    if len(data_values) == len(COLUMN_NAMES):
                        table.add_row(elapsed, *data_values)
                    else:
                        table.add_row(elapsed, "[red]Waiting for data...[/red]")

                    # 6. Update the live display
                    live.update(table, refresh=True)

except serial.SerialException as e:
    console.print(f"\nError opening serial port: {e}", style="bold red")
except KeyboardInterrupt:
    pass
finally:
    if "ser" in locals() and ser.is_open:
        ser.close()
update_symlink(output_path, root_dir_name + "/last.csv")
print(f"\nOutput saved to {output_path}")
print(f"Done.")
