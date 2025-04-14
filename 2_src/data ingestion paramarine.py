import subprocess
import socket
import time

# === CONFIGURATION ===
PARAMARINE_PATH = r"C:\Program Files\QinetiQ\Paramarine V24.1\bin\Paramarine.exe"
PORT = 5000
HOST = "127.0.0.1"

# === FUNCTION TO START PARAMARINE ===
def start_paramarine():
    cmd = f'"{PARAMARINE_PATH}" /port:{PORT}'
    print(f"▶️ Launching Paramarine with:\n{cmd}")
    subprocess.Popen(cmd, shell=True)
    print("⏳ Waiting for Paramarine to initialize...")

# === FUNCTION TO WAIT FOR SOCKET TO BE READY ===
def wait_for_paramarine(timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((HOST, PORT), timeout=2):
                print("✅ Paramarine socket is ready.")
                return
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(1)
    raise RuntimeError("❌ Timeout: Paramarine socket never became available.")

# === FUNCTION TO SEND A SINGLE KCL COMMAND ===
def send_kcl(command: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall((command + "\n").encode("utf-8"))  # use newline (LF)
        response = s.recv(4096).decode("utf-8")
        print(f">>> {command}")
        print(f"<<< {response}")
        return response

# === SAFETY TEST: SEND A BASIC COMMAND ===
def open_design_only():
    start_paramarine()
    wait_for_paramarine()

    print("🕒 Waiting extra time after socket becomes available...")
    time.sleep(10)  # Allow full GUI/init to complete

    # Send a simple KCL command that should never crash
    send_kcl("get_number_selected_objects")

# === RUN ===
if __name__ == "__main__":
    open_design_only()