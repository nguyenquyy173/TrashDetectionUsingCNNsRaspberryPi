import minimalmodbus
import time
import serial

# --- 1. CONFIGURATION PARAMETERS ---
PORT = "/dev/ttyUSB0"
SLAVE_ID = 1
BAUDRATE = 9600

# --- 2. INSTRUMENT SETUP (Aggressive Timing and Framing Fixes) ---
try:
    inst = minimalmodbus.Instrument(PORT, SLAVE_ID)
    
    # HMI REQUIRED SETTINGS (9600, 8, EVEN, 1) - CONFIRMED BY IMAGE
    inst.serial.baudrate = BAUDRATE
    inst.serial.bytesize = 8
    inst.serial.parity = serial.PARITY_EVEN  
    inst.serial.stopbits = 1
    
    # *** FINAL CRITICAL TIMING AND FRAMING ADJUSTMENTS ***
    inst.serial.timeout = 1.0        
    inst.interchar_timeout = 0.5     # ?? INCREASED SIGNIFICANTLY (0.5s) for Pi stability
    inst.mode = minimalmodbus.MODE_RTU
    
    inst.close_port_after_each_call = True  # Guarantees RTU 3.5T gap
    inst.skip_gabage_bytes = True           # ?? NEW: Instructs library to ignore leading trash data
    
    # Disable flow control (standard practice)
    inst.serial.rtscts = False 
    inst.serial.dsrdtr = False 
    inst.serial.xonxoff = False 
    
    time.sleep(0.2) 
    
except Exception as e:
    print(f"FATAL: Failed to initialize MinimalModbus Instrument: {e}")
    exit()

print(f"? Port: {PORT} (ID: {SLAVE_ID}, Settings: {BAUDRATE}, 8, EVEN, 1) - Final Code Attempt")

# --- 3. MODBUS HELPER FUNCTIONS (unchanged, as the error is in the setup) ---

def set_LW(addr, val):
    print(f"Attempting to write {val} to LW{addr}...")
    for attempt in range(1, 4):
        try:
            # Write and then immediately read back
            inst.write_register(addr, int(val), functioncode=16)
            read_val = inst.read_register(addr, functioncode=3)
            
            if read_val == int(val):
                print(f"   SUCCESS: LW{addr} set to {read_val} (Attempt {attempt})")
                return True
            else:
                print(f"   Verification FAILED: Wrote {val}, Read back {read_val}")
                
        except Exception as e:
            print(f"   Error writing LW{addr} on attempt {attempt}: {e}")
            time.sleep(0.5)
            
    print(f"? FAILED to reliably write to LW{addr} after 3 attempts.")
    return False

def get_LW(addr):
    val = None
    for attempt in range(1, 4):
        try:
            val = inst.read_register(addr, functioncode=3)
            print(f"   READ: LW{addr} value is {val} (Attempt {attempt})")
            return val
            
        except Exception as e:
            print(f"   Error reading LW{addr} on attempt {attempt}: {e}")
            time.sleep(0.5)
            
    print(f"? FAILED to reliably read from LW{addr} after 3 attempts.")
    return None

# --- 4. MAIN EXECUTION BLOCK ---

if __name__ == '__main__':
    # Write Cycle
    if set_LW(0, 1): time.sleep(0.1)
    if set_LW(0, 2): time.sleep(0.1)
    if set_LW(0, 3): time.sleep(0.1)

    # Read Cycle
    print("\nStarting Read Cycle:")
    get_LW(0)
    time.sleep(0.1) 
    get_LW(0)
    time.sleep(0.1) 
    get_LW(0)
    
    print("\nExecution complete.")
