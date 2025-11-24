import minimalmodbus
import time

PORT = "/dev/ttyUSB0"
SLAVE_ID = 1

inst = minimalmodbus.Instrument(PORT, SLAVE_ID)
inst.serial.baudrate = 9600
inst.serial.bytesize = 8
inst.serial.parity   = minimalmodbus.serial.PARITY_EVEN
inst.serial.stopbits = 1
inst.serial.timeout  = 0.5
inst.mode = minimalmodbus.MODE_RTU

def AddressLW(val, addr):
	for _ in range(3):
		try:
			inst.write_register(addr, int(val), functioncode=16)
			temp = inst.read_register(addr)
			print(f'Value at LW{addr}: {temp}')
		except Exception as e:
			print("Error write LW: ", e)
			time.sleep(0.5)
			
def AddressReadLB(addr):
	val = -1
	for _ in range(3):
		try:
			val = inst.read_bit(addr, functioncode=1)
			print(f'Value at LB{addr}: {val}')
		except Exception as e:
			print('Error read LB: ', e)
			time.sleep(0.5)
			
	return val
	
def AddressWriteLB():
	for _ in range(3):
		try:
			inst.write_bit(0, 0, functioncode=5)
			inst.write_bit(1, 0, functioncode=5)
			inst.write_bit(2, 0, functioncode=5)
			inst.write_bit(3, 0, functioncode=5)
			#inst.write_bit(5,0, functioncode=5)
			print(f'write at LB successfully')
		except Exception as e:
			print('cannot write at LB')
			time.sleep(0.5)

if __name__== '__main__':
	print('opening port:', PORT)
	AddressLW(11, 4)
	AddressReadLB(4)
	AddressWriteLB()
	#print(AddressLB(0))
