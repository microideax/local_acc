file copy -force /home/adsc/work/local_acc/bitstream/8chfilter/8chfilter.bit .
write_cfgmem -force -format MCS -size 64 -interface SPIx8 -loadbit "up 0x00000000 8chfilter.bit" vcu118_pcie_x16_gen3.mcs
