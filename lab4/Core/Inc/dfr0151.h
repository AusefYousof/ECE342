#ifndef __DS1307_H
#define __DS1307_H

#include "main.h"

#define ADDR_DS1307 (uint16_t)(0x68 << 1)
#define ADDR_EEPROM (uint16_t)(0x50 << 1)

uint8_t bcd2bin(uint8_t data);
uint8_t bin2bcd(uint8_t data);

void rtc_init();
uint8_t rtc_read(uint8_t address);
void rtc_write(uint8_t address,uint8_t data);
void rtc_get_time(uint8_t *hour,uint8_t *min,uint8_t *sec);
void rtc_set_time(uint8_t hour,uint8_t min,uint8_t sec);
void rtc_get_date(uint8_t *week_day, uint8_t *day,uint8_t *month,uint8_t *year);
void rtc_set_date(uint8_t week_day, uint8_t day,uint8_t month,uint8_t year);

void eeprom_write(uint16_t address, uint8_t* data, uint16_t size);
uint8_t eeprom_read(uint16_t address);

#endif
