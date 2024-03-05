#include "dfr0151.h"
#include <stdio.h>
#include <string.h>
#include "config.h"

extern I2C_HandleTypeDef hi2c1;


uint8_t bcd2bin(uint8_t data){
 return ((data >> 4) * 10) + (data & 0x0F);
}

uint8_t bin2bcd(uint8_t data){
  return ((data / 10) << 4)|(data % 10);
}

uint8_t rtc_read(uint8_t address)
{
  uint8_t data;

  if (HAL_I2C_Mem_Read(&hi2c1,ADDR_DS1307,address,I2C_MEMADD_SIZE_8BIT,&data,1,100) != HAL_OK) {
    Error_Handler();
  }

  return data;
}

void rtc_write(uint8_t address,uint8_t data)
{
  if (HAL_I2C_Mem_Write(&hi2c1,ADDR_DS1307,address,I2C_MEMADD_SIZE_8BIT,&data,1,100) != HAL_OK){
    Error_Handler();
  }
}

void rtc_init()
{
  // Initialize Real-Time Clock peripheral
  uint8_t rs=0, sqwe=1, out=0;
  
  rs&=3;
  if (sqwe) rs|=0x10;
  if (out) rs|=0x80;

  rtc_write(0x07,rs);
}

void rtc_get_time(uint8_t *hour,uint8_t *min,uint8_t *sec)
{
	uint8_t arr[3];
	uint8_t data;
	for (uint8_t i = 0; i < 3; i++){
		if (HAL_I2C_Mem_Read(&hi2c1,ADDR_DS1307,bin2bcd(i),I2C_MEMADD_SIZE_8BIT,&data,1,100) != HAL_OK) {
			Error_Handler();
		}
		arr[i] = data;
	}
	*sec = arr[0];
	*min = arr[1];
	*hour = arr[2];
}

void rtc_set_time(uint8_t hour,uint8_t min,uint8_t sec)
{
	uint8_t values[3] = {sec, min, hour};
	for(uint8_t i = 0; i < 3; i++){
		if (HAL_I2C_Mem_Write(&hi2c1,ADDR_DS1307,bin2bcd(i),I2C_MEMADD_SIZE_8BIT,&values[i],1,100) != HAL_OK){
			Error_Handler();
		}
	}
}

void rtc_get_date(uint8_t *week_day,uint8_t *day,uint8_t *month,uint8_t *year)
{
  uint8_t arr[4];
	uint8_t data;
	for (uint8_t i = 3; i < 7; i++){
		if (HAL_I2C_Mem_Read(&hi2c1,ADDR_DS1307,bin2bcd(i),I2C_MEMADD_SIZE_8BIT,&data,1,100) != HAL_OK) {
			Error_Handler();
		}
		arr[i-3] = data;
	}
	*week_day = arr[0];
	*day = arr[1];
	*month = arr[2];
	*year = arr[3];
	
}

void rtc_set_date(uint8_t week_day,uint8_t day,uint8_t month,uint8_t year)
{
  uint8_t values[4] = {week_day, day, month, year};
	for(uint8_t i = 3; i < 7; i++){
		if (HAL_I2C_Mem_Write(&hi2c1,ADDR_DS1307,bin2bcd(i),I2C_MEMADD_SIZE_8BIT,&values[i-3],1,100) != HAL_OK){
			Error_Handler();
		}
	}
}

void eeprom_write(uint16_t address, uint8_t *data, uint16_t size) {
		if (HAL_I2C_Mem_Write(&hi2c1, ADDR_EEPROM, address, I2C_MEMADD_SIZE_16BIT, data, size, HAL_MAX_DELAY) != HAL_OK){
			Error_Handler();
		}
}

uint8_t eeprom_read(uint16_t address) {
    uint8_t data;
    if (HAL_I2C_Mem_Read(&hi2c1, ADDR_EEPROM, address, I2C_MEMADD_SIZE_16BIT, &data, 1, HAL_MAX_DELAY) != HAL_OK){
			Error_Handler();
		}
    return data;
}