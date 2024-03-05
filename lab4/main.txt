/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
#include "main.h"
#include "dfr0151.h"
#include "config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>



int random = 0;


int getRandomValue(int min, int max) {
		srand(random);
    // Generate a random value within the specified range this is for part 2 
    uint8_t randomValue = rand() % (max - min + 1) + min;
    return randomValue;
}


int main(void)
{
  /* Reset of all peripherals. */
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_I2C1_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();  
  rtc_init();

	
	//init some values and ease of access array and respective addresses to eeprom read from
	
	uint8_t sec = 0, min = 0, hour = 10, week_day = 1, day = 1, month = 7, year = 24; //init time specified in part 4
	//store pointers to the values
	uint8_t* values[7] = {&sec, &min, &hour, &week_day, &day, &month, &year};
	uint16_t addresses[7] = {0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006};
	
	//same for alarm
	uint8_t al_sec, al_min, al_hour, al_week_day, al_day, al_month, al_year;
	uint8_t* val_alarm[7] = {&al_sec, &al_min, &al_hour, &al_week_day, &al_day, &al_month, &al_year};
	uint16_t addrs_alarm[7] = {0x0007, 0x0008, 0x0009, 0x000A, 0x000B, 0x000C, 0x000D};
	
	//read alarm from eeprom memory
	for(int i = 0; i < 7; i++){
		*val_alarm[i] = eeprom_read(addrs_alarm[i]);
		HAL_Delay(11);
	}
	
	//is alarm on
	int alarm = 0;

  char msg[100];
	
	/*
	char startup[100];
	
	//part 3 testing, read from eeprom and set that to current time (time was written to eeprom off b1 press)
	for(int i = 0; i < 7; i++){
		*values[i] = eeprom_read(addresses[i]);
		HAL_Delay(11);
	}

	sprintf(startup, "EEPROM_READ UPON STARTUP: week_day: %d, day: %d, month: %d, year: %d, hour: %d min: %d, sec: %d\n", week_day, day, month, year, hour, min, sec);
	print_msg(startup);
	*/
	

	//read from eeprom on startup see if anything saved
	
	//set initial time
	rtc_set_time(hour, min ,sec);
	rtc_set_date(week_day,day,month,year);
	
  while (1)
  {
		
		//for rand seed w/o including time
		/*
		random+=1;
		if (random > 150){
			random = 0;
		}
		*/
		if(al_sec == sec && al_min == min){ //small check to see if alarm went off (can also check all other factors)
			alarm = 1;
		}

		if (alarm!=0){
			for(int i = 0; i < 10; i++){
				print_msg("ALARM\n");
				HAL_GPIO_TogglePin(LD1_GPIO_Port, LD1_Pin);
				HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
				HAL_GPIO_TogglePin(LD3_GPIO_Port, LD3_Pin);
				HAL_Delay(1000);
			}
			alarm = 0;
		}
		
    // Your code here
    HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
    HAL_Delay(500);
    //print_msg("\b\b\b\btick");

    HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
    HAL_Delay(500);
    //print_msg("\b\b\b\btock");
		
		
		//FOR TESTING PART 2(RANDOM PART)
		/*
		//part 2, sets all values to random upon button press
		if (HAL_GPIO_ReadPin(LD1_GPIO_Port,LD1_Pin) == GPIO_PIN_SET){
			//set random 
			sec = getRandomValue(0, 59);
			min = getRandomValue(0, 59);
			hour = getRandomValue(0, 23);
			
			rtc_set_time(hour, min, sec);
			
			week_day = getRandomValue(1, 7);
			day = getRandomValue(1, 31);
			month = getRandomValue(1, 12);
			year = getRandomValue(0,99);
			
			rtc_set_date(week_day, day, month, year);
		
			HAL_GPIO_WritePin(LD1_GPIO_Port,LD1_Pin, GPIO_PIN_RESET);
		}
		*/
		
		
		
		//part3, save all values to EEPROM upon button press
		/*
		if (HAL_GPIO_ReadPin(LD1_GPIO_Port,LD1_Pin) == GPIO_PIN_SET){ 
			for(int i = 0; i < 7; i++){
				eeprom_write(addresses[i], values[i], 1);
				HAL_Delay(11);
			}
			HAL_GPIO_WritePin(LD1_GPIO_Port,LD1_Pin, GPIO_PIN_RESET);
		}
		*/
		
		//part4/demo part when b1 pressed set time to july 1st 2024 10:00am and set alarm for one minute later
		if (HAL_GPIO_ReadPin(LD1_GPIO_Port,LD1_Pin) == GPIO_PIN_SET){
			//reset to 10:00 am on july1st 2024
			uint8_t sec = 0, min = 0, hour = 10, week_day = 1, day = 1, month = 7, year = 24;
			rtc_set_date(week_day, day, month, year);
			rtc_set_time(hour, min, sec);
			
			//set alarm to a minute after, also write to eeprom
			
			al_sec = 0;
			al_min = 1;
			al_hour = 10;
			al_week_day = 1;
			al_day = 1;
			al_month = 7;
			al_year = 24;
			
			for(int i = 0; i < 7; i++){
				eeprom_write(addrs_alarm[i], val_alarm[i], 1);
				HAL_Delay(11);
			}
			
			HAL_GPIO_WritePin(LD1_GPIO_Port,LD1_Pin, GPIO_PIN_RESET);
		}
		
		
		
		//GENERAL TESTING AND OUTPUT OF CLOCK & ALARM
		
		rtc_get_time(&hour, &min, &sec);
		sprintf(msg, "Time = %d:%d:%d        ", hour, min, sec);
		print_msg(msg);	
		rtc_get_date(&week_day,&day,&month,&year);
		sprintf(msg, "     Date: %d/%d/20%d on %dth day of the week\n", month, day, year, week_day);
		print_msg(msg);
		
		
		sprintf(msg, "Alarm Time = %d:%d:%d        ", al_hour, al_min, al_sec);
		print_msg(msg);	
		sprintf(msg, "Alarm Date: %d/%d/20%d on %dth day of the week\n", al_month, al_day, al_year, al_week_day);
		print_msg(msg);
		
  }
}