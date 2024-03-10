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
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "ov7670.h"


/* USER CODE BEGIN PV */
#define PREAMBLE "\r\n!START!\r\n"
#define DELTA_PREAMBLE "\r\n!DELTA!\r\n"
#define SUFFIX "!END!\r\n"
//N = 5 VIDEO COMPRESSION ALGO
#define FULL_FRAME_INTERVAL 5

uint16_t snapshot_buff[IMG_ROWS * IMG_COLS];
uint8_t old_snapshot_buff[IMG_ROWS * IMG_COLS];
//SAVE LENGTH TO SAVE CPU CYCLES
int snapshot_length = IMG_ROWS * IMG_COLS;


uint8_t tx_buff[sizeof(PREAMBLE) - 1 + 2 * IMG_ROWS * IMG_COLS + sizeof(SUFFIX) - 1];
size_t tx_buff_len = 0;

//DMA INT FLAG 
uint8_t dma_flag = 0;

//FRAME COUNT FOR VIDEO COMPRESSION ALGO
int frame_count = 0;


// Your function definitions here
void print_buf(void);


///////HELPER FOR PART 3.5
void send_image() {
  // Send image data through serial port.
  // Your code here
  tx_buff_len = 0;
	uint8_t *buf = (uint8_t*)snapshot_buff;
	
	//copy in preambles to initiate photo transfer
  memcpy(tx_buff, &PREAMBLE, sizeof(PREAMBLE)-1);
	//keep track of length for uart_bin send
	tx_buff_len += sizeof(PREAMBLE) -1;
	
	//get every other index to obtain Y byte
	for(int i = 1; i < snapshot_length*2; i+=2){
		tx_buff[tx_buff_len++] = buf[i];
	}
	//append suffix
  memcpy(&tx_buff[tx_buff_len], &SUFFIX, sizeof(SUFFIX)-1);
	tx_buff_len+= sizeof(SUFFIX)-1;
	
	//resume DCMI (suspended in f4xx it interrupt handler)
	HAL_DCMI_Resume(&hdcmi);
	//send bin
	uart_send_bin(tx_buff,tx_buff_len);

}

void send_trunc() {
    tx_buff_len = 0;
    uint8_t *buf = (uint8_t*)snapshot_buff;
	
		
    memcpy(tx_buff, PREAMBLE, sizeof(PREAMBLE)-1);
    tx_buff_len += sizeof(PREAMBLE) -1;

    // Truncate and pack every two pixels into one byte
    for (int i = 1; i < snapshot_length * 2; i += 4) { // Increment by 4 to process every two pixels (take care of i+2 in pixel2)
        // Truncate by shifting out last 4 and ORING together 4MSB of both pixels into one byte 
        uint8_t pixel1 = (buf[i] >> 4) & 0x0F; // extract and truncate the first pixel
        uint8_t pixel2 = (buf[i + 2] >> 4) & 0x0F; // extract and truncate the next pixel
        tx_buff[tx_buff_len++] = (pixel1 << 4) | pixel2; // OR INTO ONE BYTE
    }
		
		//append suffix
    memcpy(&tx_buff[tx_buff_len], SUFFIX, sizeof(SUFFIX)-1);
    tx_buff_len += sizeof(SUFFIX)-1;

    HAL_DCMI_Resume(&hdcmi);

    uart_send_bin(tx_buff, tx_buff_len);
}

//Helper for 6.3 combine truncation and RLE
void trunc_rle() {
		//user iter to keep track of "length" (where i should index)
    int len = 0;
    uint8_t *buf = (uint8_t*)snapshot_buff;
	
	
	
    memcpy(tx_buff, PREAMBLE, sizeof(PREAMBLE) - 1); 
    len += sizeof(PREAMBLE) - 1;
    
    
    for (int i = 1; i < snapshot_length * 2; i += 2) {
        uint8_t pixel = buf[i] & 0xF0; // Current truncated pixel
        int count = 1;
        // Find repeating pixels
																						//IF NEXT PIXELS = CURRENT PIXELS INCREMEMNT COUNT (RLE)
        while (i < snapshot_length * 2 - 2 && (buf[i + 2] & 0xF0) == pixel) { //TRUNCATED PIXEL 2 MSB
            count++;
            i += 2;
            if (count == 0xF) break; // MOST CONSECUTIVE SAME PIXELS WE CAN REPRESENT
        }

        // Encode runs
				//CHECK IF WE ARE UNDER THE MAX WE CAN REPRESENT ELSE WE NEED TO USE MUTIPLE RUNS (KEEP LOOPING)
        while (count > 0xF) {
            tx_buff[len++] = pixel | 0xF; // Encode pixel with max count
            count -= 0xF;
        }
        tx_buff[len++] = pixel | count; // Encode remaining pixels
    }
		// UPDATE OLD SNAPSHOT BUFF (USEFUL FOR COMPRESSION ALGO)
		for (int i = 1; i < snapshot_length*2; i+=2){
			old_snapshot_buff[i/2] = buf[i];
		}
    
    memcpy(&tx_buff[len], SUFFIX, sizeof(SUFFIX) - 1); // Assuming SUFFIX does not include null terminator
    len += sizeof(SUFFIX) - 1;
    
    HAL_DCMI_Resume(&hdcmi);
    
    uart_send_bin(tx_buff, len); // iter is already the correct length
}

void send_delta(){
	
  int len = 0;
	uint8_t *buf = (uint8_t*)snapshot_buff;
  memcpy(tx_buff, &DELTA_PREAMBLE, sizeof(DELTA_PREAMBLE));
	len += sizeof(DELTA_PREAMBLE);
	
	//GET DIFFERENCE
	for (int i = 1; i < snapshot_length*2; i+=2){
		//SUBTRACT PIXEL DIFFERENCE OF OLD BUFFER
		int val = (0xF0 & buf[i]) - (0xF0 &old_snapshot_buff[i/2]);
		old_snapshot_buff[i/2] = buf[i];
		buf[i/2] = val;
	}

	
	// SIMILAR CODE FOR TRUNCATING AND RLE BECAUSE IM TOO LAZY TO WRITE A CONVENIENT FUNCTION
	
    for (int i = 1; i < snapshot_length * 2; i += 2) {
        uint8_t pixel = buf[i] & 0xF0; // Current truncated pixel
        int count = 1;
        // Find repeating pixels
																						//IF NEXT PIXELS = CURRENT PIXELS INCREMEMNT COUNT (RLE)
        while (i < snapshot_length * 2 - 2 && (buf[i + 2] & 0xF0) == pixel) { //TRUNCATED PIXEL 2 MSB
            count++;
            i += 2;
            if (count == 0xF) break; // MOST CONSECUTIVE SAME PIXELS WE CAN REPRESENT
        }

        // Encode runs
				//CHECK IF WE ARE UNDER THE MAX WE CAN REPRESENT ELSE WE NEED TO USE MUTIPLE RUNS (KEEP LOOPING)
        while (count > 0xF) {
            tx_buff[len++] = pixel | 0xF; // Encode pixel with max count
            count -= 0xF;
        }
        tx_buff[len++] = pixel | count; // Encode remaining pixels
    }
	
	memcpy(&tx_buff[len], &SUFFIX, sizeof(SUFFIX));
	HAL_DCMI_Resume(&hdcmi);
	
	uart_send_bin(tx_buff,len+sizeof(SUFFIX));
	
}
	
int main(void)
{
  /* Reset of all peripherals */
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_DCMI_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();
  MX_I2C2_Init();
  MX_TIM1_Init();
  MX_TIM6_Init();

  char msg[100];

  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
  ov7670_init();
  HAL_Delay(100);
  ov7670_capture(snapshot_buff);

  // Your startup code here
	//PLEASE UNCOMMENT OUT THE PART U WANT TO TEST AND COMMENT WHAT YOU ARENT TESTING

	/*
	////////////////////////////////////////////////////////////
	///////PART 2.1 - SENDING IMAGES THROUGH SERIAL PORT////////
	////////////////////////////////////////////////////////////
    // Copy PREAMBLE to tx_buff
    memcpy(tx_buff, PREAMBLE, sizeof(PREAMBLE) - 1);
    tx_buff_len += sizeof(PREAMBLE) - 1;

    // Set middle section to 0x00
    memset(tx_buff + tx_buff_len, 0x00, 2 * IMG_ROWS * IMG_COLS);
    tx_buff_len += 2 * IMG_ROWS * IMG_COLS;

    // Copy SUFFIX to tx_buff
    memcpy(tx_buff + tx_buff_len, SUFFIX, sizeof(SUFFIX) - 1);
    tx_buff_len += sizeof(SUFFIX) - 1;
	
		uart_send_bin(tx_buff, tx_buff_len);
		
		//////////////////////////////////////////////////////////
		//////////////////////PART 2.1 END////////////////////////
		//////////////////////////////////////////////////////////
		*/
		//note this was turned into the function send_image() for convenience

	
  while (1)
  {
		/*
		////////// PART 3.1 - TESTING READ FROM OV REGISTER ///////
		uint8_t result = ov7670_read(0x0A);
		if(result == 0x76){
			print_msg("NICE");
			HAL_Delay(1000);
		}
		////////////////////// PART 3.1 END //////////////////////
		*/
		
		
		/////////// PART 3.4  - CAPTURING ONE FRAME //////////////
		
    if (HAL_GPIO_ReadPin(USER_Btn_GPIO_Port, USER_Btn_Pin)) {
      HAL_Delay(100);  // debounce
			ov7670_snapshot(snapshot_buff);
			
		//IF DMA INT HANDLER SETS THIS TO 1, WAIT FOR THAT
			while(!dma_flag){
				HAL_Delay(1);
			}
			//clear dcmi flag
			dma_flag = 0;
			
			//send image (snapshot buff is global dont have to pass)
			send_image();
			
      print_msg("Snap!\r\n");
    }
		
		
		////////////////////// PART 3.4 END //////////////////////
		
		
		
		//Code to take snapshot continuosly, clear dma flag
		/*
		ov7670_snapshot(snapshot_buff);
		while(!dma_flag){
				HAL_Delay(300);  //try a smaller delay for video to limit frame lag
		}
		dma_flag = 0;
		*/
		
		
		///////////////// PART 4.2  - CTS IMAGE SEND////////////
		//already in while 1:
		//uart with dma only (for table)
		//send_image();
		
		///////////////////// PART 4.2 END ///////////////////////
		
		
		/////////////// PART 6.2 JUST TRUNCATION ////////////
		//4 bit pixel part on table (for testing)
		//send_trunc();
		
		////////////////////PART 6.2 END /////////////////////////
		
		
		//////////////PART 6.3 TRUNCATION & RLE //////////////////
		
		//trunc_rle();
		
		///////////////// PART 6.3 END ///////////////////////////
		
		
		//////// PART 7 - VIDEO COMPRESSION ALGORITHM ////////////
		
		
		//IF WE ARE ON FIRST FRAME OR EVERY N FRAMES WE SEND A START FRAME
		/*
		if (frame_count == 1 || frame_count % FULL_FRAME_INTERVAL == 0){
			trunc_rle();
		}
		//OTHERWISE WE ARE SENDING A DELTA FRAME
		else {
			send_delta();
		}
		*/
		

		/////////////////// PART 7 END //////////////////////////
		
  }
}


void print_buf() {
  // Send image data through serial port.
  // Your code here
}